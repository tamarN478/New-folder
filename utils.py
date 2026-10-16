"""utility functions for loading, cleaning, and exporting datasets."""

import io

import pandas as pd
import plotly.express as px
import streamlit as st

MAX_UNDO_STEPS = 20

CLEANING_ACTIONS = [
    "Drop Duplicates",
    "Drop Rows with Missing Values",
    "Fill Missing Values",
    "Change Column Type",
    "Rename Column",
    "Drop Columns",
    "Remove Outliers (IQR)",
    "Trim Spaces",
    "Fix Capitalization",
]


def load_data(uploaded_file):
    """loads a csv or excel file into a pandas dataframe."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)


def _students_sample():
    """small hand-made dataset that mirrors what people type in manually."""
    return pd.DataFrame(
        {
            "faculty": ["Computer Science", "Math", "Biology", "Physics", "Computer Science",
                        "Math", "Biology", "Physics", "Computer Science", "Biology"],
            "year": [3, 2, 4, 1, 1, 4, 2, 3, 4, 1],
            "gpa": [3.6, 3.2, 3.8, 2.9, 3.1, 3.9, 3.4, 3.0, 3.7, 2.8],
            "study_hours": [20, 15, 25, 10, 12, 28, 18, 14, 24, 9],
            "city": ["Tbilisi", "Batumi", "Tbilisi", "Kutaisi", "Batumi",
                     "Tbilisi", "Kutaisi", "Tbilisi", "Batumi", "Tbilisi"],
        }
    )


def _faculties_sample():
    """small lookup table that joins with the Students sample on 'faculty'."""
    return pd.DataFrame(
        {
            "faculty": ["Computer Science", "Math", "Biology", "Physics", "Chemistry"],
            "building": ["North Hall", "Main Building", "Life Sciences", "Main Building", "Lab Block"],
            "founded": [1985, 1918, 1930, 1918, 1952],
            "students_enrolled": [1200, 640, 910, 480, 520],
        }
    )


SAMPLE_DATASETS = {
    "Students": _students_sample,
    "Faculties": _faculties_sample,
    "Restaurant tips": px.data.tips,
    "Iris flowers": px.data.iris,
    "Gapminder (countries over time)": px.data.gapminder,
}


# ---------------------------------------------------------------------------
# dataset state: the active dataset, its original version, and undo history
# ---------------------------------------------------------------------------

def set_new_dataset(df, source_name):
    """make df the active dataset, starting a fresh history for it."""
    st.session_state.raw_df = df
    st.session_state.original_df = df.copy()
    st.session_state.history = []
    st.session_state.dataset_source = source_name


def apply_change(new_df, description):
    """replace the active dataset and remember the previous version for undo."""
    history = st.session_state.setdefault("history", [])
    history.append((description, st.session_state.raw_df))
    del history[:-MAX_UNDO_STEPS]
    st.session_state.raw_df = new_df


def undo_last_change():
    """restore the dataset as it was before the most recent change."""
    history = st.session_state.get("history", [])
    if not history:
        return None
    description, previous_df = history.pop()
    st.session_state.raw_df = previous_df
    return description


def reset_to_original():
    """throw away every change and go back to the dataset as it was loaded."""
    if "original_df" not in st.session_state:
        return
    apply_change(st.session_state.original_df.copy(), "Reset to original")


# ---------------------------------------------------------------------------
# exporting
# ---------------------------------------------------------------------------

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buffer.getvalue()


def download_buttons(df, file_stem="dataset", key="download"):
    """render side-by-side csv and excel download buttons for df."""
    csv_col, excel_col = st.columns(2)
    csv_col.download_button(
        "Download CSV",
        to_csv_bytes(df),
        file_name=f"{file_stem}.csv",
        mime="text/csv",
        key=f"{key}_csv",
        icon=":material/download:",
        width="stretch",
    )
    excel_col.download_button(
        "Download Excel",
        to_excel_bytes(df),
        file_name=f"{file_stem}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{key}_excel",
        icon=":material/download:",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

def _fill_missing(df, columns, method, custom_value):
    for column in columns:
        series = df[column]
        if method == "Mean":
            fill = pd.to_numeric(series, errors="coerce").mean()
        elif method == "Median":
            fill = pd.to_numeric(series, errors="coerce").median()
        elif method == "Most Common Value (Mode)":
            modes = series.mode()
            fill = modes.iloc[0] if not modes.empty else None
        else:
            fill = custom_value
            if pd.api.types.is_numeric_dtype(series):
                fill = pd.to_numeric(custom_value, errors="coerce")
        if fill is not None and not pd.isna(fill):
            df[column] = series.fillna(fill)
    return df


def _change_type(df, columns, target_type):
    for column in columns:
        if target_type == "Number":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif target_type == "Text":
            df[column] = df[column].astype("string")
        elif target_type == "Date":
            df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")
        elif target_type == "Category":
            df[column] = df[column].astype("category")
    return df


def _remove_outliers(df, columns):
    keep = pd.Series(True, index=df.index)
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        spread = q3 - q1
        inside = values.between(q1 - 1.5 * spread, q3 + 1.5 * spread) | values.isna()
        keep &= inside
    return df[keep]


def _text_columns(df, columns):
    return [
        column for column in columns
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])
    ]


def clean_data(df, action, columns=None, option=None, value=None):
    """performs data cleaning operations on the dataframe.

    columns: the columns the action applies to (all columns when empty).
    option:  the action's choice, e.g. the fill method or target type.
    value:   free-form input, e.g. a custom fill value or a new column name.
    """
    cleaned_df = df.copy()
    columns = list(columns) if columns else list(cleaned_df.columns)

    if action == "Drop Duplicates":
        cleaned_df = cleaned_df.drop_duplicates()
    elif action == "Drop Rows with Missing Values":
        cleaned_df = cleaned_df.dropna(subset=columns)
    elif action == "Fill Missing Values":
        cleaned_df = _fill_missing(cleaned_df, columns, option, value)
    elif action == "Change Column Type":
        cleaned_df = _change_type(cleaned_df, columns, option)
    elif action == "Rename Column" and value:
        cleaned_df = cleaned_df.rename(columns={columns[0]: value})
    elif action == "Drop Columns":
        cleaned_df = cleaned_df.drop(columns=columns)
    elif action == "Remove Outliers (IQR)":
        cleaned_df = _remove_outliers(cleaned_df, columns)
    elif action == "Trim Spaces":
        for column in _text_columns(cleaned_df, columns):
            cleaned_df[column] = cleaned_df[column].str.strip()
    elif action == "Fix Capitalization":
        for column in _text_columns(cleaned_df, columns):
            text = cleaned_df[column].str
            cleaned_df[column] = {
                "lowercase": text.lower,
                "UPPERCASE": text.upper,
                "Title Case": text.title,
            }[option]()

    return cleaned_df.reset_index(drop=True)

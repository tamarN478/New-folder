"""eda and data cleaning page."""

import pandas as pd
import streamlit as st
from utils import (
    CLEANING_ACTIONS,
    apply_change,
    clean_data,
    download_buttons,
    reset_to_original,
    undo_last_change,
)

st.title("📊 Exploratory Data Analysis & Data Cleaning")

if "raw_df" not in st.session_state or st.session_state.raw_df.empty:
    st.warning("Please upload a dataset on the main Home page first!")
else:
    df = st.session_state.raw_df

    # Sidebar Filtering
    st.sidebar.subheader("Filter Dataset")
    enable_filtering = st.sidebar.checkbox("Enable Data Filtering")

    filtered_df = df.copy()
    if enable_filtering:
        filter_column = st.sidebar.selectbox("Select Column to Filter", options=df.columns.tolist())
        column_values = df[filter_column]
        if pd.api.types.is_numeric_dtype(column_values) and column_values.notna().any():
            low, high = float(column_values.min()), float(column_values.max())
            if low < high:
                selected_range = st.sidebar.slider(
                    "Select Range", min_value=low, max_value=high, value=(low, high)
                )
                filtered_df = df[column_values.between(*selected_range)]
            else:
                st.sidebar.caption(f"Every value in this column is {low:g}.")
        else:
            unique_vals = column_values.dropna().unique()
            selected_vals = st.sidebar.multiselect(
                "Select Values", options=unique_vals, default=list(unique_vals)
            )
            if selected_vals:
                filtered_df = df[column_values.isin(selected_vals)]
        st.sidebar.markdown(f"**Filtered Rows:** {filtered_df.shape[0]} of {df.shape[0]}")

    tab_eda, tab_cleaning = st.tabs(["📊 Data Overview", "🧹 Data Cleaning Suite"])

    with tab_eda:
        st.subheader("Data Preview")
        st.dataframe(filtered_df.head(), width="stretch")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", filtered_df.shape[0])
        col2.metric("Total Columns", filtered_df.shape[1])
        col3.metric("Missing Cells", int(filtered_df.isnull().sum().sum()))

        with st.expander("Summary Statistics"):
            # Mixed numbers and text per column; show as text so the table renders cleanly.
            summary = filtered_df.describe(include="all")
            st.dataframe(summary.astype(object).where(summary.notna(), "").astype(str), width="stretch")

        with st.expander("Missing Values Overview"):
            missing = filtered_df.isnull().sum()
            missing_pct = (missing / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            missing_df = pd.DataFrame({"Missing Values": missing, "Percentage (%)": missing_pct})
            st.dataframe(missing_df, width="stretch")

        st.subheader("Download Data")
        if enable_filtering:
            st.caption("Downloads include only the rows that match your filter.")
        download_buttons(filtered_df, file_stem="cleaned_data", key="eda_download")

    with tab_cleaning:
        st.subheader("Interactive Data Cleaning")

        history = st.session_state.get("history", [])
        undo_col, reset_col, info_col = st.columns([1, 1, 2])
        if undo_col.button("↩ Undo last step", disabled=not history, width="stretch"):
            undone = undo_last_change()
            st.toast(f"Undid: {undone}")
            st.rerun()
        if reset_col.button("⟲ Reset to original", disabled=not history, width="stretch"):
            reset_to_original()
            st.toast("Restored the original data. You can still undo this.")
            st.rerun()
        if history:
            info_col.caption(
                "Steps so far: " + " → ".join(description for description, _ in history)
            )

        if "cleaning_message" in st.session_state:
            st.success(st.session_state.pop("cleaning_message"))

        cleaning_action = st.selectbox("Select Cleaning Action", ["None"] + CLEANING_ACTIONS)

        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        columns, option, value = None, None, None
        ready = cleaning_action != "None"

        if cleaning_action == "Drop Duplicates":
            st.caption("Removes rows that are exact copies of an earlier row.")
        elif cleaning_action == "Drop Rows with Missing Values":
            columns = st.multiselect(
                "Only look at these columns (leave empty for all)", all_columns
            )
        elif cleaning_action == "Fill Missing Values":
            columns = st.multiselect(
                "Columns to fill", all_columns,
                default=[c for c in all_columns if df[c].isnull().any()],
            )
            option = st.radio(
                "Fill with",
                ["Mean", "Median", "Most Common Value (Mode)", "Custom Value"],
                horizontal=True,
                help="Mean and Median only work on numeric columns.",
            )
            if option == "Custom Value":
                value = st.text_input("Value to fill in")
                ready = bool(value)
            ready = ready and bool(columns)
        elif cleaning_action == "Change Column Type":
            columns = st.multiselect("Columns", all_columns)
            option = st.radio("New type", ["Number", "Text", "Date", "Category"], horizontal=True)
            st.caption("Values that can't be converted become empty (missing).")
            ready = bool(columns)
        elif cleaning_action == "Rename Column":
            columns = [st.selectbox("Column", all_columns)]
            value = st.text_input("New name").strip()
            if value in all_columns:
                st.error(f"A column named '{value}' already exists.")
            ready = bool(value) and value not in all_columns
        elif cleaning_action == "Drop Columns":
            columns = st.multiselect("Columns to remove", all_columns)
            ready = bool(columns) and len(columns) < len(all_columns)
        elif cleaning_action == "Remove Outliers (IQR)":
            columns = st.multiselect("Numeric columns", numeric_cols, default=numeric_cols[:1])
            st.caption(
                "Removes rows with values far outside the typical range "
                "(more than 1.5× the interquartile range beyond the middle 50%)."
            )
            ready = bool(columns)
        elif cleaning_action == "Trim Spaces":
            columns = st.multiselect("Text columns", text_cols, default=text_cols)
            st.caption("Removes extra spaces at the start and end, e.g. ' Math ' → 'Math'.")
            ready = bool(columns)
        elif cleaning_action == "Fix Capitalization":
            columns = st.multiselect("Text columns", text_cols, default=text_cols)
            option = st.radio("Style", ["Title Case", "lowercase", "UPPERCASE"], horizontal=True)
            ready = bool(columns)

        if st.button("Apply Cleaning Action", type="primary", disabled=not ready):
            try:
                cleaned = clean_data(df, cleaning_action, columns, option, value)
            except Exception as error:
                st.error(f"Couldn't apply '{cleaning_action}': {error}")
            else:
                apply_change(cleaned, cleaning_action)
                removed = len(df) - len(cleaned)
                detail = f" Removed {removed} row(s)." if removed else ""
                st.session_state.cleaning_message = f"Applied '{cleaning_action}'.{detail}"
                st.rerun()

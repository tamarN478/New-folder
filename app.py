"""Main entrypoint for the multi-page Streamlit data app."""

import io

import pandas as pd
import streamlit as st
from utils import SAMPLE_DATASETS, load_data, set_new_dataset


st.set_page_config(
    page_title="AI Data Explorer",
    page_icon="🚀",
    layout="wide",
)


def _column_letter(position: int) -> str:
    """Spreadsheet-style label for a column position: 0 -> A, 25 -> Z, 26 -> AA."""
    letters = ""
    position += 1
    while position:
        position, remainder = divmod(position - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _new_grid(names: list[str], empty_rows: int = 3) -> pd.DataFrame:
    """Build a manual-entry grid whose first row holds the column names."""
    rows = [list(names)] + [[None] * len(names) for _ in range(empty_rows)]
    grid = pd.DataFrame(rows, dtype="object")
    grid.columns = [_column_letter(i) for i in range(len(names))]
    return grid


def _grid_to_df(grid: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Turn the grid into a dataset named by its first row, or explain what's wrong."""
    names = [str(value).strip() if pd.notna(value) else "" for value in grid.iloc[0]]
    for letter, name in zip(grid.columns, names):
        if not name:
            return None, f"Column {letter} needs a name in the first row."
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        return None, f"Two columns can't have the same name: {', '.join(duplicates)}."
    df = grid.iloc[1:].copy()
    df.columns = names
    return df.reset_index(drop=True), None


def _commit_manual_grid(new_grid: pd.DataFrame) -> None:
    """Replace the manual grid's data and remount the editor with it."""
    new_grid = new_grid.reset_index(drop=True)
    new_grid.columns = [_column_letter(i) for i in range(len(new_grid.columns))]
    st.session_state.manual_grid = new_grid
    st.session_state.manual_grid_draft = new_grid
    # A new key gives a fresh editor, so it shows the new data without stale edits.
    st.session_state.manual_editor_version += 1
    st.rerun()


def _to_analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows and turn columns that hold only numbers into numeric columns."""
    df = df.replace("", None).dropna(how="all").reset_index(drop=True)
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() == df[column].notna().sum():
            df[column] = converted
    return df


def manual_data_entry() -> None:
    """Render an editable grid and store its contents as the active dataset."""
    st.subheader("Enter data manually")
    st.write(
        "The **first row holds the column names**: click a name and type to change it. "
        "Enter your data in the rows below. Your changes are saved automatically."
    )

    if "manual_editor_version" not in st.session_state:
        st.session_state.manual_editor_version = 0
    if "manual_hidden" not in st.session_state:
        st.session_state.manual_hidden = []
    if "manual_grid" not in st.session_state:
        st.session_state.manual_grid = _new_grid(["Column 1", "Column 2"])

    editor_key = f"manual_editor_{st.session_state.manual_editor_version}"

    # Streamlit drops widget state when you visit another page, so when the
    # editor is mounted again, start it from the latest saved draft.
    if editor_key not in st.session_state and "manual_grid_draft" in st.session_state:
        st.session_state.manual_grid = st.session_state.manual_grid_draft

    source = st.session_state.manual_grid
    hidden = [
        letter
        for letter, name in zip(source.columns, source.iloc[0])
        if name in st.session_state.manual_hidden
    ]
    visible = [letter for letter in source.columns if letter not in hidden]

    # The data passed in must stay the same between reruns; feeding the edited
    # result back in resets the editor and makes typed values disappear.
    # Hidden columns are left out of the view but are still returned.
    edited_grid = st.data_editor(
        source,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_order=visible,
        key=editor_key,
    )

    # Deleting the first row would turn the next data row into column names,
    # so put the names back and keep the rest of the edits.
    if 0 in st.session_state[editor_key].get("deleted_rows", []):
        st.toast("The first row holds the column names, so it can't be deleted.")
        _commit_manual_grid(pd.concat([source.iloc[[0]], edited_grid], ignore_index=True))

    st.session_state.manual_grid_draft = edited_grid
    names = [str(value).strip() if pd.notna(value) else "" for value in edited_grid.iloc[0]]
    column_labels = [f"{letter}: {name}" for letter, name in zip(edited_grid.columns, names)]

    row_col, add_col, hide_col, delete_col, paste_col, clear_col = st.columns(6)

    with row_col:
        if st.button("＋ Add row", width="stretch"):
            empty_row = pd.DataFrame([[None] * len(edited_grid.columns)], columns=edited_grid.columns)
            _commit_manual_grid(pd.concat([edited_grid, empty_row], ignore_index=True))

    with add_col:
        with st.popover("＋ Add column", width="stretch"):
            with st.form("add_manual_column", clear_on_submit=True, border=False):
                new_column = st.text_input("Column name").strip()
                place_after = st.selectbox(
                    "Place after",
                    range(len(column_labels)),
                    index=len(column_labels) - 1,
                    format_func=lambda i: column_labels[i],
                )
                if st.form_submit_button("Add"):
                    if not new_column:
                        st.error("Please enter a column name.")
                    elif new_column in names:
                        st.error(f"Column '{new_column}' already exists.")
                    else:
                        new_grid = edited_grid.copy()
                        new_grid.insert(
                            place_after + 1,
                            "new",
                            pd.Series([new_column] + [None] * (len(new_grid) - 1), dtype="object"),
                        )
                        _commit_manual_grid(new_grid)

    with hide_col:
        with st.popover("👁 Hide columns", width="stretch"):
            with st.form("hide_manual_columns", border=False):
                to_hide = st.multiselect(
                    "Columns to hide",
                    names,
                    default=[name for name in st.session_state.manual_hidden if name in names],
                    max_selections=len(names) - 1,
                    help="Hidden columns keep their data and are still used in analysis.",
                )
                if st.form_submit_button("Apply"):
                    st.session_state.manual_hidden = to_hide
                    _commit_manual_grid(edited_grid)

    with delete_col:
        with st.popover("🗑 Delete column", width="stretch"):
            to_delete = st.selectbox(
                "Column", range(len(column_labels)), format_func=lambda i: column_labels[i]
            )
            if st.button("Delete", disabled=len(column_labels) <= 1):
                _commit_manual_grid(edited_grid.drop(columns=[edited_grid.columns[to_delete]]))

    with paste_col:
        with st.popover("📋 Paste from Excel", width="stretch"):
            with st.form("paste_manual_data", border=False):
                st.caption(
                    "Copy cells in Excel or Google Sheets (Ctrl+C), paste them here (Ctrl+V). "
                    "This replaces the current table."
                )
                pasted = st.text_area("Pasted cells", height=150)
                has_header = st.checkbox("First line has the column names", value=True)
                if st.form_submit_button("Load into table"):
                    pasted_grid = _parse_pasted_cells(pasted, has_header)
                    if pasted_grid is None:
                        st.error("Couldn't read any cells. Paste at least one line of data.")
                    else:
                        st.session_state.manual_hidden = []
                        _commit_manual_grid(pasted_grid)

    with clear_col:
        with st.popover("🧹 Clear table", width="stretch"):
            st.write("This deletes all data and column names and starts over.")
            if st.button("Yes, clear everything", type="primary"):
                st.session_state.manual_hidden = []
                _commit_manual_grid(_new_grid(["Column 1", "Column 2"]))

    manual_df, error = _grid_to_df(edited_grid)
    if error:
        st.error(error)
        st.session_state.setdefault("raw_df", pd.DataFrame())
        return

    # Only replace the active dataset when the table actually changed, so
    # cleaning done on the EDA page survives a visit back to this page.
    analysis_df = _to_analysis_df(manual_df)
    previous = st.session_state.get("manual_pushed_df")
    if (
        st.session_state.get("dataset_source") != "manual"
        or previous is None
        or not analysis_df.equals(previous)
    ):
        set_new_dataset(analysis_df, "manual")
        st.session_state.manual_pushed_df = analysis_df


def _parse_pasted_cells(text: str, has_header: bool) -> pd.DataFrame | None:
    """Turn cells copied from a spreadsheet (tab-separated) into a manual-entry grid."""
    if not text.strip():
        return None
    separator = "\t" if "\t" in text else ","
    try:
        rows = pd.read_csv(
            io.StringIO(text.strip("\n")),
            sep=separator,
            header=None,
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=True,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    rows = rows.replace("", None)
    if has_header:
        names = [
            str(name).strip() if name else f"Column {position + 1}"
            for position, name in enumerate(rows.iloc[0])
        ]
        body = rows.iloc[1:]
    else:
        names = [f"Column {position + 1}" for position in range(len(rows.columns))]
        body = rows
    grid = pd.DataFrame([names] + body.values.tolist(), dtype="object")
    return grid.astype(object).where(grid.notna(), None)


def home_page() -> None:
    """Render the home page and dataset uploader."""
    st.title("Welcome to the AI Data Analysis Application 🚀")
    st.write("This application helps you explore, clean, visualize, and analyze datasets with ease.")

    with st.sidebar:
        st.header("📂 Dataset Management")
        input_method = st.radio(
            "Choose how to provide your data",
            ["Upload a file", "Enter data manually", "Try a sample dataset"],
            key="data_input_method",
        )

    if input_method == "Upload a file":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            key="main_uploader",
        )

        if uploaded_file is None:
            st.info("Use the file uploader in the sidebar to get started.")
            return

        source = f"file:{uploaded_file.name}:{uploaded_file.size}"
        try:
            if st.session_state.get("dataset_source") != source:
                set_new_dataset(load_data(uploaded_file), source)
            st.sidebar.success("Dataset loaded successfully!")
        except Exception as error:
            st.sidebar.error(f"Error loading file: {error}")
            return

        st.success(f"Active File: **{uploaded_file.name}**")
    elif input_method == "Try a sample dataset":
        sample_name = st.sidebar.selectbox("Sample dataset", list(SAMPLE_DATASETS))
        source = f"sample:{sample_name}"
        if st.session_state.get("dataset_source") != source:
            set_new_dataset(SAMPLE_DATASETS[sample_name](), source)
        st.success(f"Active Sample: **{sample_name}**. Open a page from the sidebar to explore it.")
    else:
        manual_data_entry()
        if st.session_state.raw_df.empty:
            st.info("Type some values into the table to start analyzing.")
            return
        st.success("Manual data is ready for the charts and analysis tools.")

    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.raw_df.head(), width="stretch")
    if st.session_state.get("history"):
        st.caption(
            f"ℹ️ This data has {len(st.session_state.history)} cleaning step(s) applied "
            "on the EDA & Cleaning page."
        )


pages = {
    "Main": [
        st.Page(home_page, title="Home & Upload", icon="🏠", default=True),
    ],
    "Analysis Tools": [
        st.Page("pages/1_📊_EDA_and_Cleaning.py", title="EDA & Cleaning", icon="📊"),
        st.Page("pages/2_📈_Chart_Builder.py", title="Chart Builder", icon="📈"),
        st.Page("pages/3_💡_AI_Insights.py", title="AI Insights", icon="💡"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
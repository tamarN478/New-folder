"""home page: load a dataset and see an overview of it."""

import streamlit as st
from manual_entry import manual_data_entry
from ui import describe_source, page_header
from utils import SAMPLE_DATASETS, load_data, set_new_dataset

INPUT_METHODS = {
    "upload": ":material/upload_file: Upload a file",
    "manual": ":material/edit_note: Enter data manually",
    "sample": ":material/dataset: Try a sample dataset",
}
NEXT_STEPS = [
    (":material/query_stats:", "Explore & clean",
     "Summary statistics, automatic charts, filters, and one-click cleaning tools with undo.",
     "pages/eda_cleaning.py"),
    (":material/bar_chart:", "Build charts",
     "Bar, line, scatter, pie, box plots and heatmaps, with grouping by average or sum.",
     "pages/chart_builder.py"),
    (":material/auto_awesome:", "Ask AI",
     "Ask questions in plain language or get an instant summary of your dataset.",
     "pages/ai_insights.py"),
]


def load_from_upload():
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file", type=["csv", "xlsx", "xls"], key="main_uploader"
    )
    if uploaded_file is None:
        return
    source = f"file:{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("dataset_source") == source:
        return
    try:
        set_new_dataset(load_data(uploaded_file), source)
    except Exception as error:
        st.error(f"Couldn't read this file: {error}", icon=":material/error:")
    else:
        st.rerun()  # refresh the sidebar's dataset card


def load_from_samples():
    names = list(SAMPLE_DATASETS)
    current = st.session_state.get("dataset_source", "")
    current_name = current.split(":", 1)[1] if current.startswith("sample:") else None
    sample_name = st.selectbox(
        "Sample dataset",
        names,
        index=names.index(current_name) if current_name in names else 0,
    )
    source = f"sample:{sample_name}"
    if st.session_state.get("dataset_source") != source:
        set_new_dataset(SAMPLE_DATASETS[sample_name](), source)
        st.rerun()  # refresh the sidebar's dataset card


def dataset_overview(df):
    st.subheader("Your dataset")
    st.caption(describe_source(st.session_state.get("dataset_source")))
    metrics = st.container(horizontal=True)
    metrics.metric("Rows", f"{len(df):,}", border=True, icon=":material/table_rows:")
    metrics.metric("Columns", len(df.columns), border=True, icon=":material/view_column:")
    metrics.metric("Numeric columns", len(df.select_dtypes("number").columns), border=True,
                   icon=":material/numbers:")
    metrics.metric("Missing cells", f"{int(df.isnull().sum().sum()):,}", border=True,
                   icon=":material/help_outline:")
    st.dataframe(df.head(10), width="stretch")
    steps = len(st.session_state.get("history", []))
    if steps:
        st.caption(f"{steps} cleaning step(s) applied on the Explore & Clean page.")


def next_steps():
    st.subheader("What you can do")
    for column, (icon, title, text, page) in zip(st.columns(len(NEXT_STEPS)), NEXT_STEPS):
        with column.container(border=True, height="stretch"):
            st.markdown(f"#### {icon} {title}")
            st.caption(text)
            st.page_link(page, label="Open", icon=":material/arrow_forward:")


page_header(
    "AI Data Explorer",
    "Load a dataset, clean it, chart it, and ask AI about it, all in one place.",
)

st.subheader("Load your data")
if "data_input_method" not in st.session_state:
    st.session_state.data_input_method = st.session_state.get("input_method_choice", "upload")
method = st.segmented_control(
    "Data source",
    options=list(INPUT_METHODS),
    format_func=INPUT_METHODS.get,
    required=True,
    key="data_input_method",
    label_visibility="collapsed",
)
# The widget forgets its value when you visit another page, so remember it separately.
st.session_state.input_method_choice = method

with st.container(border=True):
    if method == "upload":
        load_from_upload()
    elif method == "sample":
        load_from_samples()
    else:
        manual_data_entry()

df = st.session_state.get("raw_df")
source = st.session_state.get("dataset_source") or ""
source_matches_tab = source.startswith({"upload": "file:", "sample": "sample:", "manual": "manual"}[method])
if df is not None and not df.empty:
    if source_matches_tab:
        dataset_overview(df)
    else:
        st.caption(
            f"Currently using **{describe_source(source)}**. It stays active until you "
            "load something here."
        )
next_steps()

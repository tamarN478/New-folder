"""Streamlit application for uploading a dataset and performing basic EDA."""

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Dataset Explorer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_dataset(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Load a supported uploaded file into a DataFrame."""
    extension = Path(uploaded_file.name).suffix.lower()

    if extension == ".csv":
        return pd.read_csv(uploaded_file)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


st.title("Dataset Explorer")
st.write("Upload a CSV or Excel file to inspect its structure and summary statistics.")

with st.sidebar:
    st.header("Upload dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: .csv, .xlsx, and .xls",
    )

if uploaded_file is None:
    st.info("Use the file uploader in the sidebar to get started.")
else:
    try:
        df = load_dataset(uploaded_file)
    except (ValueError, pd.errors.ParserError, ImportError, OSError) as error:
        st.error(f"Could not load {uploaded_file.name}: {error}")
    else:
        st.header("Dataset Overview")

        row_count, column_count = df.shape
        metric_columns = st.columns(2)
        metric_columns[0].metric("Rows", f"{row_count:,}")
        metric_columns[1].metric("Columns", f"{column_count:,}")

        st.subheader("Data preview")
        st.dataframe(df.head(), use_container_width=True)

        with st.expander("Summary statistics"):
            st.dataframe(df.describe(include="all"), use_container_width=True)

        st.subheader("Missing values by column")
        missing_values = pd.DataFrame(
            {
                "Missing values": df.isna().sum(),
                "Missing percentage": (df.isna().mean() * 100).round(2),
            }
        )
        st.dataframe(missing_values, use_container_width=True)
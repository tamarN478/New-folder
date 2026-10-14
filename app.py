"""Main entrypoint for the multi-page Streamlit data app."""

import streamlit as st
from utils import load_data


st.set_page_config(
    page_title="AI Data Explorer",
    page_icon="🚀",
    layout="wide",
)


def home_page() -> None:
    """Render the home page and dataset uploader."""
    st.title("Welcome to the AI Data Analysis Application 🚀")
    st.write("This application helps you explore, clean, visualize, and analyze datasets with ease.")

    st.info(
        "👈 **Get Started:** Upload your CSV or Excel dataset using the sidebar on the left, "
        "then select a page from the sidebar menu to begin analyzing!"
    )

    with st.sidebar:
        st.header("📂 Dataset Management")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            key="main_uploader",
        )

    if uploaded_file is not None:
        try:
            if "raw_df" not in st.session_state:
                st.session_state.raw_df = load_data(uploaded_file)
            st.sidebar.success("Dataset loaded successfully!")

            st.success(f"Active File: **{uploaded_file.name}**")
            st.subheader("Dataset Preview")
            st.dataframe(st.session_state.raw_df.head(), width=1200)
        except Exception as error:
            st.sidebar.error(f"Error loading file: {error}")
    else:
        st.warning("No dataset uploaded yet. Please use the sidebar to upload a file.")


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
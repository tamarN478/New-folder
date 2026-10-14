"""eda and data cleaning page."""

import pandas as pd
import streamlit as st
from utils import clean_data

st.title("📊 Exploratory Data Analysis & Data Cleaning")

if "raw_df" not in st.session_state:
    st.warning("Please upload a dataset on the main Home page first!")
else:
    df = st.session_state.raw_df

    # Sidebar Filtering
    st.sidebar.subheader("Filter Dataset")
    enable_filtering = st.sidebar.checkbox("Enable Data Filtering")
    
    filtered_df = df.copy()
    if enable_filtering:
        filter_column = st.sidebar.selectbox("Select Column to Filter", options=df.columns.tolist())
        unique_vals = df[filter_column].dropna().unique()
        selected_vals = st.sidebar.multiselect("Select Values", options=unique_vals, default=list(unique_vals))
        if selected_vals:
            filtered_df = df[df[filter_column].isin(selected_vals)]
        st.sidebar.markdown(f"**Filtered Rows:** {filtered_df.shape[0]} of {df.shape[0]}")

    tab_eda, tab_cleaning = st.tabs(["📊 Data Overview", "🧹 Data Cleaning Suite"])

    with tab_eda:
        st.subheader("Data Preview")
        st.dataframe(filtered_df.head(), width=1200)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", filtered_df.shape[0])
        with col2:
            st.metric("Total Columns", filtered_df.shape[1])

        with st.expander("Summary Statistics"):
            st.dataframe(filtered_df.describe(include="all"), width=1200)

        with st.expander("Missing Values Overview"):
            missing = filtered_df.isnull().sum()
            missing_pct = (missing / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            missing_df = pd.DataFrame({"Missing Values": missing, "Percentage (%)": missing_pct})
            st.dataframe(missing_df, width=1200)

    with tab_cleaning:
        st.subheader("Interactive Data Cleaning")
        cleaning_action = st.selectbox(
            "Select Cleaning Action",
            ["None", "Drop Duplicates", "Drop Rows with Missing Values", "Fill Missing Numeric Values (Mean)"]
        )

        target_col = None
        if cleaning_action == "Fill Missing Numeric Values (Mean)":
            numeric_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()
            target_col = st.selectbox("Select Numeric Column", options=numeric_cols)

        if st.button("Apply Cleaning Action"):
            st.session_state.raw_df = clean_data(df, cleaning_action, target_col)
            st.success("Cleaning action applied successfully!")
            st.rerun()
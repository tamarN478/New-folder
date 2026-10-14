"""streamlit application for uploading a dataset, exploratory data analysis, chart builder, ai insights, cleaning, and export."""

from pathlib import Path
import plotly.express as px
import pandas as pd
import streamlit as st
from utils import load_data, clean_data

st.set_page_config(
    page_title="Dataset Explorer & Chart Builder",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Dataset Explorer & Interactive Chart Builder")

# Sidebar file uploader
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        # Load dataset using modular helper function
        if "raw_df" not in st.session_state:
            st.session_state.raw_df = load_data(uploaded_file)

        df = st.session_state.raw_df

        # --- SIDEBAR FILTERING ---
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

        # Tab navigation including the new Data Cleaning tab
        tab_eda, tab_cleaning, tab_charts, tab_insights, tab_export = st.tabs([
            "📊 EDA", 
            "🧹 Data Cleaning",
            "📈 Chart Builder", 
            "💡 AI Insights",
            "📥 Export"
        ])

        # --- TAB 1: EDA ---
        with tab_eda:
            st.subheader("Data Preview (Filtered)" if enable_filtering else "Data Preview")
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
                missing_df = pd.DataFrame(
                    {"Missing Values": missing, "Percentage (%)": missing_pct}
                )
                st.dataframe(missing_df, width=1200)

        # --- TAB 2: DATA CLEANING ---
        with tab_cleaning:
            st.subheader("Interactive Data Cleaning Suite")
            st.write("Modify and clean your dataset dynamically before building charts or exporting.")

            col_action, col_preview = st.columns([1, 2])

            with col_action:
                cleaning_action = st.selectbox(
                    "Select Cleaning Action",
                    [
                        "None",
                        "Drop Duplicates",
                        "Drop Rows with Missing Values",
                        "Fill Missing Numeric Values (Mean)"
                    ]
                )

                target_col = None
                if cleaning_action == "Fill Missing Numeric Values (Mean)":
                    numeric_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()
                    target_col = st.selectbox("Select Numeric Column", options=numeric_cols)

                if st.button("Apply Cleaning Action"):
                    st.session_state.raw_df = clean_data(df, cleaning_action, target_col)
                    st.success("Cleaning action applied successfully! Check the updated row counts.")
                    st.rerun()

            with col_preview:
                st.write("Current Dataset Status:")
                st.metric("Active Rows", filtered_df.shape[0])
                st.metric("Active Missing Cells", int(filtered_df.isnull().sum().sum()))

        # --- TAB 3: CHART BUILDER ---
        with tab_charts:
            st.subheader("Create Custom Visualizations")

            col_controls, col_display = st.columns([1, 2])
            all_columns = filtered_df.columns.tolist()

            with col_controls:
                chart_type = st.selectbox(
                    "Select Chart Type",
                    ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot"],
                )

                x_axis = st.selectbox("X-Axis Column", options=all_columns)
                y_axis = st.selectbox(
                    "Y-Axis Column",
                    options=[None] + all_columns,
                    index=1 if len(all_columns) > 1 else 0,
                )
                color_by = st.selectbox(
                    "Color / Group By (Optional)", options=[None] + all_columns
                )
                chart_title = st.text_input("Chart Title", f"{chart_type} of {x_axis}")

            with col_display:
                fig = None
                if chart_type == "Bar Chart" and y_axis:
                    fig = px.bar(filtered_df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
                elif chart_type == "Line Chart" and y_axis:
                    fig = px.line(filtered_df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
                elif chart_type == "Scatter Plot" and y_axis:
                    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
                elif chart_type == "Histogram":
                    fig = px.histogram(filtered_df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
                elif chart_type == "Box Plot":
                    fig = px.box(filtered_df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
                else:
                    st.warning("Please select a valid Y-axis column.")

                if fig:
                    st.plotly_chart(fig, width=1200)

        # --- TAB 4: AI INSIGHTS ---
        with tab_insights:
            st.subheader("Automated Dataset Insights")
            numeric_df = filtered_df.select_dtypes(include=["number"])

            if not numeric_df.empty and numeric_df.shape[1] > 1:
                corr_matrix = numeric_df.corr()
                st.markdown("#### 🔍 Correlation Highlights")
                found_corr = False
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        col_a = corr_matrix.columns[i]
                        col_b = corr_matrix.columns[j]
                        val = corr_matrix.iloc[i, j]
                        if abs(val) > 0.6:
                            found_corr = True
                            direction = "positive" if val > 0 else "negative"
                            st.success(f"Strong **{direction} correlation** ({val:.2f}) found between **{col_a}** and **{col_b}**.")
                
                if not found_corr:
                    st.info("No strong linear correlations (> 0.6 absolute value) detected among numeric columns.")
            else:
                st.warning("Not enough numeric columns available to generate automated statistical correlations.")

        # --- TAB 5: EXPORT DATA ---
        with tab_export:
            st.subheader("Download Filtered Dataset")
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered CSV",
                data=csv_data,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Use the sidebar on the left to upload a CSV or Excel dataset to get started.")
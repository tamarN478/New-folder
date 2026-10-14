"""streamlit application for uploading a dataset, performing basic EDA, and creating interactive charts."""

from pathlib import Path
import plotly.express as px
import pandas as pd
import streamlit as st

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
        # Load dataset based on extension
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Tab navigation for EDA and Chart Builder
        tab_eda, tab_charts = st.tabs(["📊 Exploratory Data Analysis", "📈 Chart Builder"])

        # --- TAB 1: EDA ---
        with tab_eda:
            st.subheader("Data Preview")
            st.dataframe(df.head(), width=1200)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])

            with st.expander("Summary Statistics"):
                st.dataframe(df.describe(include="all"), width=1200)

            with st.expander("Missing Values Overview"):
                missing = df.isnull().sum()
                missing_pct = (missing / len(df)) * 100
                missing_df = pd.DataFrame(
                    {"Missing Values": missing, "Percentage (%)": missing_pct}
                )
                st.dataframe(missing_df, width=1200)

        # --- TAB 2: CHART BUILDER ---
        with tab_charts:
            st.subheader("Create Custom Visualizations")

            col_controls, col_display = st.columns([1, 2])

            all_columns = df.columns.tolist()

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
                if chart_type == "Bar Chart":
                    if y_axis:
                        fig = px.bar(df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
                    else:
                        st.warning("Please select a Y-axis column for the Bar Chart.")

                elif chart_type == "Line Chart":
                    if y_axis:
                        fig = px.line(df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
                    else:
                        st.warning("Please select a Y-axis column for the Line Chart.")

                elif chart_type == "Scatter Plot":
                    if y_axis:
                        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
                    else:
                        st.warning("Please select a Y-axis column for the Scatter Plot.")

                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, y=y_axis, color=color_by, title=chart_title)

                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_by, title=chart_title)

                if fig:
                    st.plotly_chart(fig, width=1200)

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Use the sidebar on the left to upload a CSV or Excel dataset to get started.")
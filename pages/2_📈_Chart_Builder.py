"""interactive chart builder page."""

import plotly.express as px
import streamlit as st

st.title("📈 Interactive Chart Builder")

if "raw_df" not in st.session_state:
    st.warning("Please upload a dataset on the main Home page first!")
else:
    df = st.session_state.raw_df
    all_columns = df.columns.tolist()

    col_controls, col_display = st.columns([1, 2])

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
        color_by = st.selectbox("Color / Group By (Optional)", options=[None] + all_columns)
        chart_title = st.text_input("Chart Title", f"{chart_type} of {x_axis}")

    with col_display:
        fig = None
        if chart_type == "Bar Chart" and y_axis:
            fig = px.bar(df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
        elif chart_type == "Line Chart" and y_axis:
            fig = px.line(df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
        elif chart_type == "Scatter Plot" and y_axis:
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_axis, y=y_axis, color=color_by, title=chart_title)
        else:
            st.warning("Please select a valid Y-axis column.")

        if fig:
            st.plotly_chart(fig, width=1200)
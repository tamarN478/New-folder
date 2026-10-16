"""interactive chart builder page."""

import pandas as pd
import plotly.express as px
import streamlit as st
from charts import (
    MAX_SERIES,
    correlation_heatmap,
    fold_to_top_categories,
    series_colors,
    style_figure,
)
from ui import page_header, require_dataset

page_header("Chart Builder", "Pick a chart type and columns; the chart updates as you go.")

CHART_TYPES = [
    "Bar Chart", "Line Chart", "Scatter Plot", "Histogram",
    "Box Plot", "Pie Chart", "Correlation Heatmap",
]
AGGREGATIONS = {
    "Average": "mean", "Sum": "sum", "Count": "count", "Median": "median",
    "Min": "min", "Max": "max", "None (raw rows)": None,
}
# Charts where several rows with the same X value should be combined into one.
AGGREGATABLE = {"Bar Chart", "Line Chart", "Pie Chart"}

df = require_dataset()
all_columns = df.columns.tolist()
numeric_columns = df.select_dtypes(include="number").columns.tolist()

col_controls, col_display = st.columns([1, 2])

with col_controls, st.container(border=True):
    chart_type = st.selectbox("Chart type", CHART_TYPES)

    x_axis = y_axis = color_by = None
    aggregation = None
    if chart_type == "Correlation Heatmap":
        st.caption("Shows how strongly each pair of numeric columns moves together.")
    elif chart_type == "Pie Chart":
        x_axis = st.selectbox("Slices (category column)", options=all_columns)
        y_axis = st.selectbox(
            "Slice size (numeric column, optional: counts rows if empty)",
            options=[None] + numeric_columns,
        )
    else:
        x_axis = st.selectbox("X axis", options=all_columns)
        y_options = [None] + all_columns
        default_y = next((c for c in numeric_columns if c != x_axis), None)
        y_axis = st.selectbox(
            "Y axis",
            options=y_options,
            index=y_options.index(default_y) if default_y else 0,
        )
        color_by = st.selectbox("Color by (optional)", options=[None] + all_columns)

    if chart_type in AGGREGATABLE and y_axis:
        aggregation = AGGREGATIONS[
            st.selectbox(
                "Combine rows with the same value using",
                list(AGGREGATIONS),
                help="E.g. 'Average' turns many students per faculty into one bar "
                     "showing the average GPA of each faculty.",
            )
        ]

    default_title = (
        "Correlation Heatmap" if chart_type == "Correlation Heatmap"
        else f"{chart_type} of {x_axis}"
    )
    chart_title = st.text_input("Title", default_title)

with col_display, st.container(border=True):
    plot_df = df
    if color_by:
        # More than a handful of colors can't be told apart, so fold the rest into "Other".
        plot_df = fold_to_top_categories(plot_df, color_by)
        if df[color_by].nunique() > MAX_SERIES:
            st.caption(
                f"'{color_by}' has {df[color_by].nunique()} values; the smaller ones "
                "are grouped as 'Other' so the colors stay readable."
            )

    if aggregation and y_axis:
        group_columns = [c for c in [x_axis, color_by] if c]
        if aggregation != "count" and y_axis not in numeric_columns:
            st.warning(f"'{y_axis}' isn't numeric, so it can only be combined with **Count**.")
            aggregation = "count"
        plot_df = (
            plot_df.groupby(group_columns, dropna=False, observed=True)[y_axis]
            .agg(aggregation)
            .reset_index()
        )

    colors = series_colors()
    common = dict(color=color_by, title=chart_title, color_discrete_sequence=colors)
    fig = None
    try:
        if chart_type == "Correlation Heatmap":
            if len(numeric_columns) < 2:
                st.warning("A heatmap needs at least two numeric columns.")
            else:
                fig = correlation_heatmap(df, chart_title)
        elif chart_type == "Pie Chart":
            if y_axis:
                fig = px.pie(plot_df, names=x_axis, values=y_axis, title=chart_title,
                             color_discrete_sequence=colors)
            else:
                counts = fold_to_top_categories(df, x_axis)[x_axis].astype(str).value_counts()
                fig = px.pie(names=counts.index, values=counts.values, title=chart_title,
                             color_discrete_sequence=colors)
        elif chart_type == "Bar Chart" and y_axis:
            fig = px.bar(plot_df, x=x_axis, y=y_axis, barmode="group", **common)
        elif chart_type == "Line Chart" and y_axis:
            fig = px.line(plot_df.sort_values(x_axis), x=x_axis, y=y_axis, markers=True, **common)
        elif chart_type == "Scatter Plot" and y_axis:
            fig = px.scatter(plot_df, x=x_axis, y=y_axis, **common)
        elif chart_type == "Histogram":
            fig = px.histogram(plot_df, x=x_axis, y=y_axis, barmode="group", **common)
        elif chart_type == "Box Plot":
            fig = px.box(plot_df, x=x_axis, y=y_axis, **common)
        else:
            st.info("Choose a Y-axis column to draw this chart.", icon=":material/info:")
    except (ValueError, TypeError) as error:
        st.error(f"Couldn't draw this chart with the selected columns: {error}")

    if fig:
        st.plotly_chart(style_figure(fig), width="stretch")
        st.caption("To save as a PNG image, use the camera icon at the chart's top-right.")
        st.download_button(
            "Download interactive chart (HTML)",
            fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
            file_name=f"{chart_title or 'chart'}.html",
            mime="text/html",
            icon=":material/download:",
        )
        if aggregation:
            with st.expander("Show the numbers behind this chart"):
                st.dataframe(plot_df, width="stretch", hide_index=True)

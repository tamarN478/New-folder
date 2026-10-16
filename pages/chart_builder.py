"""interactive chart builder page."""

import uuid

import streamlit as st
from charts import AGGREGATABLE, AGGREGATIONS, CHART_TYPES, ChartError, build_chart
from ui import page_header, require_dataset

page_header("Chart Builder", "Pick a chart type and columns; the chart updates as you go.")

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
        # A pie only makes sense with a handful of slices.
        slice_columns = [c for c in all_columns if df[c].nunique() <= 30] or all_columns
        x_axis = st.selectbox("Slices (category column)", options=slice_columns)
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
        aggregation = st.selectbox(
            "Combine rows with the same value using",
            list(AGGREGATIONS),
            help="E.g. 'Average' turns many students per faculty into one bar "
                 "showing the average GPA of each faculty.",
        )

    default_title = (
        "Correlation Heatmap" if chart_type == "Correlation Heatmap"
        else f"{chart_type} of {x_axis}"
    )
    chart_title = st.text_input("Title", default_title)

spec = {
    "chart_type": chart_type, "x": x_axis, "y": y_axis, "color": color_by,
    "aggregation": aggregation, "title": chart_title,
}

with col_display, st.container(border=True):
    try:
        fig, plot_df, notes = build_chart(df, spec)
    except ChartError as error:
        st.info(str(error), icon=":material/info:")
        st.stop()

    for note in notes:
        st.caption(note)
    st.plotly_chart(fig, width="stretch")
    st.caption("To save as a PNG image, use the camera icon at the chart's top-right.")

    actions = st.container(horizontal=True, gap="small")
    if actions.button("Pin to dashboard", icon=":material/push_pin:", type="primary"):
        st.session_state.setdefault("dashboard_charts", []).append({**spec, "id": uuid.uuid4().hex[:8]})
        st.toast("Pinned to the dashboard", icon=":material/push_pin:")
    actions.download_button(
        "Download chart (HTML)",
        fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
        file_name=f"{chart_title or 'chart'}.html",
        mime="text/html",
        icon=":material/download:",
    )
    pinned = len(st.session_state.get("dashboard_charts", []))
    if pinned:
        actions.page_link("pages/dashboard.py", label=f"Dashboard ({pinned})",
                          icon=":material/dashboard:")
    if aggregation and AGGREGATIONS[aggregation]:
        with st.expander("Show the numbers behind this chart"):
            st.dataframe(plot_df, width="stretch", hide_index=True)

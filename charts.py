"""shared chart colors and styling so every chart in the app looks consistent."""

import pandas as pd
import plotly.express as px
import streamlit as st

# Categorical colors, assigned in this fixed order (never cycled or shuffled).
# Light and dark variants are tuned to stay distinguishable for colorblind readers.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

# Diverging scale for correlations: blue (negative) -> gray (none) -> red (positive).
DIVERGING_LIGHT = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
DIVERGING_DARK = [[0.0, "#3987e5"], [0.5, "#383835"], [1.0, "#e66767"]]

MAX_SERIES = len(SERIES_LIGHT)


def _is_dark():
    try:
        return st.context.theme.type == "dark"
    except AttributeError:
        return False


def series_colors():
    return SERIES_DARK if _is_dark() else SERIES_LIGHT


def diverging_scale():
    return DIVERGING_DARK if _is_dark() else DIVERGING_LIGHT


def style_figure(fig):
    """apply thin marks, rounded bar ends, and quiet axes."""
    fig.update_layout(
        barcornerradius=4,
        bargap=0.25,
        legend_title_text="",
        margin=dict(t=60, l=10, r=10, b=10),
        hoverlabel=dict(namelength=-1),
        font=dict(family="Inter, sans-serif"),
        title_font=dict(size=16),
    )
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    fig.update_traces(selector=dict(type="scatter", mode="lines"), line_width=2)
    fig.update_traces(selector=dict(type="scatter", mode="markers"), marker_size=8)
    fig.update_traces(
        selector=dict(type="pie"),
        marker_line_width=2,
        marker_line_color="rgba(0,0,0,0)",
        textinfo="percent+label",
    )
    return fig


def fold_to_top_categories(df, column, limit=MAX_SERIES):
    """keep the most common values of a column and label the rest 'Other'."""
    top = df[column].value_counts().index[: limit - 1]
    if df[column].nunique() <= limit:
        return df
    folded = df.copy()
    folded[column] = folded[column].where(folded[column].isin(top), "Other").astype(str)
    return folded


def correlation_heatmap(df, title="Correlation between numeric columns"):
    corr = df.select_dtypes(include="number").corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale=diverging_scale(),
        title=title,
        aspect="auto",
    )
    fig.update_layout(coloraxis_colorbar_title="")
    return fig


def auto_histogram(df, column):
    fig = px.histogram(df, x=column, title=column, color_discrete_sequence=series_colors())
    fig.update_layout(yaxis_title="Count", xaxis_title="", showlegend=False)
    return style_figure(fig)


def auto_top_values(df, column, limit=10):
    counts = df[column].astype(str).value_counts().head(limit).sort_values()
    data = pd.DataFrame({column: counts.index, "Count": counts.values})
    fig = px.bar(
        data, x="Count", y=column, orientation="h",
        title=f"{column} (top {min(limit, len(counts))})",
        color_discrete_sequence=series_colors(),
    )
    fig.update_layout(yaxis_title="", showlegend=False)
    return style_figure(fig)


# ---------------------------------------------------------------------------
# chart specs: a chart described as plain settings, so it can be saved,
# pinned to the dashboard, and redrawn later from the current data
# ---------------------------------------------------------------------------

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


class ChartError(ValueError):
    """the chart can't be drawn with these settings; the message says why."""


def build_chart(df, spec):
    """draw a chart from a spec dict.

    spec keys: chart_type, x, y, color, aggregation (a key of AGGREGATIONS), title.
    Returns (figure, the data actually plotted, list of notes for the reader).
    """
    chart_type, title = spec["chart_type"], spec.get("title") or spec["chart_type"]
    x, y, color = spec.get("x"), spec.get("y"), spec.get("color")
    aggregation = AGGREGATIONS.get(spec.get("aggregation")) if chart_type in AGGREGATABLE and y else None
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    missing = [c for c in (x, y, color) if c and c not in df.columns]
    if missing:
        raise ChartError(f"This chart uses '{missing[0]}', which isn't in the data anymore.")

    notes = []
    plot_df = df
    if color:
        # More than a handful of colors can't be told apart, so fold the rest into "Other".
        plot_df = fold_to_top_categories(plot_df, color)
        if df[color].nunique() > MAX_SERIES:
            notes.append(f"'{color}' has {df[color].nunique()} values; the smaller ones are "
                         "grouped as 'Other' so the colors stay readable.")

    if aggregation and y:
        if aggregation != "count" and y not in numeric_columns:
            notes.append(f"'{y}' isn't numeric, so it can only be combined with Count.")
            aggregation = "count"
        group_columns = [c for c in [x, color] if c]
        plot_df = plot_df.groupby(group_columns, dropna=False, observed=True)[y].agg(aggregation).reset_index()

    colors = series_colors()
    common = dict(color=color, title=title, color_discrete_sequence=colors)
    try:
        if chart_type == "Correlation Heatmap":
            if len(numeric_columns) < 2:
                raise ChartError("A heatmap needs at least two numeric columns.")
            fig = correlation_heatmap(df, title)
        elif chart_type == "Pie Chart":
            if y:
                fig = px.pie(plot_df, names=x, values=y, title=title, color_discrete_sequence=colors)
            else:
                counts = fold_to_top_categories(df, x)[x].astype(str).value_counts()
                fig = px.pie(names=counts.index, values=counts.values, title=title,
                             color_discrete_sequence=colors)
        elif chart_type == "Histogram":
            fig = px.histogram(plot_df, x=x, y=y, barmode="group", **common)
        elif chart_type == "Box Plot":
            fig = px.box(plot_df, x=x, y=y, **common)
        elif not y:
            raise ChartError("Choose a Y-axis column to draw this chart.")
        elif chart_type == "Bar Chart":
            fig = px.bar(plot_df, x=x, y=y, barmode="group", **common)
        elif chart_type == "Line Chart":
            fig = px.line(plot_df.sort_values(x), x=x, y=y, markers=True, **common)
        else:
            fig = px.scatter(plot_df, x=x, y=y, **common)
    except (ValueError, TypeError) as error:
        if isinstance(error, ChartError):
            raise
        raise ChartError(f"Couldn't draw this chart with the selected columns: {error}") from None
    return style_figure(fig), plot_df, notes


def describe_spec(spec):
    """a short sentence describing a chart spec."""
    parts = [spec["chart_type"]]
    if spec.get("y"):
        agg = spec.get("aggregation")
        prefix = f"{agg.lower()} of " if agg and agg != "None (raw rows)" and spec["chart_type"] in AGGREGATABLE else ""
        parts.append(f"{prefix}{spec['y']}")
    if spec.get("x"):
        parts.append(f"by {spec['x']}")
    if spec.get("color"):
        parts.append(f"colored by {spec['color']}")
    return " ".join(parts)

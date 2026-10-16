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

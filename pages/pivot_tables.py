"""group & pivot page: summarize the data like an Excel pivot table."""

import pandas as pd
import plotly.express as px
import streamlit as st
from charts import series_colors, style_figure
from ui import page_header, require_dataset
from utils import download_buttons

AGGREGATIONS = {
    "Average": "mean", "Sum": "sum", "Count": "count",
    "Median": "median", "Min": "min", "Max": "max",
}


def build_table(df, rows, column, values, aggregation, totals):
    """return the summary table for the chosen settings."""
    if not values:
        # No value column: count how many rows fall into each group.
        counted = df.assign(_rows=1)
        return pd.pivot_table(
            counted, index=rows, columns=column, values="_rows", aggfunc="sum",
            fill_value=0, margins=totals, margins_name="Total", observed=True,
        )
    return pd.pivot_table(
        df, index=rows, columns=column, values=values, aggfunc=AGGREGATIONS[aggregation],
        margins=totals, margins_name="Total", observed=True,
    )


def flatten_columns(table):
    """turn multi-level column headers into single readable names."""
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [" · ".join(str(part) for part in col if str(part)) for col in table.columns]
    else:
        table.columns = [str(col) for col in table.columns]
    return table.rename(columns={"_rows": "Rows"})


page_header("Group & Pivot", "Summarize your data by groups, like a pivot table in Excel.")
df = require_dataset()

all_columns = df.columns.tolist()
numeric_columns = df.select_dtypes("number").columns.tolist()
group_candidates = [c for c in all_columns if df[c].nunique() <= max(50, len(df) // 2)]

settings_col, result_col = st.columns([1, 2])
with settings_col, st.container(border=True):
    st.markdown("##### Settings")
    rows = st.multiselect(
        "Group rows by", group_candidates,
        default=[c for c in group_candidates if c not in numeric_columns][:1],
        help="Each different value becomes a row, e.g. one row per faculty.",
    )
    column = st.selectbox(
        "Split into columns by (optional)", [None] + [c for c in group_candidates if c not in rows],
        help="Each different value becomes a column, e.g. one column per year.",
    )
    values = st.multiselect(
        "Values to summarize", [c for c in numeric_columns if c not in rows and c != column],
        help="Leave empty to just count rows in each group.",
    )
    aggregation = st.segmented_control(
        "Calculate", list(AGGREGATIONS), default="Average", required=True, disabled=not values,
    )
    totals = st.toggle("Show totals", value=True)

with result_col, st.container(border=True):
    if not rows:
        st.info("Pick at least one column to group rows by.", icon=":material/info:")
        st.stop()
    try:
        table = flatten_columns(build_table(df, rows, column, values, aggregation, totals))
    except (ValueError, TypeError) as error:
        st.error(f"Couldn't build this table: {error}")
        st.stop()

    measure = f"{aggregation.lower()} of {', '.join(values)}" if values else "number of rows"
    split = f", split by {column}" if column else ""
    st.markdown(f"##### {measure.capitalize()} by {', '.join(rows)}{split}")
    st.dataframe(
        table.round(2), width="stretch",
        column_config={c: st.column_config.NumberColumn(format="localized") for c in table.columns},
    )
    st.caption(f"{len(table):,} groups")

    table_out = table.reset_index()
    download_buttons(table_out, file_stem="pivot_table", key="pivot_download")

    chartable = table.drop(index="Total", errors="ignore")
    chartable = chartable.drop(columns="Total", errors="ignore")
    if len(rows) == 1 and 0 < len(chartable) <= 40:
        with st.expander("Show as a chart", icon=":material/bar_chart:"):
            long = chartable.reset_index().melt(id_vars=rows, var_name="Series", value_name="Value")
            fig = px.bar(
                long, x=rows[0], y="Value", color="Series" if chartable.shape[1] > 1 else None,
                barmode="group", color_discrete_sequence=series_colors(),
            )
            fig.update_layout(yaxis_title=measure.capitalize())
            st.plotly_chart(style_figure(fig), width="stretch")

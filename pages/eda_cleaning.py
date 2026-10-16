"""eda and data cleaning page."""

import pandas as pd
import streamlit as st
from charts import auto_histogram, auto_top_values, correlation_heatmap
from quality import PENALTIES, quality_report
from transforms import DATE_PARTS, add_date_parts, add_formula_column, add_ranges, parse_boundaries
from ui import page_header, require_dataset
from utils import (
    CLEANING_ACTIONS,
    apply_change,
    clean_data,
    download_buttons,
    reset_to_original,
    undo_last_change,
)

MAX_AUTO_CHARTS = 12


def filter_rows(df):
    """let the user narrow the rows down by one column; returns the filtered rows."""
    toolbar = st.container(horizontal=True, vertical_alignment="center")
    with toolbar.popover("Filter rows", icon=":material/filter_list:"):
        enable_filtering = st.toggle("Filter by a column", key="filter_enabled")
        filtered_df = df
        if enable_filtering:
            filter_column = st.selectbox("Column", options=df.columns.tolist())
            column_values = df[filter_column]
            if pd.api.types.is_numeric_dtype(column_values) and column_values.notna().any():
                low, high = float(column_values.min()), float(column_values.max())
                if low < high:
                    selected_range = st.slider(
                        "Range", min_value=low, max_value=high, value=(low, high)
                    )
                    filtered_df = df[column_values.between(*selected_range)]
                else:
                    st.caption(f"Every value in this column is {low:g}.")
            else:
                unique_vals = column_values.dropna().unique()
                selected_vals = st.multiselect(
                    "Values", options=unique_vals, default=list(unique_vals)
                )
                if selected_vals:
                    filtered_df = df[column_values.isin(selected_vals)]
    if len(filtered_df) < len(df):
        toolbar.badge(
            f"Showing {len(filtered_df):,} of {len(df):,} rows",
            icon=":material/filter_alt:", color="blue",
        )
    return filtered_df, len(filtered_df) < len(df)


def overview_tab(df, is_filtered):
    metrics = st.container(horizontal=True)
    metrics.metric("Rows", f"{len(df):,}", border=True)
    metrics.metric("Columns", df.shape[1], border=True)
    metrics.metric("Missing cells", f"{int(df.isnull().sum().sum()):,}", border=True)
    metrics.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}", border=True)

    st.dataframe(df, width="stretch", height=320)

    stats_col, missing_col = st.columns([3, 2])
    with stats_col:
        st.markdown("##### Summary statistics")
        # Mixed numbers and text per column; show as text so the table renders cleanly.
        summary = df.describe(include="all")
        st.dataframe(summary.astype(object).where(summary.notna(), "").astype(str), width="stretch")
    with missing_col:
        st.markdown("##### Missing values")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.caption("No missing values. Nice and complete.")
        else:
            missing = missing[missing > 0].sort_values(ascending=False)
            st.dataframe(
                pd.DataFrame({
                    "Column": missing.index,
                    "Missing": missing.values,
                    "Share": missing.values / max(len(df), 1),
                }),
                hide_index=True,
                width="stretch",
                column_config={
                    "Share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                },
            )

    st.markdown("##### Download")
    if is_filtered:
        st.caption("Downloads include only the rows that match your filter.")
    download_buttons(df, file_stem="cleaned_data", key="eda_download")


def auto_charts_tab(df):
    st.caption("One chart per column, so you can spot patterns and odd values at a glance.")
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    other_columns = [c for c in df.columns if c not in numeric_columns]

    if len(numeric_columns) >= 2:
        with st.container(border=True):
            st.plotly_chart(correlation_heatmap(df), width="stretch")

    charts = [("hist", c) for c in numeric_columns] + [("top", c) for c in other_columns]
    if len(charts) > MAX_AUTO_CHARTS:
        st.caption(f"Showing the first {MAX_AUTO_CHARTS} of {len(charts)} columns.")
    grid = st.columns(2)
    for position, (kind, column) in enumerate(charts[:MAX_AUTO_CHARTS]):
        fig = auto_histogram(df, column) if kind == "hist" else auto_top_values(df, column)
        with grid[position % 2].container(border=True):
            st.plotly_chart(fig, width="stretch")


def history_panel():
    history = st.session_state.get("history", [])
    with st.container(border=True):
        st.markdown("##### History")
        buttons = st.container(horizontal=True, gap="small")
        if buttons.button("Undo", icon=":material/undo:", disabled=not history):
            undone = undo_last_change()
            st.session_state.cleaning_message = f"Undid: {undone}"
            st.rerun()
        if buttons.button("Reset", icon=":material/restart_alt:", disabled=not history,
                          help="Go back to the data as it was loaded. You can undo this too."):
            reset_to_original()
            st.session_state.cleaning_message = "Restored the original data"
            st.rerun()
        if history:
            st.markdown("\n".join(f"{i}. {description}" for i, (description, _) in enumerate(history, 1)))
        else:
            st.caption("Cleaning steps you apply will be listed here, and you can undo them.")


def quality_tab(df):
    score, summary, issues = quality_report(df)
    verdict = "Excellent" if score >= 90 else "Good" if score >= 75 else "Needs work" if score >= 50 else "Poor"

    top = st.container(horizontal=True)
    top.metric("Quality score", f"{score}/100", border=True, help="Average of the column scores below, "
               "minus points for problems with the whole dataset such as duplicate rows.")
    top.metric("Verdict", verdict, border=True)
    top.metric("Problems found", len(issues), border=True)

    st.markdown("##### Column scores")
    st.dataframe(
        summary, hide_index=True, width="stretch",
        column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d")},
    )

    st.markdown("##### Suggested fixes")
    if not issues:
        st.caption("No problems found. Your data looks clean.")
        return
    st.caption("Each fix is recorded in the History panel on the Clean data tab, so you can undo it.")
    severity_badge = {"high": ("High", "red"), "medium": ("Medium", "orange"), "low": ("Low", "gray")}
    for position, issue in enumerate(sorted(issues, key=lambda i: list(PENALTIES).index(i.severity))):
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            label, color = severity_badge[issue.severity]
            st.badge(label, color=color)
            where = f"**{issue.column}**: " if issue.column else "**Whole dataset**: "
            st.markdown(where + issue.problem, width="stretch")
            if st.button(issue.fix_label, key=f"fix_{position}", icon=":material/auto_fix_high:"):
                cleaned = clean_data(df, issue.action, issue.columns, issue.option)
                apply_change(cleaned, f"{issue.fix_label} ({issue.column or 'all rows'})")
                st.session_state.cleaning_message = f"{issue.fix_label}: done"
                st.rerun()


def new_columns_tab(df):
    if "column_message" in st.session_state:
        st.toast(st.session_state.pop("column_message"), icon=":material/check_circle:")

    numeric_cols = df.select_dtypes("number").columns.tolist()
    kind = st.segmented_control(
        "What kind of column?",
        ["Formula", "Date parts", "Number ranges"],
        default="Formula",
        required=True,
        key="new_column_kind",
    )

    with st.container(border=True):
        new_df, description = None, None
        try:
            if kind == "Formula":
                st.caption(
                    "Use column names with + − × ÷ (`+ - * /`), powers (`**`) and brackets. "
                    "Examples: `price * quantity`, `gpa / 4 * 100`, `tip / total_bill`."
                )
                st.markdown("Columns: " + ", ".join(f"`{c}`" for c in df.columns))
                name = st.text_input("New column name", key="formula_name").strip()
                expression = st.text_input("Formula", key="formula_expression")
                if expression:
                    if not name:
                        st.caption("Give the new column a name to add it.")
                    else:
                        new_df = add_formula_column(df, name, expression)
                        description = f"Add column '{name}' = {expression}"
            elif kind == "Date parts":
                column = st.selectbox("Date column", df.columns, key="date_column")
                parts = st.pills("Parts to add", list(DATE_PARTS), selection_mode="multi",
                                 default=["Year", "Month"], key="date_parts")
                if parts:
                    new_df = add_date_parts(df, column, parts)
                    description = f"Split '{column}' into {', '.join(parts).lower()}"
            else:
                if not numeric_cols:
                    st.caption("This needs a numeric column.")
                    return
                column = st.selectbox("Numeric column", numeric_cols, key="range_column")
                name = st.text_input("New column name", value=f"{column} range", key="range_name").strip()
                mode = st.radio("Ranges", ["Equal-width ranges", "My own boundaries"], horizontal=True,
                                key="range_mode")
                if mode == "Equal-width ranges":
                    bins = st.slider("How many ranges", 2, 10, 4, key="range_count")
                else:
                    boundaries = st.text_input("Boundaries", placeholder="e.g. 0, 2, 3, 3.5, 4",
                                               key="range_bounds")
                    bins = parse_boundaries(boundaries) if boundaries else None
                if bins and name:
                    new_df = add_ranges(df, column, name, bins)
                    description = f"Group '{column}' into ranges as '{name}'"
        except Exception as error:
            st.error(f"{error}", icon=":material/error:")
            new_df = None

        if new_df is not None:
            added = [c for c in new_df.columns if c not in df.columns or not new_df[c].equals(df[c])]
            st.markdown("##### Preview")
            st.dataframe(new_df.head(8), width="stretch",
                         column_config={c: st.column_config.Column(pinned=True) for c in added})
            replaced = [c for c in added if c in df.columns]
            if replaced:
                st.warning(f"'{replaced[0]}' already exists and will be replaced.")
            if st.button("Add column" + ("s" if len(added) > 1 else ""), type="primary",
                         icon=":material/add:"):
                apply_change(new_df, description)
                st.session_state.column_message = description
                st.rerun()


def cleaning_tab(df):
    if "cleaning_message" in st.session_state:
        st.toast(st.session_state.pop("cleaning_message"), icon=":material/check_circle:")

    form_col, history_col = st.columns([2, 1])
    with history_col:
        history_panel()

    with form_col, st.container(border=True):
        st.markdown("##### Cleaning action")
        cleaning_action = st.selectbox(
            "Action", CLEANING_ACTIONS, index=None, placeholder="Choose what to fix…",
            label_visibility="collapsed",
        )

        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        columns, option, value = None, None, None
        ready = cleaning_action is not None

        if cleaning_action == "Drop Duplicates":
            st.caption("Removes rows that are exact copies of an earlier row.")
        elif cleaning_action == "Drop Rows with Missing Values":
            columns = st.multiselect(
                "Only look at these columns (leave empty for all)", all_columns
            )
        elif cleaning_action == "Fill Missing Values":
            columns = st.multiselect(
                "Columns to fill", all_columns,
                default=[c for c in all_columns if df[c].isnull().any()],
            )
            option = st.segmented_control(
                "Fill with",
                ["Mean", "Median", "Most Common Value (Mode)", "Custom Value"],
                default="Mean",
                required=True,
                help="Mean and Median only work on numeric columns.",
            )
            if option == "Custom Value":
                value = st.text_input("Value to fill in")
                ready = bool(value)
            ready = ready and bool(columns)
        elif cleaning_action == "Change Column Type":
            columns = st.multiselect("Columns", all_columns)
            option = st.segmented_control(
                "New type", ["Number", "Text", "Date", "Category"], default="Number", required=True
            )
            st.caption("Values that can't be converted become empty (missing).")
            ready = bool(columns)
        elif cleaning_action == "Rename Column":
            columns = [st.selectbox("Column", all_columns)]
            value = st.text_input("New name").strip()
            if value in all_columns:
                st.error(f"A column named '{value}' already exists.")
            ready = bool(value) and value not in all_columns
        elif cleaning_action == "Drop Columns":
            columns = st.multiselect("Columns to remove", all_columns)
            ready = bool(columns) and len(columns) < len(all_columns)
        elif cleaning_action == "Remove Outliers (IQR)":
            columns = st.multiselect("Numeric columns", numeric_cols, default=numeric_cols[:1])
            st.caption(
                "Removes rows with values far outside the typical range "
                "(more than 1.5× the interquartile range beyond the middle 50%)."
            )
            ready = bool(columns)
        elif cleaning_action == "Trim Spaces":
            columns = st.multiselect("Text columns", text_cols, default=text_cols)
            st.caption("Removes extra spaces at the start and end, e.g. ' Math ' becomes 'Math'.")
            ready = bool(columns)
        elif cleaning_action == "Fix Capitalization":
            columns = st.multiselect("Text columns", text_cols, default=text_cols)
            option = st.segmented_control(
                "Style", ["Title Case", "lowercase", "UPPERCASE"], default="Title Case", required=True
            )
            ready = bool(columns)

        if st.button("Apply", type="primary", icon=":material/check:", disabled=not ready):
            try:
                cleaned = clean_data(df, cleaning_action, columns, option, value)
            except Exception as error:
                st.error(f"Couldn't apply '{cleaning_action}': {error}")
            else:
                apply_change(cleaned, cleaning_action)
                removed = len(df) - len(cleaned)
                detail = f", removed {removed} row(s)" if removed else ""
                st.session_state.cleaning_message = f"Applied {cleaning_action}{detail}"
                st.rerun()


page_header("Explore & Clean", "Get to know your data, spot problems, and fix them, with undo for every step.")
df = require_dataset()
filtered_df, is_filtered = filter_rows(df)

tab_overview, tab_quality, tab_auto, tab_cleaning, tab_columns = st.tabs([
    ":material/table_chart: Overview",
    ":material/health_and_safety: Data quality",
    ":material/insights: Automatic charts",
    ":material/cleaning_services: Clean data",
    ":material/function: New columns",
])
with tab_overview:
    overview_tab(filtered_df, is_filtered)
with tab_quality:
    quality_tab(df)
with tab_auto:
    auto_charts_tab(filtered_df)
with tab_cleaning:
    cleaning_tab(df)
with tab_columns:
    new_columns_tab(df)

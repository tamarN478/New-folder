"""combine page: join or stack a second dataset onto the current one."""

import streamlit as st
from combine import JOIN_TYPES, join_datasets, stack_datasets, suggest_keys
from ui import describe_source, page_header, require_dataset
from utils import SAMPLE_DATASETS, apply_change, load_data


@st.cache_data(show_spinner=False)
def read_upload(data, name):
    import io

    buffer = io.BytesIO(data)
    buffer.name = name
    return load_data(buffer)


def second_dataset():
    """let the user pick the dataset to combine with; returns (df, name) or (None, None)."""
    source = st.segmented_control(
        "Second dataset", ["Upload a file", "Sample dataset"], default="Upload a file",
        required=True, key="combine_source",
    )
    if source == "Upload a file":
        uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"], key="combine_upload")
        if uploaded is None:
            return None, None
        try:
            return read_upload(uploaded.getvalue(), uploaded.name), uploaded.name
        except Exception as error:
            st.error(f"Couldn't read this file: {error}")
            return None, None
    name = st.selectbox("Sample", list(SAMPLE_DATASETS), index=list(SAMPLE_DATASETS).index("Faculties"),
                        key="combine_sample")
    return SAMPLE_DATASETS[name](), name


page_header("Combine Data", "Add columns from another dataset by matching rows, or stack two datasets.")
df = require_dataset()

left_col, right_col = st.columns(2)
with left_col, st.container(border=True, height="stretch"):
    st.markdown("##### Your data")
    st.caption(f"{describe_source(st.session_state.get('dataset_source'))} · "
               f"{len(df):,} rows · {len(df.columns)} columns")
    st.dataframe(df.head(5), width="stretch")
with right_col, st.container(border=True, height="stretch"):
    st.markdown("##### Combine with")
    other, other_name = second_dataset()
    if other is not None:
        st.caption(f"{len(other):,} rows · {len(other.columns)} columns")
        st.dataframe(other.head(5), width="stretch")

if other is None:
    st.stop()

mode = st.segmented_control(
    "How to combine",
    [":material/join_inner: Join (match rows)", ":material/table_rows: Stack (add rows)"],
    default=":material/join_inner: Join (match rows)", required=True, key="combine_mode",
)

with st.container(border=True):
    result, description = None, None
    if mode.endswith("Join (match rows)"):
        st.caption("Rows are matched where the chosen columns have the same value, like VLOOKUP in Excel.")
        suggested = suggest_keys(df, other)
        keys = st.columns(3)
        left_keys = keys[0].multiselect("Match my column(s)", df.columns.tolist(), default=suggested[:1])
        right_keys = keys[1].multiselect(
            "with their column(s)", other.columns.tolist(),
            default=[k for k in left_keys if k in other.columns],
        )
        how = keys[2].selectbox("Keep", list(JOIN_TYPES))
        if left_keys and right_keys:
            try:
                result, stats, notes = join_datasets(df, other, left_keys, right_keys, JOIN_TYPES[how],
                                                     right_name=other_name)
            except (ValueError, TypeError) as error:
                st.error(str(error))
            else:
                metrics = st.container(horizontal=True)
                metrics.metric("Matched rows", f"{stats['matched']:,}", border=True)
                metrics.metric("Only in my data", f"{stats['only_mine']:,}", border=True,
                               help="These rows get empty values for the new columns.")
                if JOIN_TYPES[how] == "outer":
                    metrics.metric("Only in theirs", f"{stats['only_other']:,}", border=True)
                metrics.metric("Result", f"{len(result):,} rows × {len(result.columns)} cols", border=True)
                for note in notes:
                    st.caption(note)
                description = f"Join with {other_name} on {', '.join(left_keys)}"
        else:
            st.info("Pick the column(s) that identify matching rows on both sides.", icon=":material/info:")
    else:
        add_source = st.toggle("Add a 'source' column saying where each row came from", value=True)
        result, only_mine, only_theirs = stack_datasets(df, other, add_source, bottom_name=other_name)
        if only_mine or only_theirs:
            st.caption(
                "Columns that exist in only one dataset get empty values in the other's rows: "
                + ", ".join(f"`{c}`" for c in only_mine + only_theirs)
            )
        st.metric("Result", f"{len(result):,} rows × {len(result.columns)} cols", border=True)
        description = f"Stack rows from {other_name}"

    if result is not None:
        st.markdown("##### Preview")
        st.dataframe(result.head(20), width="stretch")
        if st.button("Use the combined data", type="primary", icon=":material/check:"):
            apply_change(result, description)
            st.toast("Combined. You can undo this on the Explore & Clean page.", icon=":material/check_circle:")
            st.rerun()

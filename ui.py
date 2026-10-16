"""shared layout pieces so every page looks the same."""

import streamlit as st


def page_header(title, subtitle):
    """a page title with a one-line description underneath."""
    st.title(title)
    st.caption(subtitle)


def require_dataset():
    """stop the page with a friendly message when no data is loaded yet."""
    if "raw_df" in st.session_state and not st.session_state.raw_df.empty:
        return st.session_state.raw_df
    with st.container(border=True):
        st.markdown("#### No dataset loaded yet")
        st.write("Upload a file, type in your own data, or pick a sample dataset to get started.")
        st.page_link("pages/home.py", label="Go to Home", icon=":material/arrow_back:")
    st.stop()


def describe_source(source):
    """turn the internal dataset source id into a short readable name."""
    if not source:
        return "No dataset"
    if source == "manual":
        return "Manually entered data"
    if source.startswith("file:"):
        return source.split(":")[1]
    if source.startswith("sample:"):
        return f"Sample: {source.split(':', 1)[1]}"
    return source


def sidebar_dataset_status():
    """a compact card in the sidebar showing which dataset is active."""
    df = st.session_state.get("raw_df")
    st.caption("ACTIVE DATASET")
    if df is None or df.empty:
        st.markdown(":gray[Nothing loaded yet]")
        return
    st.markdown(f"**{describe_source(st.session_state.get('dataset_source'))}**")
    st.markdown(f":gray[{len(df):,} rows · {len(df.columns)} columns]")
    steps = len(st.session_state.get("history", []))
    if steps:
        st.badge(f"{steps} cleaning step{'s' if steps != 1 else ''}", icon=":material/cleaning_services:", color="blue")

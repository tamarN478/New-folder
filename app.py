"""Main entrypoint for the multi-page Streamlit data app."""

import streamlit as st
from session import session_controls
from ui import sidebar_dataset_status

st.set_page_config(
    page_title="AI Data Explorer",
    page_icon=":material/analytics:",
    layout="wide",
)

pages = {
    "": [
        st.Page("pages/home.py", title="Home", icon=":material/home:", default=True),
    ],
    "Analyze": [
        st.Page("pages/eda_cleaning.py", title="Explore & Clean", icon=":material/query_stats:"),
        st.Page("pages/pivot_tables.py", title="Group & Pivot", icon=":material/pivot_table_chart:"),
        st.Page("pages/chart_builder.py", title="Chart Builder", icon=":material/bar_chart:"),
        st.Page("pages/dashboard.py", title="Dashboard", icon=":material/dashboard:"),
        st.Page("pages/ai_insights.py", title="AI Insights", icon=":material/auto_awesome:"),
    ],
}

navigation = st.navigation(pages)

# Pages rerun the app whenever they load or change the dataset, so this card is always current.
with st.sidebar:
    sidebar_dataset_status()
    st.divider()
    session_controls()
navigation.run()

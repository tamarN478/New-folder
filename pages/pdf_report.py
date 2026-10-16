"""report page: download a PDF summary of the dataset and your analysis."""

import streamlit as st
from report import SECTIONS, build_report
from ui import describe_source, page_header, require_dataset

page_header("PDF Report", "Download a tidy PDF with your data summary, quality check, charts and AI insights.")
df = require_dataset()

pinned = st.session_state.get("dashboard_charts", [])
history = [description for description, _ in st.session_state.get("history", [])]
chat = st.session_state.get("ai_display", [])


def _text(message):
    return "\n\n".join(part["text"] for part in message["parts"] if part["type"] == "text")


ai_answers = [
    (_text(chat[i]), _text(chat[i + 1]))
    for i in range(0, len(chat) - 1, 2)
    if chat[i]["role"] == "user" and chat[i + 1]["role"] == "assistant" and _text(chat[i + 1])
]

settings_col, info_col = st.columns([2, 1])
with settings_col, st.container(border=True):
    title = st.text_input("Report title", value=st.session_state.get("dashboard_title") or "Data report")
    available = [s for s in SECTIONS if s != "AI insights" or ai_answers]
    sections = st.pills("Include", available, selection_mode="multi", default=available)
    create = st.button("Create PDF", type="primary", icon=":material/picture_as_pdf:", disabled=not sections)

with info_col, st.container(border=True):
    st.markdown("##### What goes in")
    st.markdown(
        f"- **Charts:** {len(pinned)} from your dashboard" if pinned
        else "- **Charts:** a few automatic ones (pin charts to the dashboard to choose your own)"
    )
    st.markdown(f"- **Cleaning steps:** {len(history)}")
    st.markdown(f"- **AI answers:** {len(ai_answers)}" if ai_answers
                else "- **AI answers:** none yet (ask on the AI Insights page)")

if create:
    with st.spinner("Creating your report…"):
        st.session_state.report_pdf = build_report(
            df, title=title, source_name=describe_source(st.session_state.get("dataset_source")),
            sections=sections, history=history, chart_specs=pinned, ai_messages=ai_answers,
        )
        st.session_state.report_name = title or "report"

if st.session_state.get("report_pdf"):
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        st.markdown(f"**{st.session_state.report_name}.pdf** is ready "
                    f"({len(st.session_state.report_pdf) / 1024:.0f} KB).", width="stretch")
        st.download_button(
            "Download PDF", st.session_state.report_pdf,
            file_name=f"{st.session_state.report_name}.pdf", mime="application/pdf",
            icon=":material/download:", type="primary",
        )

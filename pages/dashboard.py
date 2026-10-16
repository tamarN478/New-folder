"""dashboard page: charts pinned from the Chart Builder, redrawn from the current data."""

import html

import streamlit as st
from charts import ChartError, build_chart, describe_spec
from ui import page_header, require_dataset


def move(charts, position, step):
    target = position + step
    if 0 <= target < len(charts):
        charts[position], charts[target] = charts[target], charts[position]


def dashboard_html(title, figures):
    """one self-contained HTML file with every chart, for sharing or printing."""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:Inter,system-ui,sans-serif;margin:32px;background:#fcfcfb;color:#1b1b19}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}"
        ".card{border:1px solid #e1e0d9;border-radius:10px;padding:8px;background:#fff}</style>",
        f"</head><body><h1>{html.escape(title)}</h1><div class='grid'>",
    ]
    for position, fig in enumerate(figures):
        plotly_js = "cdn" if position == 0 else False
        parts.append(f"<div class='card'>{fig.to_html(full_html=False, include_plotlyjs=plotly_js)}</div>")
    parts.append("</div></body></html>")
    return "".join(parts)


page_header("Dashboard", "Charts you pinned from the Chart Builder. They update when your data changes.")
df = require_dataset()
charts = st.session_state.setdefault("dashboard_charts", [])

if not charts:
    with st.container(border=True):
        st.markdown("#### Nothing pinned yet")
        st.write("Build a chart in the Chart Builder and click **Pin to dashboard** to add it here.")
        st.page_link("pages/chart_builder.py", label="Open Chart Builder", icon=":material/bar_chart:")
    st.stop()

toolbar = st.container(horizontal=True, vertical_alignment="bottom")
title = toolbar.text_input("Dashboard title", value=st.session_state.get("dashboard_title", "My dashboard"),
                           key="dashboard_title_input")
st.session_state.dashboard_title = title
per_row = toolbar.segmented_control("Charts per row", [1, 2, 3], default=2, required=True)

figures = []
for start in range(0, len(charts), per_row):
    row = st.columns(per_row)
    for column, position in zip(row, range(start, min(start + per_row, len(charts)))):
        spec = charts[position]
        with column.container(border=True):
            try:
                fig, _, _ = build_chart(df, spec)
            except ChartError as error:
                st.markdown(f"**{spec.get('title') or spec['chart_type']}**")
                st.warning(str(error), icon=":material/warning:")
            else:
                figures.append(fig)
                st.plotly_chart(fig, width="stretch", key=f"chart_{spec['id']}")
            st.caption(describe_spec(spec))
            controls = st.container(horizontal=True, gap="small")
            if controls.button("", icon=":material/arrow_back:", key=f"left_{spec['id']}",
                               help="Move earlier", disabled=position == 0):
                move(charts, position, -1)
                st.rerun()
            if controls.button("", icon=":material/arrow_forward:", key=f"right_{spec['id']}",
                               help="Move later", disabled=position == len(charts) - 1):
                move(charts, position, 1)
                st.rerun()
            if controls.button("Remove", icon=":material/close:", key=f"remove_{spec['id']}"):
                charts.pop(position)
                st.rerun()

st.divider()
footer = st.container(horizontal=True, gap="small")
footer.download_button(
    "Download dashboard (HTML)",
    dashboard_html(title, figures).encode("utf-8") if figures else b"",
    file_name=f"{title or 'dashboard'}.html",
    mime="text/html",
    icon=":material/download:",
    disabled=not figures,
    help="One file with every chart that you can open in any browser, share, or print to PDF.",
)
with footer.popover("Clear dashboard", icon=":material/delete_sweep:"):
    st.write("Remove all pinned charts?")
    if st.button("Yes, remove all", type="primary"):
        charts.clear()
        st.rerun()

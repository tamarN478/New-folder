"""build a PDF report: overview, statistics, data quality, cleaning steps, charts, AI insights."""

import io
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to images, never open windows
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402
from fpdf.fonts import FontFace  # noqa: E402

from charts import SERIES_LIGHT, ChartError, build_chart  # noqa: E402
from quality import quality_report  # noqa: E402

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
INK, MUTED, BORDER, ACCENT = "#1b1b19", "#6b6a65", "#e1e0d9", "#2a78d6"
SECTIONS = ["Overview", "Summary statistics", "Data quality", "Cleaning steps", "Charts", "AI insights"]


# ---------------------------------------------------------------------------
# charts as images (matplotlib, so no browser is needed)
# ---------------------------------------------------------------------------

def _style_axes(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=BORDER, linewidth=0.6)
    ax.set_axisbelow(True)


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def chart_image(df, spec):
    """draw a Chart Builder spec with matplotlib; returns a PNG buffer."""
    _, plot_df, _ = build_chart(df, spec)  # reuse the same grouping/aggregation as the app
    kind, x, y, color = spec["chart_type"], spec.get("x"), spec.get("y"), spec.get("color")
    title = spec.get("title") or kind
    fig, ax = plt.subplots(figsize=(7, 3.6))
    groups = list(plot_df.groupby(color, observed=True)) if color and kind != "Pie Chart" else [(None, plot_df)]

    if kind == "Correlation Heatmap":
        corr = df.select_dtypes("number").corr()
        image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr)), corr.columns, rotation=30, ha="right", fontsize=8, color=MUTED)
        ax.set_yticks(range(len(corr)), corr.columns, fontsize=8, color=MUTED)
        for i in range(len(corr)):
            for j in range(len(corr)):
                value = corr.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > 0.6 else INK)
        bar = fig.colorbar(image, ax=ax, shrink=0.8)
        bar.ax.tick_params(labelsize=8, colors=MUTED)
        bar.outline.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        return _png(fig)
    if kind == "Pie Chart":
        if y:
            values = plot_df.groupby(x, observed=True)[y].sum()
        else:
            values = df[x].astype(str).value_counts()
        values = values.sort_values(ascending=False)
        if len(values) > 8:
            values = pd.concat([values.iloc[:7], pd.Series({"Other": values.iloc[7:].sum()})])
        ax.pie(values, labels=values.index.astype(str), colors=SERIES_LIGHT[:len(values)], autopct="%1.0f%%",
               wedgeprops={"linewidth": 2, "edgecolor": "white"}, textprops={"fontsize": 8})
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        return _png(fig)

    for position, (name, part) in enumerate(groups):
        colour = SERIES_LIGHT[position % len(SERIES_LIGHT)]
        label = None if name is None else str(name)
        if kind == "Histogram":
            ax.hist(pd.to_numeric(part[x], errors="coerce").dropna(), bins=20, color=colour, alpha=0.85, label=label)
        elif kind == "Box Plot":
            column = y or x
            ax.boxplot([pd.to_numeric(part[column], errors="coerce").dropna()], positions=[position],
                       widths=0.5, patch_artist=True, boxprops={"facecolor": colour, "alpha": 0.6})
        elif kind == "Scatter Plot":
            ax.scatter(part[x], part[y], s=14, color=colour, label=label, alpha=0.8)
        elif kind == "Line Chart":
            ordered = part.sort_values(x)
            ax.plot(ordered[x], ordered[y], color=colour, linewidth=2, marker="o", markersize=3, label=label)
        else:  # bar chart
            offset = (position - (len(groups) - 1) / 2) * (0.8 / len(groups))
            categories = part[x].astype(str).tolist()
            positions = [plot_df[x].astype(str).drop_duplicates().tolist().index(c) + offset for c in categories]
            ax.bar(positions, part[y], width=0.8 / len(groups), color=colour, label=label)
            ticks = plot_df[x].astype(str).drop_duplicates().tolist()
            ax.set_xticks(range(len(ticks)), ticks, rotation=30 if len(ticks) > 6 else 0, ha="right" if len(ticks) > 6 else "center")
    _style_axes(ax, title)
    ax.set_xlabel(x or "", color=MUTED, fontsize=9)
    ax.set_ylabel(y or ("Count" if kind == "Histogram" else ""), color=MUTED, fontsize=9)
    if color and len(groups) > 1:
        ax.legend(frameon=False, fontsize=8)
    return _png(fig)


def automatic_specs(df, limit=4):
    """chart specs to use when nothing is pinned to the dashboard."""
    numeric = df.select_dtypes("number").columns.tolist()
    text = [c for c in df.columns if c not in numeric and 1 < df[c].nunique() <= 30]
    specs = [{"chart_type": "Histogram", "x": c, "title": f"Distribution of {c}"} for c in numeric[:2]]
    if text and numeric:
        specs.append({"chart_type": "Bar Chart", "x": text[0], "y": numeric[0], "aggregation": "Average",
                      "title": f"Average {numeric[0]} by {text[0]}"})
    if len(numeric) >= 2:
        specs.append({"chart_type": "Correlation Heatmap", "title": "Correlation between numeric columns"})
    return specs[:limit]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class ReportPDF(FPDF):
    def __init__(self, title):
        super().__init__(format="A4")
        self.report_title = title
        self.add_font("DejaVu", "", FONT_DIR / "DejaVuSans.ttf")
        self.add_font("DejaVu", "B", FONT_DIR / "DejaVuSans-Bold.ttf")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(16, 16, 16)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(MUTED)
        self.cell(0, 6, f"{self.report_title} · page {self.page_no()}", align="R")

    def heading(self, text):
        self.ln(4)
        self.set_font("DejaVu", "B", 14)
        self.set_text_color(INK)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(ACCENT)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.l_margin + 18, self.get_y())
        self.ln(3)

    def paragraph(self, text, size=10, color=INK, bold_markdown=False):
        self.set_font("DejaVu", "", size)
        self.set_text_color(color)
        self.multi_cell(0, size * 0.5, text, markdown=bold_markdown, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def data_table(self, df, widths=None):
        self.set_font("DejaVu", "", 8)
        self.set_text_color(INK)
        self.set_draw_color(BORDER)
        heading = FontFace(emphasis="BOLD", fill_color="#f3f2ee")
        with self.table(col_widths=widths, headings_style=heading, line_height=5.5,
                        borders_layout="HORIZONTAL_LINES", text_align="LEFT") as table:
            table.row([str(c) for c in df.columns])
            for _, values in df.iterrows():
                table.row(["" if pd.isna(v) else (f"{v:,.4g}" if isinstance(v, float) else str(v)) for v in values])
        self.ln(2)


def _plain_markdown(text):
    """keep **bold**, drop other markdown that fpdf can't draw."""
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return re.sub(r"(?<!\*)\*(?!\*)([^*]+)\*(?!\*)", r"\1", text)


def build_report(df, *, title, source_name, sections, history=(), chart_specs=(), ai_messages=()):
    """return the PDF report as bytes."""
    pdf = ReportPDF(title)
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 22)
    pdf.set_text_color(INK)
    pdf.multi_cell(0, 11, title, new_x="LMARGIN", new_y="NEXT")
    pdf.paragraph(f"{source_name} · {len(df):,} rows · {len(df.columns)} columns · "
                  f"created {datetime.now():%d %B %Y, %H:%M}", size=9, color=MUTED)

    if "Overview" in sections:
        pdf.heading("Overview")
        overview = pd.DataFrame({
            "Column": [str(c) for c in df.columns],
            "Type": [str(t) for t in df.dtypes],
            "Missing": [int(df[c].isna().sum()) for c in df.columns],
            "Distinct values": [int(df[c].nunique()) for c in df.columns],
        })
        pdf.data_table(overview, widths=(40, 20, 15, 25))

    numeric = df.select_dtypes("number")
    if "Summary statistics" in sections and not numeric.empty:
        pdf.heading("Summary statistics")
        stats = numeric.describe().T[["mean", "std", "min", "50%", "max"]].rename(
            columns={"mean": "Average", "std": "Spread (std)", "min": "Min", "50%": "Median", "max": "Max"})
        pdf.data_table(stats.reset_index(names="Column"), widths=(30, 15, 15, 15, 15, 15))

    if "Data quality" in sections:
        score, _, issues = quality_report(df)
        pdf.heading("Data quality")
        pdf.paragraph(f"**Quality score: {score}/100.** "
                      + ("No problems were found." if not issues else f"{len(issues)} problem(s) found:"),
                      bold_markdown=True)
        for issue in issues:
            where = issue.column or "Whole dataset"
            pdf.paragraph(f"• **{where}**: {issue.problem}", size=9, bold_markdown=True)

    if "Cleaning steps" in sections:
        pdf.heading("Cleaning steps")
        if history:
            for position, description in enumerate(history, 1):
                pdf.paragraph(f"{position}. {description}", size=9)
        else:
            pdf.paragraph("No cleaning steps were applied.", size=9, color=MUTED)

    if "Charts" in sections:
        specs = list(chart_specs) or automatic_specs(df)
        if specs:
            if pdf.get_y() > pdf.h - 120:
                pdf.add_page()  # keep the heading on the same page as the first chart
            pdf.heading("Charts")
            for spec in specs:
                try:
                    image = chart_image(df, spec)
                except (ChartError, ValueError, TypeError, KeyError) as error:
                    pdf.paragraph(f"Couldn't draw '{spec.get('title') or spec['chart_type']}': {error}",
                                  size=9, color=MUTED)
                    continue
                if pdf.get_y() > pdf.h - 110:
                    pdf.add_page()
                pdf.image(image, w=pdf.epw)
                pdf.ln(4)

    answers = [(q, a) for q, a in ai_messages]
    if "AI insights" in sections and answers:
        pdf.heading("AI insights")
        for question, answer in answers:
            pdf.paragraph(f"**{question}**", size=10, bold_markdown=True)
            pdf.paragraph(_plain_markdown(answer), size=9, bold_markdown=True)
        pdf.paragraph("Written by AI (Claude). Double-check important numbers.", size=8, color=MUTED)

    return bytes(pdf.output())

"""ai dataset insights page: quick statistics plus an AI analyst that can chart, clean and calculate."""

import os

import anthropic
import streamlit as st
from ai_agent import dataset_fingerprint, describe_dataset, run_turn
from ui import page_header, require_dataset

# label shown on the button -> (icon, request sent to the AI)
SUGGESTIONS = {
    "Summarize this dataset": (
        ":material/summarize:",
        "Give me a short overview of this dataset: what it seems to be about, what each column "
        "contains, key statistics, notable patterns or outliers, and any data quality issues. "
        "Finish with 3 questions worth exploring.",
    ),
    "Chart the key pattern": (":material/bar_chart:",
                              "Find the most interesting pattern in this data and show it in a chart."),
    "Fix data quality problems": (":material/auto_fix_high:",
                                  "Check the data for quality problems and fix the ones that are safe to fix."),
    "Which group does best?": (":material/leaderboard:",
                               "Which group or category stands out the most on the main numeric column?"),
}
AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}


def get_api_key():
    """find an API key in the environment, Streamlit secrets, or the sidebar."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass  # no secrets.toml file
    return st.sidebar.text_input(
        "Anthropic API key",
        type="password",
        key="anthropic_api_key",
        help="Kept only for this browser session. Get one at console.anthropic.com.",
    )


def show_statistical_insights(df):
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        st.caption("Not enough numeric columns to look for correlations.")
        return
    corr_matrix = numeric_df.corr()
    strong = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.6:
                strong.append({
                    "Columns": f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}",
                    "Direction": "Positive" if val > 0 else "Negative",
                    "Correlation": val,
                })
    if not strong:
        st.caption("No strong linear correlations (above 0.6 either way) among numeric columns.")
        return
    st.caption("Pairs of numeric columns that strongly move together (or in opposite directions).")
    st.dataframe(
        sorted(strong, key=lambda row: -abs(row["Correlation"])),
        hide_index=True,
        width="stretch",
        column_config={"Correlation": st.column_config.NumberColumn(format="%.2f")},
    )


def show_part(part):
    if part["type"] == "text":
        st.markdown(part["text"])
    elif part["type"] == "chart":
        st.plotly_chart(part["figure"], width="stretch")
    elif part["type"] == "action":
        st.markdown(f":blue-badge[:material/auto_fix_high: Changed your data] {part['text']}")


def ask(client, question, shown_as, df):
    """send one question (with a data update if the data changed) and show the answer."""
    transcript = st.session_state.ai_transcript
    content = []
    fingerprint = dataset_fingerprint(df)
    if st.session_state.get("ai_data_fingerprint") != fingerprint:
        content.append({"type": "text", "text": describe_dataset(df)[0]})
    content.append({"type": "text", "text": question})

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(shown_as)

    turn_start = len(transcript)
    transcript.append({"role": "user", "content": content})
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        def show_text():
            box = st.empty()
            return lambda text: box.markdown(text + " ▌")

        def show_item(item):
            show_part(item)

        try:
            with st.spinner("Thinking…", show_time=True):
                parts = run_turn(client, transcript, show_text, show_item)
        except anthropic.AuthenticationError:
            error = "That API key was rejected. Check it in the sidebar and try again."
        except anthropic.PermissionDeniedError:
            error = "This API key doesn't have access to the model. Check your Anthropic account."
        except anthropic.RateLimitError:
            error = "Too many requests right now. Wait a minute and try again."
        except anthropic.BadRequestError as bad:
            error = f"The request was rejected: {bad.message}"
        except anthropic.APIStatusError as failed:
            error = f"The AI service had a problem ({failed.status_code}). Try again shortly."
        except anthropic.APIConnectionError:
            error = "Couldn't reach the AI service. Check your internet connection."
        else:
            # Remember the data the AI was shown. If it changed the data itself, the next
            # question sends the new version.
            st.session_state.ai_data_fingerprint = fingerprint
            st.session_state.ai_display.append({"role": "user", "parts": [{"type": "text", "text": shown_as}]})
            st.session_state.ai_display.append({"role": "assistant", "parts": parts})
            st.rerun()  # redraw the page (and sidebar) with any data the AI changed
        # The turn didn't finish: drop it so the conversation stays valid for the next question.
        del transcript[turn_start:]
        st.error(error)


page_header("AI Insights", "Ask questions in plain language. The AI can also make charts and clean your data.")
df = require_dataset()

with st.expander("Quick statistical insights (no AI needed)", icon=":material/monitoring:"):
    show_statistical_insights(df)

# A different dataset makes the old conversation meaningless, so start over.
if st.session_state.get("ai_chat_source") != st.session_state.get("dataset_source"):
    st.session_state.ai_transcript = []
    st.session_state.ai_display = []
    st.session_state.pop("ai_data_fingerprint", None)
    st.session_state.ai_chat_source = st.session_state.get("dataset_source")
st.session_state.setdefault("ai_transcript", [])
display = st.session_state.setdefault("ai_display", [])

api_key = get_api_key()
if not api_key:
    st.info(
        "To use the AI chat, enter an **Anthropic API key** in the sidebar.\n\n"
        "1. Create a key at [console.anthropic.com](https://console.anthropic.com/) under API Keys.\n"
        "2. Paste it into the box in the sidebar.\n\n"
        "To avoid pasting it every time, add `ANTHROPIC_API_KEY = \"your-key\"` to a file named "
        "`.streamlit/secrets.toml` in the project folder."
    )
    st.stop()

caption = ("Your dataset is sent to Anthropic's Claude AI. The AI can make charts and change your data when "
           "you ask it to; every change can be undone on the Explore & Clean page. Double-check important numbers.")
if describe_dataset(df)[1]:
    caption += " Your data is large, so the AI sees all summary statistics but only the first part of the rows."
st.caption(caption)

for message in display:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        for part in message["parts"]:
            show_part(part)

question = shown_as = None
if not display:
    suggestions = st.container(horizontal=True, gap="small")
    for label, (icon, prompt) in SUGGESTIONS.items():
        if suggestions.button(label, icon=icon):
            question, shown_as = prompt, label
elif st.button("Clear chat", icon=":material/delete_sweep:"):
    st.session_state.ai_transcript = []
    st.session_state.ai_display = []
    st.session_state.pop("ai_data_fingerprint", None)
    st.rerun()

typed = st.chat_input("Ask a question, or ask for a chart or a fix, e.g. 'chart average tip by day'")
if typed:
    question, shown_as = typed, typed

if question:
    ask(anthropic.Anthropic(api_key=api_key), question, shown_as, df)

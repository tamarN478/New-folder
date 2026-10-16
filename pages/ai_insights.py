"""ai dataset insights page: quick statistics plus a chat that answers questions about the data."""

import os

import anthropic
import pandas as pd
import streamlit as st
from ui import page_header, require_dataset

MODEL = "claude-opus-5-5"
# Above this size only the first rows are sent, to keep requests fast and affordable.
MAX_DATA_CHARS = 300_000
# label shown on the button -> (icon, question sent to the AI); None means the summary prompt.
SUGGESTED_QUESTIONS = {
    "Summarize this dataset": (":material/summarize:", None),
    "Find interesting patterns": (":material/insights:", "What are the most interesting patterns?"),
    "Check data quality": (":material/rule:", "Are there any data quality problems?"),
    "Suggest charts": (":material/bar_chart:", "Which charts would best show this data?"),
}
AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}
SUMMARY_PROMPT = (
    "Give me a short overview of this dataset: what it seems to be about, what each "
    "column contains, key statistics, notable patterns or outliers, and any data quality "
    "issues (missing values, odd values, duplicates). Finish with 3 questions worth exploring."
)
SYSTEM_PROMPT = """You are a friendly data analyst inside a data exploration app. The user \
has loaded a dataset, which is included below. Answer their questions about it.

- Base every answer on the data provided. If a question can't be answered from it, say so.
- When you compute something (averages, counts, rankings), work carefully from the actual \
rows and show the key numbers. If only a sample of the rows is included, say that your \
answer is based on the sample.
- The user may be new to data analysis: explain in plain language and keep answers short.
- Use Markdown: short paragraphs, bullet points, and small tables where they help.
- The app also has an "EDA & Cleaning" page (filters, cleaning tools, automatic charts) and a \
"Chart Builder" page (bar, line, scatter, histogram, box, pie, correlation heatmap with \
sum/average grouping). Point the user to them when relevant."""


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


def describe_dataset(df):
    """build the text description of the dataset that is sent to the model."""
    columns = "\n".join(
        f"- {column} ({dtype}, {df[column].isnull().sum()} missing, "
        f"{df[column].nunique()} unique values)"
        for column, dtype in df.dtypes.items()
    )
    stats = df.describe(include="all").to_string()
    csv_text = df.to_csv(index=False)
    sampled = len(csv_text) > MAX_DATA_CHARS
    if sampled:
        rows_that_fit = max(10, int(len(df) * MAX_DATA_CHARS / len(csv_text)))
        csv_text = df.head(rows_that_fit).to_csv(index=False)
        rows_note = f"Only the first {rows_that_fit} of {len(df)} rows are included below."
    else:
        rows_note = f"All {len(df)} rows are included below."

    description = (
        f"Dataset: {len(df)} rows x {len(df.columns)} columns.\n\n"
        f"Columns:\n{columns}\n\n"
        f"Summary statistics (computed on ALL rows):\n{stats}\n\n"
        f"{rows_note}\n<data format=\"csv\">\n{csv_text}</data>"
    )
    return description, sampled


def stream_answer(client, data_description, messages):
    """yield the model's answer piece by piece as it is written."""
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=[
            {"type": "text", "text": SYSTEM_PROMPT},
            # The dataset is identical across chat turns, so cache it to make follow-ups cheaper.
            {"type": "text", "text": data_description, "cache_control": {"type": "ephemeral"}},
        ],
        messages=messages,
        output_config={"effort": "medium"},
        # If a safety check declines the request, retry it on Anthropic's recommended model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()

    if final.stop_reason == "refusal":
        yield "\n\n_The AI declined to answer this question. Try rephrasing it._"
    elif final.stop_reason == "max_tokens":
        yield "\n\n_(The answer was cut off because it got too long. Ask for a shorter version.)_"


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


page_header("AI Insights", "Ask questions about your data in plain language.")
df = require_dataset()

with st.expander("Quick statistical insights (no AI needed)", icon=":material/monitoring:"):
    show_statistical_insights(df)

# A different dataset makes the old conversation meaningless, so start over.
if st.session_state.get("ai_chat_source") != st.session_state.get("dataset_source"):
    st.session_state.ai_messages = []
    st.session_state.ai_chat_source = st.session_state.get("dataset_source")
messages = st.session_state.setdefault("ai_messages", [])

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

data_description, sampled = describe_dataset(df)
caption = "Your dataset is sent to Anthropic's Claude AI to answer questions. Answers can contain mistakes, so double-check important numbers."
if sampled:
    caption += " Your data is large, so the AI sees summary statistics for all rows but only the first part of the rows themselves."
st.caption(caption)

for message in messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message.get("display", message["content"]))

question = shown_as = None
if not messages:
    suggestions = st.container(horizontal=True, gap="small")
    for label, (icon, prompt) in SUGGESTED_QUESTIONS.items():
        if suggestions.button(label, icon=icon):
            question, shown_as = prompt or SUMMARY_PROMPT, label
elif st.button("Clear chat", icon=":material/delete_sweep:"):
    st.session_state.ai_messages = []
    st.rerun()

typed = st.chat_input("Ask a question about your data, e.g. 'Which group has the highest average?'")
if typed:
    question, shown_as = typed, None

if question:
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(shown_as or question)
    # Only role and content go to the API; "display" is just what this page shows.
    request_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    request_messages.append({"role": "user", "content": question})

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        client = anthropic.Anthropic(api_key=api_key)
        try:
            answer = st.write_stream(stream_answer(client, data_description, request_messages))
        except anthropic.AuthenticationError:
            st.error("That API key was rejected. Check it in the sidebar and try again.")
        except anthropic.PermissionDeniedError:
            st.error("This API key doesn't have access to the model. Check your Anthropic account.")
        except anthropic.RateLimitError:
            st.error("Too many requests right now. Wait a minute and try again.")
        except anthropic.BadRequestError as error:
            st.error(f"The request was rejected: {error.message}")
        except anthropic.APIStatusError as error:
            st.error(f"The AI service had a problem ({error.status_code}). Try again shortly.")
        except anthropic.APIConnectionError:
            st.error("Couldn't reach the AI service. Check your internet connection.")
        else:
            messages.append({"role": "user", "content": question, "display": shown_as or question})
            messages.append({"role": "assistant", "content": answer})

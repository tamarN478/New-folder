"""save the whole working session to a file and open it again later.

The file is a zip with the tables stored as Parquet and everything else as JSON.
It never contains code, so opening a session file from someone else is safe.
"""

import io
import json
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st

FORMAT_VERSION = 1
# Tables kept in st.session_state that make up a session.
SESSION_TABLES = ["raw_df", "original_df", "manual_grid", "manual_grid_draft", "manual_pushed_df"]
# Plain values (anything JSON can store) kept in st.session_state.
SESSION_VALUES = ["dataset_source", "input_method_choice", "manual_hidden", "dashboard_charts"]


def _parquet_bytes(df):
    buffer = io.BytesIO()
    try:
        df.to_parquet(buffer)
    except Exception:
        # Columns mixing numbers and text can't be stored as-is; store them as text.
        safe = df.copy()
        safe.columns = [str(c) for c in safe.columns]
        for column in safe.select_dtypes("object").columns:
            safe[column] = safe[column].map(lambda v: None if pd.isna(v) else str(v))
        buffer = io.BytesIO()
        safe.to_parquet(buffer)
    return buffer.getvalue()


def snapshot():
    """grab references to everything a session file needs (cheap: no copying)."""
    state = st.session_state
    return {
        "tables": {name: state[name] for name in SESSION_TABLES if isinstance(state.get(name), pd.DataFrame)},
        "values": {name: state[name] for name in SESSION_VALUES if name in state},
        "history": list(state.get("history", [])),
    }


def build_session_file(snap):
    """the bytes of a .zip session file for a snapshot()."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, df in snap["tables"].items():
            archive.writestr(f"tables/{name}.parquet", _parquet_bytes(df))
        for position, (_, df) in enumerate(snap["history"]):
            archive.writestr(f"history/{position}.parquet", _parquet_bytes(df))
        archive.writestr("session.json", json.dumps({
            "format_version": FORMAT_VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "values": snap["values"],
            "history": [description for description, _ in snap["history"]],
        }, indent=2, default=str))
    return buffer.getvalue()


def read_session_file(data):
    """parse a session file; raises ValueError with a readable message if it isn't one."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        meta = json.loads(archive.read("session.json"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
        raise ValueError("This isn't a session file saved by this app.") from None
    if meta.get("format_version", 0) > FORMAT_VERSION:
        raise ValueError("This session was saved by a newer version of the app.")

    def table(path):
        return pd.read_parquet(io.BytesIO(archive.read(path)))

    tables = {name: table(f"tables/{name}.parquet") for name in SESSION_TABLES
              if f"tables/{name}.parquet" in archive.namelist()}
    if "raw_df" not in tables:
        raise ValueError("This session file doesn't contain any data.")
    history = [(description, table(f"history/{i}.parquet")) for i, description in enumerate(meta["history"])]
    return {"tables": tables, "values": meta.get("values", {}), "history": history,
            "saved_at": meta.get("saved_at")}


def restore(session):
    """put a read_session_file() result back into st.session_state."""
    state = st.session_state
    for name in SESSION_TABLES + SESSION_VALUES + ["history"]:
        state.pop(name, None)
    for name, df in session["tables"].items():
        state[name] = df
    for name, value in session["values"].items():
        state[name] = value
    state.history = session["history"]
    # Let widgets pick up the restored values instead of keeping their old ones.
    state.pop("data_input_method", None)
    state.manual_editor_version = state.get("manual_editor_version", 0) + 1
    state.ai_messages = []


def session_controls():
    """save / open buttons for the sidebar."""
    st.caption("SESSION")
    row = st.container(horizontal=True, gap="small")
    has_data = isinstance(st.session_state.get("raw_df"), pd.DataFrame) and not st.session_state.raw_df.empty
    snap = snapshot()
    row.download_button(
        "Save", data=lambda: build_session_file(snap),
        file_name=f"data-explorer-session-{datetime.now():%Y%m%d-%H%M}.zip",
        mime="application/zip", icon=":material/save:", disabled=not has_data,
        help="Download your data, cleaning steps and charts as one file.",
        on_click="ignore",
    )
    with row.popover("Open", icon=":material/folder_open:"):
        uploaded = st.file_uploader("Session file", type=["zip"], key="session_upload")
        if uploaded is not None:
            upload_id = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("restored_session_id") != upload_id:
                try:
                    session = read_session_file(uploaded.getvalue())
                except ValueError as error:
                    st.error(str(error))
                else:
                    restore(session)
                    st.session_state.restored_session_id = upload_id
                    st.rerun()
            else:
                st.caption("Session opened.")

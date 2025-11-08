# Log_App.py
# Streamlit Website Log Analyzer — AI Search & Bot Insights (final robust timestamp filter)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import (
    guess_columns,
    parse_timestamp_column,
    classify_user_agent,
    BOT_SIGNATURES
)

st.set_page_config(page_title="Website Log Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("Website Log Analyzer — AI Search & Bot Insights")

# Sidebar — Upload
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV log file", type=["csv", "txt"])

@st.cache_data
def load_csv(file):
    try:
        return pd.read_csv(file, low_memory=False)
    except Exception:
        return pd.read_csv(file, engine="python", low_memory=False)

if not uploaded_file:
    st.info("Upload your CSV file (e.g., detailed_hits.csv) to start analysis.")
    st.stop()

df = load_csv(uploaded_file)
st.sidebar.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
guessed = guess_columns(df.columns.tolist())

# Column mapping
st.sidebar.header("Column Mapping")
col_ts = st.sidebar.selectbox("Timestamp column", ["(none)"] + df.columns.tolist(),
                              index=(1 if guessed.get("timestamp") in df.columns else 0))
col_ua = st.sidebar.selectbox("User-Agent column", ["(none)"] + df.columns.tolist(),
                              index=(1 if guessed.get("user_agent") in df.columns else 0))
col_path = st.sidebar.selectbox("Path/URL column", ["(none)"] + df.columns.tolist(),
                                index=(1 if guessed.get("path") in df.columns else 0))
col_status = st.sidebar.selectbox("Status column", ["(none)"] + df.columns.tolist(),
                                  index=(1 if guessed.get("status") in df.columns else 0))
col_bytes = st.sidebar.selectbox("Bytes/Size column", ["(none)"] + df.columns.tolist(),
                                 index=(1 if guessed.get("bytes") in df.columns else 0))
col_rt = st.sidebar.selectbox("Response time (ms) column", ["(none)"] + df.columns.tolist(),
                              index=(1 if guessed.get("response_time") in df.columns else 0))

working = df.copy()

# --- Robust timestamp parsing and enforcement ---
timestamp_parsed = False

# 1. Try user-selected column
if col_ts != "(none)":
    working = parse_timestamp_column(working, col_ts)
    if "timestamp" in working.columns and working["timestamp"].notna().any():
        timestamp_parsed = True

# 2. Try auto-detect if user didn't select correctly
if not timestamp_parsed:
    for c in working.columns:
        if any(k in c.lower() for k in ["time", "date", "timestamp"]):
            working = parse_timestamp_column(working, c)
            if "timestamp" in working.columns and working["timestamp"].notna().any():
                st.sidebar.success(f"Auto-detected timestamp column: {c}")
                timestamp_parsed = True
                break

# 3. Force cast to datetime even if still wrong
if "timestamp" not in working.columns and col_ts != "(none)":
    working["timestamp"] = working[col_ts]
working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=False)
working = working.dropna(subset=["timestamp"])
if not pd.api.types.is_datetime64_any_dtype(working["timestamp"]):
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=False)

# Other columns
working["user_agent"] = working[col_ua].astype(str) if col_ua != "(none)" else ""
working["path"] = working[col_path].astype(str) if col_path != "(none)" else ""
working["status"] = working[col_status].astype(str) if col_status != "(none)" else ""
working["bytes"] = pd.to_numeric(working[col_bytes], errors="coerce") if col_bytes != "(none)" else np.nan
working["response_time"] = pd.to_numeric(working[col_rt], errors="coerce") if col_rt != "(none)" else np.nan

# Classify bots
working["agent_class"] = working["user_agent"].apply(classify_user_agent)
working["is_bot"] = working["agent_class"].apply(lambda x: x != "human")

# Filters
st.sidebar.header("Filters")

min_date, max_date = None, None
if "timestamp" in working.columns and working["timestamp"].notna().any():
    min_date = working["timestamp"].min()
    max_date = working["timestamp"].max()

default_range = (min_date.date(), max_date.date()) if min_date is not None and pd.notna(min_date) and max_date is not None and pd.notna(max_date) else None
date_range = st.sidebar.date_input("Date range", value=default_range)
top_n = st.sidebar.slider("Top N items", 5, 50, 10)

# --- Safe date filtering ---
if "timestamp" in working.columns and working["timestamp"].notna().any() and isinstance(date_range, tuple) and len(date_range) == 2:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    # Ensure comparison is between datetimes only
    valid_mask = working["timestamp"].apply(lambda x: isinstance(x, pd.Timestamp))
    working = working.loc[valid_mask]
    mask = (working["timestamp"] >= start) & (working["timestamp"] <= end)
    working = working.loc[mask]

# KPIs
k1, k2, k3, k4 = st.columns(4)
total_hits = len(working)
bot_hits = int(working["is_bot"].sum())
unique_pages = working["path"].nunique()
avg_resp = working["response_time"].mean() if "response_time" in working.columns else np.nan

k1.metric("Total hits", f"{total_hits:,}")
k2.metric("Bot hits", f"{bot_hits:,}", f"{(bot_hits/total_hits*100):.1f}%" if total_hits else "0%")
k3.metric("Unique pages", f"{unique_pages:,}")
k4.metric("Avg response (ms)", f"{avg_resp:.0f}" if not np.isnan(avg_resp) else "n/a")

st.markdown("---")

# Time-series
if "timestamp" in working.columns and working["timestamp"].notna().any():
    ts = working.dropna(subset=["timestamp"]).copy().set_index("timestamp").sort_index()
    try:
        ts = ts.resample("D").agg({"is_bot": "sum", "path": "count"})
        ts["human_hits"] = ts["path"] - ts["is_bot"]
        ts = ts.reset_index().melt(id_vars="timestamp", value_vars=["path", "is_bot", "human_hits"],
                                   var_name="series", value_name="count")
        fig_ts = px.line(ts, x="timestamp", y="count", color="series", title="Daily Hits (Bots vs Humans)")
        st.plotly_chart(fig_ts, use_container_width=True)
    except Exception as e:
        st.warning(f"Unable to resample: {e}")
else:
    st.warning("Timestamp column missing or invalid for resampling.")

# Status codes
if "status" in working.columns:
    st.subheader("Status Code Distribution")
    status_df = working["status"].value_counts().reset_index()
    status_df.columns = ["status", "count"]
    fig_status = px.pie(status_df, names="status", values="count", title="HTTP Status Codes")
    st.plotly_chart(fig_status, use_container_width=True)

# Top pages crawled
st.subheader(f"Top {top_n} Pages Crawled by Bots")
bots_only = working[working["is_bot"]]
if not bots_only.empty:
    top_pages = bots_only["path"].value_counts().reset_index().head(top_n)
    top_pages.columns = ["path", "bot_hits"]
    fig_pages = px.bar(top_pages, x="bot_hits", y="path", orientation="h", title="Top Bot-Crawled Pages")
    st.plotly_chart(fig_pages, use_container_width=True)

# Bot breakdown
st.subheader("Bot Type Breakdown")
agent_counts = working["agent_class"].value_counts().reset_index()
agent_counts.columns = ["agent_class", "count"]
fig_agents = px.bar(agent_counts, x="agent_class", y="count", title="Agent Class Distribution")
st.plotly_chart(fig_agents, use_container_width=True)

# Response size
if working["bytes"].notna().sum() > 0:
    st.subheader("Response Size Distribution")
    fig_size = px.histogram(working, x="bytes", nbins=50, title="Response Size (bytes)")
    st.plotly_chart(fig_size, use_container_width=True)

# Hourly pattern
if "timestamp" in working.columns and working["timestamp"].notna().any():
    st.subheader("Hourly Hit Pattern")
    hourly = working.copy()
    hourly["hour"] = hourly["timestamp"].dt.hour
    hour_counts = hourly.groupby(["hour", "is_bot"]).size().reset_index(name="count")
    hour_counts["type"] = hour_counts["is_bot"].map({True: "bot", False: "human"})
    fig_hour = px.line(hour_counts, x="hour", y="count", color="type", markers=True, title="Hits by Hour of Day")
    st.plotly_chart(fig_hour, use_container_width=True)

# Explorer
st.subheader("Data Explorer (first 1000 rows)")
st.dataframe(working.head(1000), use_container_width=True)

st.download_button("Download Filtered CSV",
                   data=working.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_hits.csv")

st.markdown("---")
st.sidebar.markdown("---")
st.sidebar.write("Known bot signatures are predefined.")
if st.sidebar.checkbox("Show Bot Signatures"):
    st.sidebar.write(BOT_SIGNATURES)

st.success("Analysis complete.")

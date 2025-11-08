# Log_App.py
# Streamlit Bot & AI Traffic Log Analyzer
# v3 — unified, standalone version (no utils.py required)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Bot & AI Log Analyzer", layout="wide")

# ----------------------------
# BOT SIGNATURES
# ----------------------------
TRADITIONAL_BOTS = [
    "googlebot", "bingbot", "yandex", "duckduckbot", "baiduspider", "slurp", "ahrefsbot", "semrushbot"
]
AI_BOTS = [
    "gptbot", "chatgpt-user", "openai", "oai-searchbot", "perplexity", "perplexitybot",
    "claude", "claudebot", "anthropic", "mistral", "bytespider", "ccbot", "serpapi",
    "copilot", "gpt-4o", "oai", "diffbot", "pi.ai"
]
STATIC_EXTENSIONS = [
    ".css", ".js", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".map", ".eot"
]

# ----------------------------
# HELPERS
# ----------------------------
def normalize_timestamp(series):
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s.dt.tz_convert(None)

def classify_agent(ua):
    if not isinstance(ua, str) or ua.strip() == "":
        return "unknown"
    ua = ua.lower()
    for s in AI_BOTS:
        if s in ua:
            return "ai:" + s
    for s in TRADITIONAL_BOTS:
        if s in ua:
            return "traditional:" + s
    if "bot" in ua or "spider" in ua or "crawler" in ua:
        return "other_bot"
    return "human"

def detect_static(path):
    if not isinstance(path, str) or path.strip() == "":
        return False
    path = path.lower().split("?")[0]
    return any(path.endswith(ext) for ext in STATIC_EXTENSIONS)

# ----------------------------
# SIDEBAR — UPLOAD
# ----------------------------
st.sidebar.header("Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload CSV log file", type=["csv", "txt"])

@st.cache_data
def load_csv(file):
    try:
        return pd.read_csv(file, low_memory=False)
    except Exception:
        return pd.read_csv(file, engine="python", low_memory=False)

if not uploaded:
    st.info("Upload your CSV file to start analysis.")
    st.stop()

df = load_csv(uploaded)
st.sidebar.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

# ----------------------------
# COLUMN MAPPING
# ----------------------------
cols = df.columns.tolist()
def find_col(*names):
    for n in names:
        for c in cols:
            if n.lower() in c.lower():
                return c
    return None

col_time = st.sidebar.selectbox("Timestamp column", ["(none)"] + cols, index=(cols.index(find_col("time")) + 1 if find_col("time") else 0))
col_ua = st.sidebar.selectbox("User-Agent column", ["(none)"] + cols, index=(cols.index(find_col("user")) + 1 if find_col("user") else 0))
col_path = st.sidebar.selectbox("Path/URL column", ["(none)"] + cols, index=(cols.index(find_col("path")) + 1 if find_col("path") else 0))
col_status = st.sidebar.selectbox("Status column", ["(none)"] + cols, index=(cols.index(find_col("status")) + 1 if find_col("status") else 0))
col_section = st.sidebar.selectbox("Section column (optional)", ["(none)"] + cols, index=(cols.index(find_col("section")) + 1 if find_col("section") else 0))
col_static = st.sidebar.selectbox("IsStatic column (optional)", ["(none)"] + cols, index=(cols.index(find_col("static")) + 1 if find_col("static") else 0))

# ----------------------------
# PREPARE DATA
# ----------------------------
df["timestamp"] = normalize_timestamp(df[col_time]) if col_time != "(none)" else pd.NaT
df = df[df["timestamp"].notna()].copy()

df["user_agent"] = df[col_ua].astype(str) if col_ua != "(none)" else ""
df["path"] = df[col_path].astype(str) if col_path != "(none)" else ""
df["status"] = df[col_status].astype(str).str.strip() if col_status != "(none)" else "unknown"
df["section"] = df[col_section] if col_section != "(none)" else ""

if col_static != "(none)":
    df["is_static"] = df[col_static].astype(str).str.upper().isin(["TRUE", "1", "YES"])
else:
    df["is_static"] = df["path"].apply(detect_static)

df["agent_class"] = df["user_agent"].apply(classify_agent)
df["agent_group"] = df["agent_class"].apply(
    lambda x: "ai" if x.startswith("ai:") else ("traditional" if x.startswith("traditional:") else ("bot" if "bot" in x else "human"))
)
df["is_bot"] = df["agent_group"].isin(["ai", "traditional", "bot"])

df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour

# ----------------------------
# FILTERS
# ----------------------------
st.sidebar.header("Filters")

bot_types = sorted(df["agent_group"].unique())
selected_groups = st.sidebar.multiselect("Bot Type(s)", options=bot_types, default=["ai", "traditional"])

agents = sorted(df["agent_class"].unique())
selected_agents = st.sidebar.multiselect("Specific Agents (optional)", options=agents, default=[])

all_statuses = sorted(df["status"].unique())
default_statuses = [s for s in ["200", "304", "404"] if s in all_statuses]
selected_statuses = st.sidebar.multiselect(
    "Status Codes",
    options=all_statuses,
    default=default_statuses,
    help="Filter status codes included in analysis."
)

content_filter = st.sidebar.radio("URL Scope", ["All URLs", "Content-only (exclude static assets)"], index=1)

min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date Range", (min_date, max_date))
start = pd.Timestamp(date_range[0])
end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)

top_n = st.sidebar.slider("Top N URLs", 5, 100, 25)

# Apply filters
df_f = df.copy()
if selected_agents:
    df_f = df_f[df_f["agent_class"].isin(selected_agents)]
else:
    df_f = df_f[df_f["agent_group"].isin(selected_groups)]

if selected_statuses:
    df_f = df_f[df_f["status"].isin(selected_statuses)]

if content_filter == "Content-only (exclude static assets)":
    df_f = df_f[~df_f["is_static"]]

df_f = df_f[(df_f["timestamp"] >= start) & (df_f["timestamp"] < end)]

# ----------------------------
# KPIs
# ----------------------------
st.title("Bot & AI Traffic Dashboard")
c1, c2, c3, c4 = st.columns(4)
total_hits = len(df_f)
bot_hits = df_f["is_bot"].sum()
ai_hits = len(df_f[df_f["agent_group"] == "ai"])
pages = df_f["path"].nunique()

c1.metric("Total Hits", f"{total_hits:,}")
c2.metric("Bot Hits", f"{bot_hits:,}", f"{bot_hits/total_hits*100:.1f}%" if total_hits else "0%")
c3.metric("AI Bot Hits", f"{ai_hits:,}")
c4.metric("Unique URLs", f"{pages:,}")

st.markdown("---")

# ----------------------------
# VISUALS
# ----------------------------

# Traffic Over Time
if not df_f.empty:
    st.subheader("Bot Traffic Over Time (Hourly)")
    ts = df_f.groupby([pd.Grouper(key="timestamp", freq="H"), "agent_group"]).size().reset_index(name="hits")
    fig = px.area(ts, x="timestamp", y="hits", color="agent_group", title="Hourly Traffic by Bot Type")
    st.plotly_chart(fig, use_container_width=True)

# Status Codes
st.subheader("HTTP Status Codes")
status_df = df_f.groupby(["status", "agent_group"]).size().reset_index(name="count")
fig = px.bar(status_df, x="status", y="count", color="agent_group", barmode="group", title="Status Codes by Bot Type")
st.plotly_chart(fig, use_container_width=True)

# Top URLs
st.subheader(f"Top {top_n} URLs by Bot Hits")
url_stats = (
    df_f.groupby("path")
    .agg(
        hits=("path", "count"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        common_status=("status", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
    )
    .reset_index()
    .sort_values("hits", ascending=False)
    .head(top_n)
)
url_stats["first_seen"] = url_stats["first_seen"].dt.strftime("%Y-%m-%d %H:%M")
url_stats["last_seen"] = url_stats["last_seen"].dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(url_stats, use_container_width=True)
fig_urls = px.bar(url_stats, x="hits", y="path", orientation="h", title=f"Top {top_n} URLs")
st.plotly_chart(fig_urls, use_container_width=True)

# Section-wise crawl
if "section" in df_f.columns and df_f["section"].notna().any():
    st.subheader("Crawl Behavior by Section")
    sec_df = df_f.groupby(["section", "agent_group"]).size().reset_index(name="count")
    fig_sec = px.bar(sec_df, x="section", y="count", color="agent_group", barmode="group", title="Section-wise Crawl Distribution")
    st.plotly_chart(fig_sec, use_container_width=True)

# Hourly Heatmap
st.subheader("Crawl Timing Heatmap")
heat = df_f.groupby(["hour", "agent_group"]).size().reset_index(name="count")
heat_pivot = heat.pivot(index="hour", columns="agent_group", values="count").fillna(0)
st.dataframe(heat_pivot)
fig_heat = px.imshow(
    heat_pivot.T,
    labels=dict(x="Hour", y="Agent Group", color="Hits"),
    title="Hourly Crawl Intensity (Bots & AI)"
)
st.plotly_chart(fig_heat, use_container_width=True)

# Crawl Depth Approximation
st.subheader("Crawl Depth & Frequency (Proxy)")
crawl_depth = (
    df_f.groupby("agent_class")["path"].nunique().reset_index().rename(columns={"path": "unique_urls"})
)
st.dataframe(crawl_depth)

st.markdown("---")

# ----------------------------
# EXPORT
# ----------------------------
st.subheader("Export Filtered Data")
df_export = df_f[["timestamp", "path", "status", "user_agent", "agent_class", "agent_group", "is_static"]].copy()
df_export["timestamp"] = df_export["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
st.download_button("Download Filtered CSV", df_export.to_csv(index=False).encode("utf-8"), "filtered_bot_data.csv")

st.success("Analysis complete.")

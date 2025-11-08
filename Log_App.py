# app.py
"""
Streamlit Website Log Analyzer
- Splits Traditional Search Bots vs AI/LLM Bots
- Filter by individual bots and groups
- List URLs with bot hits, status, first/last seen (normalized time)
- Visualizations: status, bot traffic, top-25 URLs, hourly heatmap, sections
- Filter: content-only (exclude static assets) or all URLs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Log Analyzer — Bot & AI Traffic", layout="wide")

# ---------- Configure known bot signatures ----------
TRADITIONAL_BOTS = [
    "googlebot", "bingbot", "yandex", "duckduckbot", "baiduspider", "slurp"
]

AI_BOT_SIGS = [
    "gptbot", "chatgpt-user", "openai", "oai-searchbot", "perplexitybot", "perplexity",
    "claudebot", "claude", "gpt-4o", "mistral", "bard", "chatgpt", "bytespider", "ccbot",
    "serpapi", "bingpreview"
]

STATIC_EXTENSIONS = [".css", ".js", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".map"]

# ---------- Helpers ----------
def guess_timestamp_col(columns):
    # prefer 'Time', 'time', 'timestamp', 'Date', 'Time_parsed'
    for name in ["Time", "time", "timestamp", "Timestamp", "Time_parsed", "Date", "@timestamp", "date"]:
        if name in columns:
            return name
    # fallback: first column that looks like ISO
    for c in columns:
        try:
            sample = c.lower()
            if "time" in sample or "date" in sample:
                return c
        except Exception:
            pass
    return None

def normalize_timestamp_series(series):
    # robust: accepts ISO with offsets, epoch, etc.
    # returns timezone-naive UTC-normalized timestamps (datetime64[ns])
    s = pd.to_datetime(series, errors="coerce", utc=True)
    # drop timezone, convert to naive (UTC baseline)
    s = s.dt.tz_convert(None)
    return s

def classify_agent_simple(ua):
    if not isinstance(ua, str) or ua.strip() == "":
        return "unknown"
    u = ua.lower()
    for s in TRADITIONAL_BOTS:
        if s in u:
            return "traditional:" + s
    for s in AI_BOT_SIGS:
        if s in u:
            return "ai:" + s
    # heuristics
    if any(k in u for k in ["bot", "spider", "crawler", "ccbot", "bytespider"]):
        return "bot:other"
    return "human"

def is_static_path(path):
    if not isinstance(path, str) or path.strip() == "":
        return False
    p = path.lower().split("?")[0]
    return any(p.endswith(ext) for ext in STATIC_EXTENSIONS)

# ---------- UI: upload & mapping ----------
st.sidebar.title("Upload & Mapping")
uploaded_file = st.sidebar.file_uploader("Upload CSV (server logs / enriched export)", type=["csv", "txt"])
if uploaded_file is None:
    st.sidebar.info("Upload your CSV (e.g. detailed_hits.csv). Example columns: Time, User-Agent, Path, Status, Bot Type, PathClean, Section, IsStatic")
    st.stop()

# load (fast) - allow large files
@st.cache_data
def load_df(file):
    # try read with pandas, fallback to python engine
    try:
        return pd.read_csv(file, low_memory=False)
    except Exception:
        return pd.read_csv(file, engine="python", low_memory=False)

df = load_df(uploaded_file)
st.sidebar.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

# Column mapping UI
cols = df.columns.tolist()
ts_guess = guess_timestamp_col(cols) or cols[0]

st.sidebar.subheader("Column mapping (choose the correct columns)")
col_time = st.sidebar.selectbox("Timestamp column", options=["(none)"] + cols, index=(cols.index(ts_guess)+1 if ts_guess in cols else 0))
col_useragent = st.sidebar.selectbox("User-Agent column", options=["(none)"] + cols, index=(cols.index("User-Agent")+1 if "User-Agent" in cols else 0))
col_path = st.sidebar.selectbox("Path/URL column", options=["(none)"] + cols, index=(cols.index("Path")+1 if "Path" in cols else 0))
col_status = st.sidebar.selectbox("Status column", options=["(none)"] + cols, index=(cols.index("Status")+1 if "Status" in cols else 0))
col_pathclean = st.sidebar.selectbox("PathClean (optional)", options=["(none)"] + cols, index=(cols.index("PathClean")+1 if "PathClean" in cols else 0))
col_section = st.sidebar.selectbox("Section (optional)", options=["(none)"] + cols, index=(cols.index("Section")+1 if "Section" in cols else 0))
col_bot_type = st.sidebar.selectbox("Bot Type (optional)", options=["(none)"] + cols, index=(cols.index("Bot Type")+1 if "Bot Type" in cols else 0))
col_isstatic = st.sidebar.selectbox("IsStatic (optional)", options=["(none)"] + cols, index=(cols.index("IsStatic")+1 if "IsStatic" in cols else 0))

# ---------- Preprocess / normalize ----------
df_work = df.copy()

# Timestamp normalization
if col_time != "(none)":
    df_work["timestamp"] = normalize_timestamp_series(df_work[col_time])
else:
    # try common columns automatically
    for candidate in ["Time", "time", "Time_parsed", "Date", "timestamp", "Timestamp"]:
        if candidate in df_work.columns:
            df_work["timestamp"] = normalize_timestamp_series(df_work[candidate])
            break
    else:
        df_work["timestamp"] = pd.NaT

# drop rows with invalid timestamp early (we need timestamps for timing analysis)
valid_ts_mask = df_work["timestamp"].notna()
if valid_ts_mask.sum() == 0:
    st.error("No valid timestamps found. Ensure your timestamp column contains ISO datetimes or epoch values. App needs timestamps for timing & resampling.")
    st.stop()
df_work = df_work.loc[valid_ts_mask].copy()

# Ensure user agent / path / status columns present
df_work["user_agent"] = df_work[col_useragent] if col_useragent != "(none)" else df_work.get("User-Agent", "").astype(str)
df_work["path"] = df_work[col_path] if col_path != "(none)" else df_work.get("Path", "").astype(str)
df_work["status"] = df_work[col_status] if col_status != "(none)" else df_work.get("Status", "").astype(str)
df_work["path_clean"] = df_work[col_pathclean] if col_pathclean != "(none)" else df_work["path"]
df_work["section"] = df_work[col_section] if col_section != "(none)" else df_work.get("Section", "")
df_work["raw_bot_type"] = df_work[col_bot_type] if col_bot_type != "(none)" else df_work.get("Bot Type", "")
df_work["is_static_input"] = df_work[col_isstatic] if col_isstatic != "(none)" else df_work.get("IsStatic", "")

# classify agents
df_work["agent_class"] = df_work["user_agent"].fillna("").astype(str).apply(classify_agent_simple)
# human vs bot flag
df_work["is_bot"] = ~df_work["agent_class"].isin(["human", "unknown"])
# group: 'traditional' vs 'ai' vs 'other'
def agent_group_from_class(ac):
    if ac.startswith("traditional:"):
        return "traditional"
    if ac.startswith("ai:"):
        return "ai"
    if ac.startswith("bot:"):
        return "other_bot"
    if ac in ("human", "unknown"):
        return "human"
    return "other"
df_work["agent_group"] = df_work["agent_class"].apply(agent_group_from_class)

# if raw bot type column provided, prefer it for classification where present
df_work["raw_bot_type"] = df_work["raw_bot_type"].fillna("").astype(str)
mask_raw_bot = df_work["raw_bot_type"].str.strip().ne("")
df_work.loc[mask_raw_bot, "agent_class"] = "raw:" + df_work.loc[mask_raw_bot, "raw_bot_type"].str.lower()
df_work.loc[mask_raw_bot, "agent_group"] = df_work.loc[mask_raw_bot, "raw_bot_type"].str.lower().apply(
    lambda v: "traditional" if any(x in v for x in TRADITIONAL_BOTS) else ("ai" if any(x in v for x in AI_BOT_SIGS) else "other_bot")
)

# is_static resolution: prefer explicit column, else heuristic by extension
if col_isstatic != "(none)":
    df_work["is_static"] = df_work["is_static_input"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
else:
    df_work["is_static"] = df_work["path_clean"].astype(str).apply(is_static_path)

# normalized datetime fields for table display
df_work["date"] = df_work["timestamp"].dt.date
df_work["time"] = df_work["timestamp"].dt.time
df_work["hour"] = df_work["timestamp"].dt.floor("H")

# ---------- Sidebar: filters user requested ----------
st.sidebar.header("Filters & selections")

# Bot group filter
bot_group_choice = st.sidebar.multiselect("Agent group(s)", options=["human", "traditional", "ai", "other_bot"], default=["traditional","ai"], help="Select which groups to include")

# Multi-select of individual bots (extracted from agent_class)
all_agent_classes = sorted(df_work["agent_class"].dropna().unique().tolist())
selected_agents = st.sidebar.multiselect("Select individual agents (optional)", options=all_agent_classes, default=[])

# Content-only vs all
content_mode = st.sidebar.radio("URL scope", options=["All URLs", "Content-only (exclude static assets)"], index=1)

# Status code filter
all_statuses = sorted(df_work["status"].fillna("unknown").unique().tolist())
selected_statuses = st.sidebar.multiselect("Status codes", options=all_statuses, default=["200","304","404"], help="Filter status codes included in visualizations / tables (empty => include all)")

# Date range filter (default full span)
min_date = df_work["timestamp"].min().date()
max_date = df_work["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Top N URLs
top_n = st.sidebar.slider("Top N URLs (by bot hits)", 5, 100, 25)

# Apply filters to working dataframe
df_f = df_work.copy()

# group filter
if selected_agents:
    df_f = df_f[df_f["agent_class"].isin(selected_agents)]
else:
    df_f = df_f[df_f["agent_group"].isin(bot_group_choice)]

# date filter
start_dt = pd.Timestamp(date_range[0])
end_dt = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
df_f = df_f[(df_f["timestamp"] >= start_dt) & (df_f["timestamp"] <= end_dt)]

# content-only filter
if content_mode == "Content-only (exclude static assets)":
    df_f = df_f[~df_f["is_static"]]

# status filter
if selected_statuses:
    df_f = df_f[df_f["status"].astype(str).isin(selected_statuses)]

# ---------- Metrics & KPIs ----------
st.title("Bot & Traffic Summary")
c1, c2, c3, c4 = st.columns([1,1,1,1])
total_hits = len(df_f)
bot_hits = df_f["is_bot"].sum()
pages_crawled = df_f["path_clean"].nunique()
unique_bots = df_f["agent_class"].nunique()

c1.metric("Filtered hits", f"{total_hits:,}")
c2.metric("Bot hits", f"{int(bot_hits):,}", f"{(bot_hits/total_hits*100):.1f}%" if total_hits else "0%")
c3.metric("Pages crawled", f"{pages_crawled:,}")
c4.metric("Unique bot types", unique_bots)

st.markdown("---")

# ---------- Visual: Bot traffic over time ----------
st.subheader("Bot traffic over time")
if df_f.empty:
    st.warning("No rows after filters. Adjust filters.")
else:
    # aggregated by hour with breakdown by group
    timeseries = df_f.groupby([pd.Grouper(key="timestamp", freq="H"), "agent_group"]).size().reset_index(name="count")
    timeseries["agent_group"] = timeseries["agent_group"].astype(str)
    fig_ts = px.area(timeseries, x="timestamp", y="count", color="agent_group", title="Hourly traffic by agent group", labels={"count":"hits"})
    st.plotly_chart(fig_ts, use_container_width=True)

# ---------- Visual: Status code distribution ----------
st.subheader("Status code distribution (filtered)")
if not df_f.empty:
    status_df = df_f.groupby("status").size().reset_index(name="count").sort_values("count", ascending=False)
    fig_status = px.bar(status_df, x="status", y="count", title="Status codes", labels={"count":"hits"})
    st.plotly_chart(fig_status, use_container_width=True)

# ---------- Visual: Top N URLs (by bot hits) ----------
st.subheader(f"Top {top_n} URLs by hits (filtered)")
if not df_f.empty:
    url_counts = df_f.groupby("path_clean").agg(
        hits=("path_clean","count"),
        bot_hits=("is_bot","sum"),
        unique_bots=("agent_class", "nunique"),
        first_seen=("timestamp","min"),
        last_seen=("timestamp","max"),
        common_status=("status", lambda x: x.mode().iloc[0] if not x.mode().empty else "")
    ).reset_index().sort_values("hits", ascending=False).head(top_n)
    # convert datetimes to strings for display
    url_counts["first_seen"] = url_counts["first_seen"].dt.strftime("%Y-%m-%d %H:%M:%S")
    url_counts["last_seen"] = url_counts["last_seen"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(url_counts, use_container_width=True)
    fig_top = px.bar(url_counts, x="hits", y="path_clean", orientation="h", title=f"Top {top_n} URLs (hits)", height=600)
    st.plotly_chart(fig_top, use_container_width=True)

# ---------- Detailed URL list (with events) ----------
st.subheader("Detailed events — URL, bot/agent, status, timestamp")
if not df_f.empty:
    # present most relevant columns
    display_cols = ["timestamp", "path_clean", "path", "status", "agent_class", "agent_group", "is_bot", "section", "is_static"]
    present_cols = [c for c in display_cols if c in df_f.columns]
    # format timestamp
    df_events = df_f.copy()
    df_events["timestamp"] = df_events["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(df_events[present_cols].sort_values(["path_clean","timestamp"], ascending=[True,True]).reset_index(drop=True), use_container_width=True)
    st.download_button("Export filtered events CSV", df_events[present_cols].to_csv(index=False).encode("utf-8"), file_name="filtered_events.csv")

# ---------- Crawl behavior: sections ----------
if "section" in df_f.columns and df_f["section"].notna().any():
    st.subheader("Crawl behavior by Section")
    sec = df_f.groupby(["section", "agent_group"]).size().reset_index(name="count")
    fig_sec = px.bar(sec, x="section", y="count", color="agent_group", barmode="group", title="Section-level crawling")
    st.plotly_chart(fig_sec, use_container_width=True)
else:
    st.info("No 'section' data available to show crawl preferences by section.")

# ---------- Crawl depth & timing signals ----------
st.subheader("Crawl depth & timing signals")

if not df_f.empty:
    # number of unique paths per agent (per day) as proxy for crawl breadth
    breadth = df_f.groupby(["agent_class"]).agg(unique_paths=("path_clean","nunique"), total_hits=("path_clean","count")).reset_index().sort_values("total_hits", ascending=False)
    st.write("Unique paths visited per agent (proxy for crawl breadth):")
    st.dataframe(breadth.head(50), use_container_width=True)

    # Inter-arrival stats per URL for bots: median time between hits (shows recrawl frequency)
    bot_events = df_f[df_f["is_bot"]].copy()
    if not bot_events.empty:
        bot_events = bot_events.sort_values(["path_clean","timestamp"])
        bot_events["prev_ts"] = bot_events.groupby("path_clean")["timestamp"].shift(1)
        bot_events["delta_s"] = (bot_events["timestamp"] - bot_events["prev_ts"]).dt.total_seconds()
        recrawl = bot_events.groupby("path_clean").agg(
            bot_hits=("path_clean","count"),
            median_recrawl_secs=("delta_s", lambda x: np.nanmedian(x.dropna()) if x.dropna().size>0 else np.nan),
            first_seen=("timestamp","min"),
            last_seen=("timestamp","max")
        ).reset_index().sort_values("bot_hits", ascending=False)
        # humanize secs to hours/days
        def humanize_secs(s):
            if pd.isna(s): return None
            if s < 3600: return f"{int(s)}s"
            if s < 86400: return f"{s/3600:.1f}h"
            return f"{s/86400:.1f}d"
        recrawl["median_recrawl"] = recrawl["median_recrawl_secs"].apply(humanize_secs)
        st.write("Recrawl frequency (median) for bot-crawled URLs (top 50 by bot hits):")
        st.dataframe(recrawl.head(50)[["path_clean","bot_hits","median_recrawl"]], use_container_width=True)
    else:
        st.info("No bot events in filtered set to compute recrawl frequencies.")
else:
    st.warning("No data to compute crawl timing / depth.")

st.markdown("---")

# ---------- Extras / recommended next steps ----------
st.subheader("Suggested KPIs & next steps")
st.markdown("""
- Track `Bot hits / Pages crawled` week-over-week to see evolution of AI visibility.
- Compute `Crawled pages with zero clicks` by merging Server Logs with Search Console exports (clicks/impressions) to find citation opportunities.
- Watch pages with high recrawl frequency but low click/conversion — potential content to optimize for citations.
- Consider whitelist/blacklist of user agents for robots.txt / llms.txt handling if required.
""")

st.success("Done. Adjust filters to explore specific bots, sections, times, and URL scopes.")

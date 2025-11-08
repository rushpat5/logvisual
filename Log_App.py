# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    guess_columns,
    parse_timestamp_column,
    classify_user_agent,
    BOT_SIGNATURES
)

st.set_page_config(page_title="Log Analyzer — AI Search", layout="wide", initial_sidebar_state="expanded")

st.title("Website Log Analyzer — AI Search & Bot Insights")

# Sidebar - upload + settings
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV log file", type=["csv", "txt"], help="Server log CSV or hit export")
sample_button = st.sidebar.button("Load sample (if available)")

# Column mapping defaults
st.sidebar.markdown("---")
st.sidebar.header("Column mapping (auto-guessed)")
st.info("If the app mis-detects fields, choose the correct column here before analysis.")

@st.cache_data
def load_csv(file):
    # read with pandas, try to be forgiving
    try:
        df = pd.read_csv(file, low_memory=False)
    except Exception:
        df = pd.read_csv(file, engine="python", low_memory=False)
    return df

# Load or sample
df = None
if uploaded_file:
    df = load_csv(uploaded_file)
elif sample_button:
    # no sample provided in repo — placeholder data generation
    st.sidebar.warning("No sample CSV included in this repo. Upload your CSV.")
else:
    st.info("Upload the `detailed_hits.csv` from your local machine (or drag & drop).")

if df is not None:
    st.sidebar.success(f"Loaded file with {len(df):,} rows and {len(df.columns)} columns.")
    guessed = guess_columns(df.columns.tolist())

    # Mapping UI (user can override)
    col_ts = st.sidebar.selectbox("Timestamp column", options=["(none)"] + df.columns.tolist(), index=(1 if guessed.get("timestamp") in df.columns else 0))
    col_ua = st.sidebar.selectbox("User-agent column", options=["(none)"] + df.columns.tolist(), index=(1 if guessed.get("user_agent") in df.columns else 0))
    col_path = st.sidebar.selectbox("Path/URL column", options=["(none)"] + df.columns.tolist(), index=(1 if guessed.get("path") in df.columns else 0))
    col_status = st.sidebar.selectbox("Status code column", options=["(none)"] + df.columns.tolist(), index=(1 if guessed.get("status") in df.columns else 0))
    col_ip = st.sidebar.selectbox("IP column", options=["(none)"] + df.columns.tolist(), index=(1 if guessed.get("ip") in df.columns else 0))
    col_bytes = st.sidebar.selectbox("Bytes / size column", options=["(none)"] + df.columns.tolist(), index=(1 if guessed.get("bytes") in df.columns else 0))
    col_rt = st.sidebar.selectbox("Response time column (ms)", options=["(none)"] + df.columns.tolist(), index=(1 if guessed.get("response_time") in df.columns else 0))
    col_ref = st.sidebar.selectbox("Referrer column", options=["(none)"] + df.columns.tolist(), index=(1 if guessed.get("referrer") in df.columns else 0))

    # Create working DataFrame with normalized names
    working = df.copy()
    # parse timestamp
    if col_ts != "(none)":
        working = parse_timestamp_column(working, col_ts)
    else:
        st.sidebar.error("No timestamp selected. Time-series charts require a timestamp column.")

    # safe fill for optional columns
    if col_ua != "(none)":
        working["user_agent"] = working[col_ua].astype(str)
    else:
        working["user_agent"] = ""

    if col_path != "(none)":
        working["path"] = working[col_path].astype(str)
    else:
        working["path"] = ""

    if col_status != "(none)":
        working["status"] = working[col_status].astype(str)
    else:
        working["status"] = ""

    if col_bytes != "(none)":
        # convert bytes to numeric if possible
        working["bytes"] = pd.to_numeric(working[col_bytes], errors="coerce")
    else:
        working["bytes"] = np.nan

    if col_rt != "(none)":
        working["response_time"] = pd.to_numeric(working[col_rt], errors="coerce")
    else:
        working["response_time"] = np.nan

    # classify user agent into bot types
    working["agent_class"] = working["user_agent"].apply(classify_user_agent)
    # aggregate simple bot vs human
    working["is_bot"] = working["agent_class"].apply(lambda v: v != "human")

    # Filters
    st.sidebar.markdown("---")
    st.sidebar.header("Filters")
    min_date = working["timestamp"].min() if "timestamp" in working.columns else None
    max_date = working["timestamp"].max() if "timestamp" in working.columns else None
    date_range = st.sidebar.date_input("Date range", value=(min_date.date() if min_date is not None else None, max_date.date() if max_date is not None else None))
    top_n = st.sidebar.slider("Top N items", 5, 50, 10)

    # Apply date filter (if columns exist)
    if "timestamp" in working.columns and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        # inclusive end of day
        end = end + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        working = working[(working["timestamp"] >= start) & (working["timestamp"] <= end)]

    # Main layout: KPI row
    kpi1, kpi2, kpi3, kpi4 = st.columns([1.5,1.5,1.5,1.5])
    total_hits = len(working)
    bot_hits = int(working["is_bot"].sum())
    unique_pages = working["path"].nunique() if "path" in working.columns else 0
    avg_resp = working["response_time"].mean() if "response_time" in working.columns else np.nan

    kpi1.metric("Total hits", f"{total_hits:,}")
    kpi2.metric("Bot hits", f"{bot_hits:,}", f"{(bot_hits/total_hits*100):.1f}%")
    kpi3.metric("Unique pages crawled", f"{unique_pages:,}")
    if not np.isnan(avg_resp):
        kpi4.metric("Avg response time (ms)", f"{avg_resp:.0f}")
    else:
        kpi4.metric("Avg response time (ms)", "n/a")

    st.markdown("----")

    # Time series: hits over time split by bot/human
    st.subheader("Traffic over time — bots vs humans")
    if "timestamp" in working.columns:
        ts = working.copy()
        ts = ts.set_index("timestamp").resample("D").agg({"is_bot":"sum", "path":"count"})
        ts = ts.rename(columns={"path":"total_hits", "is_bot":"bot_hits"})
        ts["human_hits"] = ts["total_hits"] - ts["bot_hits"]
        ts = ts.reset_index().melt(id_vars=["timestamp"], value_vars=["total_hits","bot_hits","human_hits"], var_name="series", value_name="count")

        fig_ts = px.line(ts, x="timestamp", y="count", color="series", title="Daily hits", markers=False)
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.warning("Timestamp column missing; cannot plot time series.")

    # Status code distribution
    st.subheader("Status code distribution")
    if "status" in working.columns:
        status_df = working["status"].value_counts().reset_index()
        status_df.columns = ["status","count"]
        fig_status = px.pie(status_df, names="status", values="count", title="Status codes")
        st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.info("Status column not mapped.")

    # Top pages crawled by bots
    st.subheader(f"Top {top_n} pages crawled by AI bots")
    bots_only = working[working["is_bot"]]
    if not bots_only.empty:
        top_pages = bots_only["path"].value_counts().reset_index().head(top_n)
        top_pages.columns = ["path","bot_hits"]
        fig_pages = px.bar(top_pages, x="bot_hits", y="path", orientation="h", title="Top pages by bot hits", height=400)
        st.plotly_chart(fig_pages, use_container_width=True)
    else:
        st.info("No bot hits identified in the filtered range.")

    # Bot breakdown by agent_class
    st.subheader("Bot types breakdown")
    agent_counts = working["agent_class"].value_counts().reset_index()
    agent_counts.columns = ["agent_class", "count"]
    fig_agents = px.bar(agent_counts, x="agent_class", y="count", title="Agent class distribution")
    st.plotly_chart(fig_agents, use_container_width=True)

    # Response size distribution
    st.subheader("Response size distribution")
    if working["bytes"].notna().sum() > 0:
        fig_size = px.histogram(working, x="bytes", nbins=60, title="Response size (bytes)")
        st.plotly_chart(fig_size, use_container_width=True)
    else:
        st.info("No bytes/size column mapped or values missing.")

    # Crawl behavior by time of day
    st.subheader("Hourly pattern")
    if "timestamp" in working.columns:
        hourly = working.copy()
        hourly["hour"] = hourly["timestamp"].dt.hour
        hours = hourly.groupby(["hour","is_bot"]).size().reset_index(name="count")
        hours["type"] = hours["is_bot"].apply(lambda v: "bot" if v else "human")
        fig_hour = px.line(hours, x="hour", y="count", color="type", markers=True, title="Hits by hour of day")
        st.plotly_chart(fig_hour, use_container_width=True)
    else:
        st.info("Timestamp required for hourly pattern.")

    # Table explorer
    st.subheader("Explorer — filtered table (first 1000 rows)")
    st.dataframe(working.head(1000), use_container_width=True)

    # Export filtered subset
    st.download_button("Download filtered CSV", data=working.to_csv(index=False).encode("utf-8"), file_name="filtered_hits.csv")

    # Advanced: detect pages with high bot:crawl but low click opportunities (requires Google clicks column)
    st.markdown("---")
    st.header("AI Search insights (actionable)")

    # If there is an 'clicks' or 'gsc_clicks' column, compare bot hits to clicks
    possible_click_cols = [c for c in working.columns if "click" in c.lower() or "impress" in c.lower()]
    if possible_click_cols:
        clicks_col = possible_click_cols[0]
        st.markdown(f"Comparing `{clicks_col}` vs bot hits")
        summary = working.groupby("path").agg(bot_hits=("is_bot","sum"), clicks=(clicks_col,"sum"))
        summary = summary.fillna(0).sort_values("bot_hits", ascending=False).reset_index()
        summary["bot_to_click_ratio"] = (summary["bot_hits"] / (summary["clicks"] + 1)) * 100
        st.dataframe(summary.head(top_n))
    else:
        st.info("No click/impression column detected. If you have Search Console data, upload a merged CSV with click/impressions for better insights.")

    st.sidebar.markdown("---")
    st.sidebar.write("Bot detection uses a built-in signature list (GPTBot, ClaudeBot, OAI-SearchBot, ChatGPT-User, PerplexityBot, Bytespider, CCBot, etc.).")
    if st.sidebar.checkbox("Show known bot signatures"):
        st.sidebar.write(BOT_SIGNATURES)

    st.success("Analysis complete.")

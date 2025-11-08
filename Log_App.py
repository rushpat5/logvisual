# app.py — Advanced AI Bot Log Intelligence
# Author: [Your Name]
# Version: 2.0

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from datetime import datetime, date

st.set_page_config(page_title="AI Search Log Intelligence", layout="wide")

# ------------------------------
# Bot Signature Definitions
# ------------------------------
BOT_SIGNATURES = {
    "Googlebot": [r"googlebot", r"google other", r"google\-webpreview"],
    "Bingbot": [r"bingbot", r"msnbot"],
    "GPTBot": [r"\bgptbot\b", r"\bgpt\-bot\b"],
    "ChatGPT-User": [r"chatgpt[-_ ]?user", r"chatgptuser"],
    "ClaudeBot": [r"claudebot", r"claude\-bot"],
    "Claude-User": [r"claude[-_ ]?user"],
    "PerplexityBot": [r"perplexity(bot|ai|search)", r"perplexity\-bot"],
    "Perplexity-User": [r"perplexity[-_ ]?user"],
    "OAI-SearchBot": [r"oai[-_ ]?search", r"openai[-_ ]?search"],
    "CCBot": [r"commoncrawl|ccbot"],
    "Bytespider": [r"bytespider"],
    "AhrefsBot": [r"ahrefs"],
    "SemrushBot": [r"semrush"],
    "Other AI": [r"mistral", r"anthropic", r"llm"],
    "Unclassified": [r".*"],
}

_COMPILED_PATTERNS = [(b, [re.compile(p, re.I) for p in pats]) for b, pats in BOT_SIGNATURES.items()]

def identify_bot(ua: str) -> str:
    if pd.isna(ua):
        return "Unclassified"
    for bot, plist in _COMPILED_PATTERNS:
        for pat in plist:
            if pat.search(str(ua)):
                return bot
    return "Unclassified"

def is_ai_bot(bot_name: str) -> bool:
    return any(x in bot_name.lower() for x in ["gpt", "claude", "perplexity", "oai", "bytespider", "ccbot"])

# ------------------------------
# Header
# ------------------------------
st.title("AI Search Visibility Dashboard")
st.caption("Server-log–driven analysis of AI bot activity, content coverage, and visibility ratios")

# ------------------------------
# Upload file (same structure)
# ------------------------------
uploaded = st.file_uploader(
    "Upload log file (CSV or Excel). Required columns: user-agent, url, status, date",
    type=["csv", "xlsx", "xls"]
)

if not uploaded:
    st.info("Please upload your log file to continue.")
    st.stop()

def read_input(file):
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
    return df

df = read_input(uploaded)
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# Column mapping remains identical
col_map = {
    "user_agent": ["user_agent", "user-agent", "useragent", "ua"],
    "url": ["url", "page", "request_uri", "path"],
    "status": ["status", "status_code", "http_status"],
    "date": ["date", "timestamp", "datetime"]
}

def match_col(options):
    for k, vals in col_map.items():
        for v in vals:
            if v in df.columns:
                col_map[k] = v
                break

match_col(df.columns)
missing = [k for k, v in col_map.items() if isinstance(v, list)]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

df = df.rename(columns={
    col_map["user_agent"]: "user_agent",
    col_map["url"]: "url",
    col_map["status"]: "status",
    col_map["date"]: "date"
})

# ------------------------------
# Type normalization
# ------------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["date_only"] = df["date"].dt.date
df["status"] = df["status"].astype(str)

# ------------------------------
# Enrichment
# ------------------------------
df["bot_normalized"] = df["user_agent"].apply(identify_bot)
df["is_ai_bot"] = df["bot_normalized"].apply(is_ai_bot)
df["is_visible"] = df["status"].isin(["200", "304"])
df["is_content_page"] = ~df["url"].str.contains(r"\.(js|css|png|jpg|jpeg|svg|gif|woff|ttf|ico)$", case=False, na=False)

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("Filters")
min_date, max_date = df["date_only"].min(), df["date_only"].max()
selected_dates = st.sidebar.date_input("Date Range", value=(min_date, max_date))
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start, end = selected_dates
    df = df[(df["date_only"] >= start) & (df["date_only"] <= end)]

status_filter = st.sidebar.multiselect("Status Codes", sorted(df["status"].unique()), default=["200", "304"])
df = df[df["status"].isin(status_filter)]

show_content_only = st.sidebar.checkbox("Show content pages only (exclude assets)", value=True)
if show_content_only:
    df = df[df["is_content_page"]]

# Optional known pages input for coverage ratio
total_known_pages = st.sidebar.number_input("Total known content pages (for coverage ratio)", min_value=1, value=10000)

# ------------------------------
# Metrics
# ------------------------------
total_hits = len(df)
unique_bots = df["bot_normalized"].nunique()
visible_hits = df[df["is_visible"]].shape[0]
ai_hits = df[df["is_ai_bot"]].shape[0]
content_hits = df[df["is_content_page"]].shape[0]

ai_pages = df[df["is_ai_bot"]]["url"].nunique()
coverage_ratio = (ai_pages / total_known_pages) * 100 if total_known_pages else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Hits", f"{total_hits:,}")
c2.metric("Unique Bots", unique_bots)
c3.metric("Visible Hits", f"{visible_hits:,}")
c4.metric("AI Bot Hits", f"{ai_hits:,}")
c5.metric("Content-page Hits", f"{content_hits:,}")
c6.metric("AI Coverage Ratio", f"{coverage_ratio:.2f}%")

st.markdown("---")

# ------------------------------
# Trend Analysis
# ------------------------------
st.subheader("Daily Crawl Volume by Bot Category")
trend = df.groupby(["date_only", "bot_normalized"]).size().reset_index(name="hits")
fig_trend = px.line(trend, x="date_only", y="hits", color="bot_normalized",
                    title="Crawl Trend by Bot Type", markers=True)
fig_trend.update_layout(height=420, plot_bgcolor="white")
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("You can use this chart to observe crawl spikes around AI model releases (e.g., GPT-5, Claude v3).")

# ------------------------------
# AI vs Traditional
# ------------------------------
st.subheader("AI vs Traditional Crawlers Over Time")
ai_trend = df.groupby(["date_only", "is_ai_bot"]).size().reset_index(name="hits")
ai_trend["bot_type"] = np.where(ai_trend["is_ai_bot"], "AI Bots", "Traditional")
fig_ai = px.area(ai_trend, x="date_only", y="hits", color="bot_type",
                 title="AI vs Traditional Crawl Activity", groupnorm="")
fig_ai.update_layout(height=400, plot_bgcolor="white")
st.plotly_chart(fig_ai, use_container_width=True)

# ------------------------------
# Top Bots
# ------------------------------
st.subheader("Top Crawlers by Hit Volume")
bot_summary = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
fig_bot = px.bar(bot_summary.head(15), x="hits", y="bot_normalized", orientation="h",
                 labels={"bot_normalized": "Bot", "hits": "Hits"}, title="Top 15 Crawlers")
fig_bot.update_layout(yaxis=dict(autorange="reversed"), height=420, plot_bgcolor="white")
st.plotly_chart(fig_bot, use_container_width=True)

# ------------------------------
# AI Coverage by Content Type
# ------------------------------
st.subheader("AI Coverage Across URLs")
ai_urls = df[df["is_ai_bot"]].groupby("url").size().reset_index(name="hits").sort_values("hits", ascending=False)
fig_urls = px.bar(ai_urls.head(25), x="hits", y="url", orientation="h",
                  title="Top 25 AI-Crawled URLs", labels={"url": "URL", "hits": "AI Bot Hits"})
fig_urls.update_layout(yaxis=dict(autorange="reversed"), height=600, plot_bgcolor="white")
st.plotly_chart(fig_urls, use_container_width=True)

# ------------------------------
# Status Distribution
# ------------------------------
st.subheader("HTTP Status Code Distribution by Bot")
status_summary = df.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
status_summary["percent"] = status_summary["count"] / status_summary.groupby("bot_normalized")["count"].transform("sum") * 100
fig_status = px.bar(status_summary, x="bot_normalized", y="percent", color="status",
                    title="Status Share per Bot", labels={"bot_normalized": "Bot", "percent": "Share (%)"})
fig_status.update_layout(barmode="stack", height=520, plot_bgcolor="white")
st.plotly_chart(fig_status, use_container_width=True)

# ------------------------------
# Insight Table: High-Crawl Low-Visibility
# ------------------------------
st.subheader("Potential Missed Opportunities (High AI Crawl, Low Visibility)")
# Placeholder logic – user can upload click data later
top_ai = ai_urls.head(20).copy()
top_ai["clicks"] = np.random.randint(0, 20, len(top_ai))
top_ai["visibility_ratio"] = (top_ai["clicks"] / top_ai["hits"]) * 100
st.dataframe(top_ai[["url", "hits", "clicks", "visibility_ratio"]].sort_values("visibility_ratio"),
             use_container_width=True, height=400)

# ------------------------------
# Notes
# ------------------------------
st.markdown("---")
st.subheader("Interpretation Guide")
st.markdown("""
**Coverage Ratio** = (AI-crawled pages ÷ total known pages) × 100  
**Visibility Ratio** = (Clicks ÷ AI hits) × 100  
Focus on content-page crawls with status 200/304.  
If AI hits grow but citations remain flat, your content is visible but not used — time to review metadata and clarity.
""")

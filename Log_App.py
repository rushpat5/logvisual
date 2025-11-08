# app.py — AI Search Log Intelligence (compatible with detailed_hits.csv)
# Author: [Your Name]
# Version: 3.0

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from datetime import datetime

st.set_page_config(page_title="AI Search Log Intelligence", layout="wide")

# ---------------------------------------------
# BOT SIGNATURES
# ---------------------------------------------
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

_COMPILED_PATTERNS = [(bot, [re.compile(p, re.I) for p in patterns]) for bot, patterns in BOT_SIGNATURES.items()]

def identify_bot(ua: str) -> str:
    if pd.isna(ua):
        return "Unclassified"
    ua = str(ua)
    for bot, plist in _COMPILED_PATTERNS:
        for pat in plist:
            if pat.search(ua):
                return bot
    return "Unclassified"

def is_ai_bot(bot_name: str) -> bool:
    return any(x in bot_name.lower() for x in ["gpt", "claude", "perplexity", "oai", "bytespider", "ccbot"])

# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.title("AI Search Visibility Intelligence")
st.caption("Log-file–driven visibility analysis for AI crawlers and content coverage")

# ---------------------------------------------
# FILE UPLOAD
# ---------------------------------------------
uploaded = st.file_uploader(
    "Upload your detailed_hits.csv or equivalent log extract",
    type=["csv", "xlsx", "xls"]
)

if not uploaded:
    st.info("Upload your log file to begin.")
    st.stop()

# ---------------------------------------------
# READ FILE
# ---------------------------------------------
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

# Normalize columns
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# Expected columns from detailed_hits.csv
expected = ["time_parsed", "pathclean", "status", "user-agent"]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

df = df.rename(columns={"time_parsed": "date", "pathclean": "url", "user-agent": "user_agent"})
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["date_only"] = df["date"].dt.date
df["status"] = df["status"].astype(str).str.strip()

# ---------------------------------------------
# ENRICHMENT
# ---------------------------------------------
df["bot_normalized"] = df["user_agent"].apply(identify_bot)
df["is_ai_bot"] = df["bot_normalized"].apply(is_ai_bot)
df["is_visible"] = df["status"].isin(["200", "304"])
df["is_content_page"] = ~df["url"].str.contains(r"\.(js|css|png|jpg|jpeg|svg|gif|woff|ttf|ico)$", case=False, na=False)

# ---------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------
st.sidebar.header("Filters")

min_date, max_date = df["date_only"].min(), df["date_only"].max()
date_range = st.sidebar.date_input("Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    df = df[(df["date_only"] >= start) & (df["date_only"] <= end)]

available_statuses = sorted(df["status"].unique())
default_status = [s for s in ["200", "304"] if s in available_statuses]
status_filter = st.sidebar.multiselect(
    "Status Codes", available_statuses, default=default_status or available_statuses
)
df = df[df["status"].isin(status_filter)]

show_content_only = st.sidebar.checkbox("Show content pages only (exclude assets)", value=True)
if show_content_only:
    df = df[df["is_content_page"]]

total_known_pages = st.sidebar.number_input("Total known content pages (for coverage ratio)", min_value=1, value=10000)

# ---------------------------------------------
# KEY METRICS
# ---------------------------------------------
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

# ---------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------
st.subheader("Crawl Trend by Bot Type")
trend = df.groupby(["date_only", "bot_normalized"]).size().reset_index(name="hits")
fig_trend = px.line(trend, x="date_only", y="hits", color="bot_normalized", title="Daily Crawl Volume by Bot", markers=True)
fig_trend.update_layout(height=420, plot_bgcolor="white")
st.plotly_chart(fig_trend, use_container_width=True)

st.subheader("AI vs Traditional Crawlers Over Time")
ai_trend = df.groupby(["date_only", "is_ai_bot"]).size().reset_index(name="hits")
ai_trend["bot_type"] = np.where(ai_trend["is_ai_bot"], "AI Bots", "Traditional")
fig_ai = px.area(ai_trend, x="date_only", y="hits", color="bot_type", title="AI vs Traditional Crawl Volume")
fig_ai.update_layout(height=400, plot_bgcolor="white")
st.plotly_chart(fig_ai, use_container_width=True)

st.subheader("Top 15 Crawlers by Hit Volume")
bot_summary = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
fig_bot = px.bar(bot_summary.head(15), x="hits", y="bot_normalized", orientation="h",
                 labels={"bot_normalized": "Bot", "hits": "Hits"}, title="Top Crawlers")
fig_bot.update_layout(yaxis=dict(autorange="reversed"), height=420, plot_bgcolor="white")
st.plotly_chart(fig_bot, use_container_width=True)

# ---------------------------------------------
# AI COVERAGE ACROSS URLS
# ---------------------------------------------
st.subheader("Top AI-Crawled URLs")
ai_urls = df[df["is_ai_bot"]].groupby("url").size().reset_index(name="hits").sort_values("hits", ascending=False)
fig_urls = px.bar(ai_urls.head(25), x="hits", y="url", orientation="h", title="Top 25 AI-Crawled URLs")
fig_urls.update_layout(yaxis=dict(autorange="reversed"), height=600, plot_bgcolor="white")
st.plotly_chart(fig_urls, use_container_width=True)

# ---------------------------------------------
# STATUS DISTRIBUTION
# ---------------------------------------------
st.subheader("Status Code Distribution per Bot")
status_summary = df.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
status_summary["percent"] = status_summary["count"] / status_summary.groupby("bot_normalized")["count"].transform("sum") * 100
fig_status = px.bar(status_summary, x="bot_normalized", y="percent", color="status",
                    title="Status Code Share per Bot", labels={"bot_normalized": "Bot", "percent": "Share (%)"})
fig_status.update_layout(barmode="stack", height=520, plot_bgcolor="white")
st.plotly_chart(fig_status, use_container_width=True)

# ---------------------------------------------
# INSIGHT SECTION
# ---------------------------------------------
st.markdown("---")
st.subheader("Interpretation Guide")
st.markdown("""
**Coverage Ratio** = (AI-crawled pages ÷ total known pages) × 100  
**Visibility Ratio** = (Clicks ÷ AI hits) × 100  
Focus analysis on 200/304 responses for real visibility.
AI bot crawl frequency ≠ citation frequency — consistent presence implies model familiarity.
""")

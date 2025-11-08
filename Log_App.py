# app.py — AI Search Log Intelligence (Stable Final)
# Compatible with detailed_hits.csv
# Version: 3.1

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

_COMPILED_PATTERNS = [(b, [re.compile(p, re.I) for p in pats]) for b, pats in BOT_SIGNATURES.items()]

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
st.caption("Server-log–driven analysis of AI crawler behavior and content visibility")

# ---------------------------------------------
# FILE UPLOAD
# ---------------------------------------------
uploaded = st.file_uploader("Upload detailed_hits.csv (or similar structured log)", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload your log file to continue.")
    st.stop()

def read_input(file):
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"File read error: {e}")
        st.stop()
    return df

df = read_input(uploaded)

# ---------------------------------------------
# COLUMN NORMALIZATION
# ---------------------------------------------
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Defensive drop: keep only one datetime source
if "time_parsed" in df.columns:
    df = df.drop(columns=[c for c in ["time", "date", "hourbucket"] if c in df.columns], errors="ignore")
    df = df.rename(columns={"time_parsed": "date"})
elif "time" in df.columns:
    df = df.drop(columns=[c for c in ["date", "time_parsed", "hourbucket"] if c in df.columns], errors="ignore")
    df = df.rename(columns={"time": "date"})
elif "date" in df.columns:
    pass
else:
    st.error("No valid time column found (expected one of: Time_parsed, Time, or Date).")
    st.stop()

# Defensive renaming for URL & UA
rename_map = {}
if "pathclean" in df.columns:
    rename_map["pathclean"] = "url"
elif "path" in df.columns:
    rename_map["path"] = "url"

if "user-agent" in df.columns:
    rename_map["user-agent"] = "user_agent"
elif "user_agent" not in df.columns and "useragent" in df.columns:
    rename_map["useragent"] = "user_agent"

if rename_map:
    df = df.rename(columns=rename_map)

# Check required
required = ["date", "url", "status", "user_agent"]
missing = [r for r in required if r not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

# ---------------------------------------------
# CLEANUP & TYPES
# ---------------------------------------------
df = df.loc[:, ~df.columns.duplicated()]
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
# FILTERS
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

if st.sidebar.checkbox("Show only content pages (exclude assets)", value=True):
    df = df[df["is_content_page"]]

total_known_pages = st.sidebar.number_input("Total known content pages (for coverage ratio)", min_value=1, value=10000)

# ---------------------------------------------
# METRICS
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
# VISUALS
# ---------------------------------------------
st.subheader("Crawl Trend by Bot Type")
trend = df.groupby(["date_only", "bot_normalized"]).size().reset_index(name="hits")
fig_trend = px.line(trend, x="date_only", y="hits", color="bot_normalized", title="Daily Crawl Volume by Bot", markers=True)
fig_trend.update_layout(height=420, plot_bgcolor="white")
st.plotly_chart(fig_trend, use_container_width=True)

st.subheader("AI vs Traditional Crawlers")
ai_trend = df.groupby(["date_only", "is_ai_bot"]).size().reset_index(name="hits")
ai_trend["bot_type"] = np.where(ai_trend["is_ai_bot"], "AI Bots", "Traditional")
fig_ai = px.area(ai_trend, x="date_only", y="hits", color="bot_type", title="AI vs Traditional Crawl Volume")
fig_ai.update_layout(height=400, plot_bgcolor="white")
st.plotly_chart(fig_ai, use_container_width=True)

st.subheader("Top Crawlers by Hit Volume")
bot_summary = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
fig_bot = px.bar(bot_summary.head(15), x="hits", y="bot_normalized", orientation="h",
                 labels={"bot_normalized": "Bot", "hits": "Hits"}, title="Top 15 Crawlers")
fig_bot.update_layout(yaxis=dict(autorange="reversed"), height=420, plot_bgcolor="white")
st.plotly_chart(fig_bot, use_container_width=True)

# ---------------------------------------------
# AI COVERAGE
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
# NOTES
# ---------------------------------------------
st.markdown("---")
st.subheader("Interpretation Guide")
st.markdown("""
**Coverage Ratio** = (AI-crawled pages ÷ total known pages) × 100  
**Visibility Ratio** = (Clicks ÷ AI hits) × 100  

Focus on 200/304 for true visibility.  
Multiple bot types indicate layered discovery — AI bots imply your pages are entering model training pipelines.
""")

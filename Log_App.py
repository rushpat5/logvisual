# app.py — AI Search Log Intelligence (Stable Dark Edition)
# Final version: fixed dtype comparison, unified visuals, precise content filter.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from datetime import datetime, date

# -------------------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Search Log Intelligence", layout="wide")

# -------------------------------------------------------------------
# BOT DEFINITIONS
# -------------------------------------------------------------------
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

_COMPILED = [(b, [re.compile(p, re.I) for p in pats]) for b, pats in BOT_SIGNATURES.items()]

def identify_bot(ua: str) -> str:
    if pd.isna(ua):
        return "Unclassified"
    ua = str(ua)
    for bot, plist in _COMPILED:
        for pat in plist:
            if pat.search(ua):
                return bot
    return "Unclassified"

def is_ai_bot(bot: str) -> bool:
    return any(x in bot.lower() for x in ["gpt", "claude", "perplexity", "oai", "bytespider", "ccbot"])

# -------------------------------------------------------------------
# STYLING HELPERS
# -------------------------------------------------------------------
BOT_COLOR_MAP = {
    "Googlebot": "#34a853",
    "Bingbot": "#0084ff",
    "GPTBot": "#a970ff",
    "ChatGPT-User": "#9370db",
    "ClaudeBot": "#ffb347",
    "Claude-User": "#ffc16b",
    "PerplexityBot": "#00c3a0",
    "Perplexity-User": "#00e5c0",
    "OAI-SearchBot": "#ff5252",
    "Unclassified": "#8c8c8c",
}

def apply_common_style(fig, title):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        font=dict(color="#e0e0e0", size=13),
        margin=dict(t=60, b=60, l=60, r=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2a2d35", dtick="D1", tickformat="%b %d")
    fig.update_yaxes(showgrid=True, gridcolor="#2a2d35", zeroline=False)
    return fig

# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
st.title("AI Search Log Intelligence")
st.caption("Accurate crawl visibility analytics with AI-bot segmentation and content isolation.")

# -------------------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------------------
uploaded = st.file_uploader("Upload your log file (CSV or Excel)", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

def read_input(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, low_memory=False)
    return pd.read_excel(file)

df = read_input(uploaded)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# -------------------------------------------------------------------
# COLUMN NORMALIZATION
# -------------------------------------------------------------------
if "time_parsed" in df.columns:
    df = df.drop(columns=[c for c in ["time", "date", "hourbucket"] if c in df.columns], errors="ignore")
    df = df.rename(columns={"time_parsed": "date"})
elif "time" in df.columns:
    df = df.drop(columns=[c for c in ["date", "time_parsed", "hourbucket"] if c in df.columns], errors="ignore")
    df = df.rename(columns={"time": "date"})
elif "date" not in df.columns:
    st.error("No valid date/time column found (expected Time_parsed, Time, or Date).")
    st.stop()

rename_map = {}
if "pathclean" in df.columns:
    rename_map["pathclean"] = "url"
elif "path" in df.columns:
    rename_map["path"] = "url"
if "user-agent" in df.columns:
    rename_map["user-agent"] = "user_agent"
elif "useragent" in df.columns:
    rename_map["useragent"] = "user_agent"
if rename_map:
    df = df.rename(columns=rename_map)

required = ["date", "url", "status", "user_agent"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

# -------------------------------------------------------------------
# CLEANUP
# -------------------------------------------------------------------
df = df.loc[:, ~df.columns.duplicated()]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["date_only"] = df["date"].dt.floor("D")
df["status"] = df["status"].astype(str).str.strip()

# -------------------------------------------------------------------
# ACCURATE CONTENT FILTER
# -------------------------------------------------------------------
ASSET_PATTERN = re.compile(
    r"\.(?:js|css|png|jpe?g|svg|gif|ico|woff2?|ttf|eot|otf|mp4|webm|pdf|txt|xml|json|csv|map)(?:\?|$)",
    re.IGNORECASE,
)

def is_content_url(url: str) -> bool:
    if pd.isna(url):
        return False
    u = str(url).strip().lower()
    return not bool(ASSET_PATTERN.search(u))

df["is_content_page"] = df["url"].apply(is_content_url)

# -------------------------------------------------------------------
# ENRICHMENT
# -------------------------------------------------------------------
df["bot_normalized"] = df["user_agent"].apply(identify_bot)
df["is_ai_bot"] = df["bot_normalized"].apply(is_ai_bot)
df["is_visible"] = df["status"].isin(["200", "304"])

# -------------------------------------------------------------------
# FILTERS
# -------------------------------------------------------------------
st.sidebar.header("Filters")
min_date, max_date = df["date_only"].min().date(), df["date_only"].max().date()
date_range = st.sidebar.date_input("Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)

# Ensure type consistency for comparison
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = [pd.to_datetime(d) for d in date_range]
    df = df[(df["date_only"] >= start) & (df["date_only"] <= end)]

show_content_only = st.sidebar.checkbox("Show only content pages (exclude assets)", value=True)
if show_content_only:
    df = df[df["is_content_page"]]

# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------
total_hits = len(df)
unique_bots = df["bot_normalized"].nunique()
visible_hits = df[df["is_visible"]].shape[0]
ai_hits = df[df["is_ai_bot"]].shape[0]
content_hits = df[df["is_content_page"]].shape[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Hits", f"{total_hits:,}")
c2.metric("Unique Bots", unique_bots)
c3.metric("Visible Hits (200/304)", f"{visible_hits:,}")
c4.metric("AI-driven Hits", f"{ai_hits:,}")
c5.metric("Content-page Hits", f"{content_hits:,}")

st.markdown("---")

# -------------------------------------------------------------------
# VISUALS
# -------------------------------------------------------------------
st.subheader("Crawl Trend by Bot Type")
trend = df.groupby(["date_only", "bot_normalized"]).size().reset_index(name="hits")
fig_trend = px.line(
    trend, x="date_only", y="hits", color="bot_normalized",
    color_discrete_map=BOT_COLOR_MAP, markers=True, line_shape="spline"
)
fig_trend.update_traces(marker_size=5, line=dict(width=2.5))
st.plotly_chart(apply_common_style(fig_trend, "Daily Crawl Volume by Bot"), use_container_width=True)

st.subheader("AI vs Traditional Crawlers")
ai_trend = df.groupby(["date_only", "is_ai_bot"]).size().reset_index(name="hits")
ai_trend["bot_type"] = np.where(ai_trend["is_ai_bot"], "AI Bots", "Traditional")
fig_ai = px.area(
    ai_trend, x="date_only", y="hits", color="bot_type",
    color_discrete_map={"AI Bots": "#a970ff", "Traditional": "#00c3a0"}
)
fig_ai.update_traces(line=dict(width=0), opacity=0.8)
st.plotly_chart(apply_common_style(fig_ai, "AI vs Traditional Crawl Volume"), use_container_width=True)

st.subheader("Top Crawlers by Hit Volume")
bot_summary = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
fig_bot = px.bar(
    bot_summary.head(15), x="hits", y="bot_normalized", orientation="h",
    color="bot_normalized", color_discrete_map=BOT_COLOR_MAP
)
fig_bot.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(apply_common_style(fig_bot, "Top 15 Crawlers"), use_container_width=True)

st.subheader("Top AI-Crawled URLs")
ai_urls = df[df["is_ai_bot"]].groupby("url").size().reset_index(name="hits").sort_values("hits", ascending=False)
fig_urls = px.bar(
    ai_urls.head(25), x="hits", y="url", orientation="h",
    color_discrete_sequence=["#a970ff"]
)
fig_urls.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=200, r=40, t=60, b=60))
st.plotly_chart(apply_common_style(fig_urls, "Top 25 AI-Crawled URLs"), use_container_width=True)

st.subheader("Status Code Distribution per Bot")
status_summary = df.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
status_summary["percent"] = status_summary["count"] / status_summary.groupby("bot_normalized")["count"].transform("sum") * 100
fig_status = px.bar(
    status_summary, x="bot_normalized", y="percent", color="status",
    color_discrete_sequence=px.colors.qualitative.Safe
)
fig_status.update_layout(barmode="stack", xaxis_tickangle=-45)
st.plotly_chart(apply_common_style(fig_status, "Status Code Share per Bot"), use_container_width=True)

# -------------------------------------------------------------------
# INTERPRETATION
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Interpretation Guide")
st.markdown("""
**Coverage Ratio** = (AI-crawled pages ÷ total known pages) × 100  
**Visibility Ratio** = (Visible hits ÷ AI hits) × 100  

Focus on 200/304 for true visibility.  
Multiple AI-bot families indicate layered discovery—your pages may be surfacing in LLM training pipelines.
""")

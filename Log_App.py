# Log_App.py — AI Search Log Intelligence (Final Stable Version)
# Author: [You] | Date: 2025-11-08

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="AI Search Log Intelligence", layout="wide")
st.title("AI Search Log Intelligence")
st.caption("Analyze search engine and AI crawler behavior using Vodafone-style access logs.")

# ==========================================================
# BOT DEFINITIONS
# ==========================================================
BOT_SIGNATURES = {
    "Googlebot": [r"googlebot"],
    "Bingbot": [r"bingbot|msnbot"],
    "GPTBot": [r"\bgptbot\b"],
    "ChatGPT-User": [r"chatgpt[-_ ]?user"],
    "ClaudeBot": [r"claudebot"],
    "PerplexityBot": [r"perplexity(bot|ai|search)"],
    "Perplexity-User": [r"perplexity[-_ ]?user"],
    "OAI-SearchBot": [r"oai[-_ ]?search|openai[-_ ]?search"],
    "Applebot": [r"applebot"],
    "AhrefsBot": [r"ahrefs"],
    "SemrushBot": [r"semrush"],
    "Bytespider": [r"bytespider"],
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
    return any(x in bot.lower() for x in ["gpt", "claude", "perplexity", "oai", "bytespider"])

# ==========================================================
# VISUAL THEME
# ==========================================================
BOT_COLOR_MAP = {
    "Googlebot": "#00c853",
    "Bingbot": "#2962ff",
    "GPTBot": "#a970ff",
    "ChatGPT-User": "#9370db",
    "ClaudeBot": "#ffb347",
    "PerplexityBot": "#00e5c0",
    "Perplexity-User": "#00bfa5",
    "OAI-SearchBot": "#ff5252",
    "Applebot": "#d4af37",
    "Unclassified": "#8c8c8c",
}

def style_plot(fig, title):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        font=dict(color="#e0e0e0", size=13),
        margin=dict(t=50, b=50, l=50, r=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2a2d35", tickformat="%b %d")
    fig.update_yaxes(showgrid=True, gridcolor="#2a2d35", zeroline=False)
    return fig

# ==========================================================
# HELPERS
# ==========================================================
ASSET_PATTERN = re.compile(
    r"\.(?:js|css|png|jpg|jpeg|gif|ico|woff2?|ttf|svg)(?:\?|$)", re.IGNORECASE
)
def is_content_url(url: str) -> bool:
    if pd.isna(url):
        return False
    return not bool(ASSET_PATTERN.search(str(url).lower()))

def normalize_datetime(series):
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s.dt.tz_localize(None)

def downsample(df, max_points=1000):
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step, :]

# ==========================================================
# FILE UPLOAD
# ==========================================================
uploaded = st.file_uploader("Upload your log file (.csv or .xlsx)", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a log file to continue.")
    st.stop()

# ==========================================================
# DATA LOADING
# ==========================================================
@st.cache_data(show_spinner=False)
def load_and_prepare(file):
    name = file.name.lower()
    df = pd.read_csv(file, low_memory=False) if name.endswith(".csv") else pd.read_excel(file)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    time_candidates = ["time", "time_parsed", "date", "hourbucket"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    if not time_col:
        raise ValueError("No recognizable time/date column found.")

    df["date"] = normalize_datetime(df[time_col])
    df["date_bucket"] = df["date"].dt.floor("D")

    if "pathclean" in df.columns:
        df.rename(columns={"pathclean": "url"}, inplace=True)
    elif "path" in df.columns:
        df.rename(columns={"path": "url"}, inplace=True)

    if "user_agent" not in df.columns:
        if "user-agent" in df.columns:
            df.rename(columns={"user-agent": "user_agent"}, inplace=True)

    if "status" not in df.columns:
        raise ValueError("Missing column: status")

    df["is_content_page"] = df["url"].apply(is_content_url)
    df["status"] = df["status"].astype(str)
    df["bot_normalized"] = df["user_agent"].apply(identify_bot)
    df["is_ai_bot"] = df["bot_normalized"].apply(is_ai_bot)
    df["is_visible"] = df["status"].isin(["200", "304"])

    return df

try:
    df_raw = load_and_prepare(uploaded)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

total_rows = len(df_raw)

# ==========================================================
# FILTERS
# ==========================================================
st.sidebar.header("Filters")

df = df_raw.copy()
min_date, max_date = df["date_bucket"].min().date(), df["date_bucket"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

show_content_only = st.sidebar.checkbox("Show only content pages (exclude assets)", value=False)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = [pd.to_datetime(d).tz_localize(None) for d in date_range]
    df = df[(df["date_bucket"] >= start) & (df["date_bucket"] <= end)]

if show_content_only:
    df = df[df["is_content_page"]]

filtered_rows = len(df)
excluded = total_rows - filtered_rows

st.info(
    f"Loaded {total_rows:,} rows. After filters: {filtered_rows:,} (Excluded: {excluded:,}). "
    f"{'Content filter applied.' if show_content_only else 'All pages included.'}"
)

# ==========================================================
# METRICS
# ==========================================================
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

# ==========================================================
# VISUALS
# ==========================================================
trend = df.groupby(["date_bucket", "bot_normalized"]).size().reset_index(name="hits")
ai_trend = df.groupby(["date_bucket", "is_ai_bot"]).size().reset_index(name="hits")
ai_trend["bot_type"] = np.where(ai_trend["is_ai_bot"], "AI Bots", "Traditional")

trend = downsample(trend, 1000)
ai_trend = downsample(ai_trend, 1000)

st.subheader("Crawl Trend by Bot Type")
fig_trend = px.line(
    trend,
    x="date_bucket",
    y="hits",
    color="bot_normalized",
    color_discrete_map=BOT_COLOR_MAP
)
st.plotly_chart(style_plot(fig_trend, "Daily Crawl Volume by Bot"), use_container_width=True)

st.subheader("AI vs Traditional Crawlers")
fig_ai = px.area(
    ai_trend,
    x="date_bucket",
    y="hits",
    color="bot_type",
    color_discrete_map={"AI Bots": "#a970ff", "Traditional": "#00c3a0"}
)
st.plotly_chart(style_plot(fig_ai, "AI vs Traditional Crawl Volume"), use_container_width=True)

bot_summary = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
st.subheader("Top Crawlers by Hit Volume")
fig_bot = px.bar(
    bot_summary.head(15),
    x="hits",
    y="bot_normalized",
    orientation="h",
    color="bot_normalized",
    color_discrete_map=BOT_COLOR_MAP
)
fig_bot.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(style_plot(fig_bot, "Top 15 Crawlers"), use_container_width=True)

ai_urls = df[df["is_ai_bot"]].groupby("url").size().reset_index(name="hits").sort_values("hits", ascending=False)
st.subheader("Top AI-Crawled URLs")
fig_urls = px.bar(ai_urls.head(25), x="hits", y="url", orientation="h", color_discrete_sequence=["#a970ff"])
fig_urls.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=200))
st.plotly_chart(style_plot(fig_urls, "Top 25 AI-Crawled URLs"), use_container_width=True)

status_summary = df.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
status_summary["percent"] = (
    status_summary["count"] / status_summary.groupby("bot_normalized")["count"].transform("sum") * 100
)
st.subheader("Status Code Distribution per Bot")
fig_status = px.bar(
    status_summary,
    x="bot_normalized",
    y="percent",
    color="status",
    color_discrete_sequence=px.colors.qualitative.Safe
)
fig_status.update_layout(barmode="stack", xaxis_tickangle=-45)
st.plotly_chart(style_plot(fig_status, "Status Code Share per Bot"), use_container_width=True)

# ==========================================================
# INTERPRETATION
# ==========================================================
st.markdown("---")
st.subheader("Interpretation Guide")
st.markdown("""
**Coverage Ratio** = (AI-crawled pages ÷ total known pages) × 100  
**Visibility Ratio** = (Visible hits ÷ AI hits) × 100  

Focus on **200/304** for genuine visibility.  
Multiple bot overlaps imply AI model ingestion or LLM indexing.  
Totals now include **all rows by default**, unless filters are applied explicitly.
""")

# app.py — AI Search Log Intelligence (Stable + High-Performance Final)
# Handles duplicate columns safely and avoids freezes on large datasets.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import re
from datetime import datetime

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="AI Search Log Intelligence", layout="wide")

# ----------------------------------------------------------
# BOT SIGNATURES
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# COLOR MAP
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
ASSET_PATTERN = re.compile(
    r"\.(?:js|css|png|jpe?g|svg|gif|ico|woff2?|ttf|eot|otf|mp4|webm|pdf|txt|xml|json|csv|map)(?:\?|$)",
    re.IGNORECASE,
)

def is_content_url(url: str) -> bool:
    if pd.isna(url):
        return False
    return not bool(ASSET_PATTERN.search(str(url).lower()))

def downsample(df, max_points=500):
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step, :]

# ----------------------------------------------------------
# APP HEADER
# ----------------------------------------------------------
st.title("AI Search Log Intelligence")
st.caption("Log-scale crawl analytics optimized for AI visibility discovery.")

uploaded = st.file_uploader("Upload your log file (CSV or Excel)", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

# ----------------------------------------------------------
# CACHED PROCESSOR
# ----------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_process(file):
    name = file.name.lower()
    df = pd.read_csv(file, low_memory=False) if name.endswith(".csv") else pd.read_excel(file)

    # 1. Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicates early

    # 2. Handle date/time variants
    date_cols = [c for c in df.columns if "time" in c or "date" in c]
    if not date_cols:
        raise ValueError("No valid date/time column found.")
    # pick the most detailed column (usually time_parsed)
    date_col = sorted(date_cols, key=lambda x: 0 if "parsed" in x else 1)[0]
    df = df.rename(columns={date_col: "date"})
    df = df.drop(columns=[c for c in date_cols if c != date_col], errors="ignore")

    # 3. Other renames
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
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing column: {r}")

    # 4. Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["date_bucket"] = df["date"].dt.floor("D")
    df["status"] = df["status"].astype(str).str.strip()

    # 5. Enrichment
    df["is_content_page"] = df["url"].apply(is_content_url)
    df["bot_normalized"] = df["user_agent"].apply(identify_bot)
    df["is_ai_bot"] = df["bot_normalized"].apply(is_ai_bot)
    df["is_visible"] = df["status"].isin(["200", "304"])
    return df

try:
    df = load_and_process(uploaded)
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# ----------------------------------------------------------
# FILTERS
# ----------------------------------------------------------
st.sidebar.header("Filters")
min_date, max_date = df["date_bucket"].min().date(), df["date_bucket"].max().date()
date_range = st.sidebar.date_input("Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = [pd.to_datetime(d) for d in date_range]
    df = df[(df["date_bucket"] >= start) & (df["date_bucket"] <= end)]

show_content_only = st.sidebar.checkbox("Show only content pages (exclude assets)", value=True)
if show_content_only:
    df = df[df["is_content_page"]]

# ----------------------------------------------------------
# METRICS
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# AGGREGATION (w/ downsampling)
# ----------------------------------------------------------
trend = df.groupby(["date_bucket", "bot_normalized"]).size().reset_index(name="hits")
ai_trend = df.groupby(["date_bucket", "is_ai_bot"]).size().reset_index(name="hits")
ai_trend["bot_type"] = np.where(ai_trend["is_ai_bot"], "AI Bots", "Traditional")

bot_summary = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
ai_urls = df[df["is_ai_bot"]].groupby("url").size().reset_index(name="hits").sort_values("hits", ascending=False)
status_summary = df.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
status_summary["percent"] = status_summary["count"] / status_summary.groupby("bot_normalized")["count"].transform("sum") * 100

trend = downsample(trend, max_points=1000)
ai_trend = downsample(ai_trend, max_points=1000)

# ----------------------------------------------------------
# VISUALS
# ----------------------------------------------------------
st.subheader("Crawl Trend by Bot Type")
if len(trend) > 5000:
    chart = (
        alt.Chart(trend)
        .mark_line()
        .encode(x="date_bucket:T", y="hits:Q", color="bot_normalized:N")
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    fig_trend = px.line(
        trend, x="date_bucket", y="hits", color="bot_normalized",
        color_discrete_map=BOT_COLOR_MAP
    )
    fig_trend.update_traces(line=dict(width=2))
    st.plotly_chart(style_plot(fig_trend, "Daily Crawl Volume by Bot"), use_container_width=True)

st.subheader("AI vs Traditional Crawlers")
fig_ai = px.area(
    ai_trend, x="date_bucket", y="hits", color="bot_type",
    color_discrete_map={"AI Bots": "#a970ff", "Traditional": "#00c3a0"}
)
fig_ai.update_traces(opacity=0.8)
st.plotly_chart(style_plot(fig_ai, "AI vs Traditional Crawl Volume"), use_container_width=True)

st.subheader("Top Crawlers by Hit Volume")
fig_bot = px.bar(
    bot_summary.head(15), x="hits", y="bot_normalized", orientation="h",
    color="bot_normalized", color_discrete_map=BOT_COLOR_MAP
)
fig_bot.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(style_plot(fig_bot, "Top 15 Crawlers"), use_container_width=True)

st.subheader("Top AI-Crawled URLs")
fig_urls = px.bar(
    ai_urls.head(25), x="hits", y="url", orientation="h",
    color_discrete_sequence=["#a970ff"]
)
fig_urls.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=200, r=40, t=60, b=60))
st.plotly_chart(style_plot(fig_urls, "Top 25 AI-Crawled URLs"), use_container_width=True)

st.subheader("Status Code Distribution per Bot")
fig_status = px.bar(
    status_summary, x="bot_normalized", y="percent", color="status",
    color_discrete_sequence=px.colors.qualitative.Safe
)
fig_status.update_layout(barmode="stack", xaxis_tickangle=-45)
st.plotly_chart(style_plot(fig_status, "Status Code Share per Bot"), use_container_width=True)

# ----------------------------------------------------------
# INTERPRETATION
# ----------------------------------------------------------
st.markdown("---")
st.subheader("Interpretation Guide")
st.markdown("""
**Coverage Ratio** = (AI-crawled pages ÷ total known pages) × 100  
**Visibility Ratio** = (Visible hits ÷ AI hits) × 100  

Focus on 200/304 for true visibility.  
Layered AI crawlers indicate inclusion in LLM discovery and indexing pipelines.
""")

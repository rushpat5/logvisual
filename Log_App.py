# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from datetime import datetime

st.set_page_config(page_title="Bot Traffic Intelligence", layout="wide")

# ------------------------------------
# Bot Signatures
# ------------------------------------
BOT_SIGNATURES = {
    "Googlebot": [r"googlebot", r"google other", r"google\-webpreview"],
    "Bingbot": [r"bingbot", r"msnbot"],
    "Applebot": [r"applebot"],
    "GPTBot": [r"\bgptbot\b", r"\bgpt\-bot\b"],
    "ChatGPT-User": [r"chatgpt[-_ ]?user", r"chatgptuser"],
    "ClaudeBot": [r"claudebot", r"claude\-bot"],
    "PerplexityBot": [r"perplexity(bot|ai|search)", r"perplexity\-bot", r"perplexitysearch"],
    "Perplexity-User": [r"perplexity[-_ ]?user"],
    "OAI-SearchBot": [r"oai[-_ ]?search", r"openai[-_ ]?search"],
    "CommonCrawl": [r"commoncrawl|ccbot"],
    "AhrefsBot": [r"ahrefs"],
    "SemrushBot": [r"semrush"],
    "Bytespider": [r"bytespider"],
    "MozBot": [r"moz|majestic"],
    "Unclassified Bot": [r""],
}

_COMPILED_PATTERNS = [
    (bot, [re.compile(p, flags=re.I) for p in patterns])
    for bot, patterns in BOT_SIGNATURES.items()
]

def identify_bot(ua: str) -> str:
    if pd.isna(ua):
        return "Unclassified Bot"
    ua = str(ua)
    for bot, pat_list in _COMPILED_PATTERNS:
        for p in pat_list:
            if p.search(ua):
                return bot
    return "Unclassified Bot"


# ------------------------------------
# Header
# ------------------------------------
st.title("Unified Bot Intelligence")
st.caption("Log-file-driven visibility and AI Search analysis — meaningful metrics, not decoration.")

# ------------------------------------
# File Upload
# ------------------------------------
uploaded = st.file_uploader(
    "Upload log file (Excel or CSV). Required columns: user-agent, url, status, date",
    type=["xlsx", "xls", "csv"]
)

if not uploaded:
    st.info("Upload an input file to begin.")
    st.stop()

# ------------------------------------
# Read and Clean Data
# ------------------------------------
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

# Column mapping
user_candidates = ["user_agent", "user-agent", "useragent", "ua"]
url_candidates = ["url", "page", "request_uri", "path", "pathclean"]
status_candidates = ["status", "http_status", "status_code"]
date_candidates = ["date", "timestamp", "time", "datetime"]

def find_col(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    return None

user_col = find_col(user_candidates, df.columns)
url_col = find_col(url_candidates, df.columns)
status_col = find_col(status_candidates, df.columns)
date_col = find_col(date_candidates, df.columns)

missing = []
if not user_col:
    missing.append("user-agent")
if not url_col:
    missing.append("url")
if not status_col:
    missing.append("status")
if not date_col:
    missing.append("date")

if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

# Rename columns
df = df.rename(columns={user_col: "user_agent", url_col: "url", status_col: "status", date_col: "date"})

# Parse types
df["user_agent"] = df["user_agent"].astype(str)
df["url"] = df["url"].astype(str)
df["status"] = df["status"].astype(str)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["url", "date"])
if df.empty:
    st.error("No valid data rows found after cleaning.")
    st.stop()

# ------------------------------------
# Enrichment
# ------------------------------------
df["bot_normalized"] = df["user_agent"].apply(identify_bot)
df["is_ai_bot"] = df["bot_normalized"].str.contains("gpt|claude|perplexity|oai", case=False, na=False)
df["is_visible"] = df["status"].isin(["200", "304"])
df["date_only"] = df["date"].dt.date

# ------------------------------------
# Sidebar Filters
# ------------------------------------
st.sidebar.header("Filters")

min_date, max_date = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    df = df[(df["date_only"] >= start) & (df["date_only"] <= end)]

# Bots: include all by default, persist selection
bots = sorted(df["bot_normalized"].unique())
if "bot_selection" not in st.session_state:
    st.session_state.bot_selection = bots

bot_filter = st.sidebar.multiselect(
    "Bots to include",
    bots,
    default=st.session_state.bot_selection,
)
st.session_state.bot_selection = bot_filter or bots
df = df[df["bot_normalized"].isin(st.session_state.bot_selection)]

# Status codes with safe default handling
statuses = sorted(df["status"].astype(str).unique())
default_status = [s for s in ["200", "304"] if s in statuses]
status_filter = st.sidebar.multiselect("Status codes", statuses, default=default_status or statuses)
df = df[df["status"].isin(status_filter)]

top_n = st.sidebar.number_input("Top N bots/URLs", min_value=5, max_value=100, value=15)

# ------------------------------------
# Key Metrics
# ------------------------------------
total_hits = len(df)
unique_bots = df["bot_normalized"].nunique()
unique_urls = df["url"].nunique()
visible_hits = df[df["is_visible"]].shape[0]
ai_hits = df[df["is_ai_bot"]].shape[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Hits", f"{total_hits:,}")
c2.metric("Unique Bots", unique_bots)
c3.metric("Unique URLs", unique_urls)
c4.metric("Visible Hits", f"{visible_hits:,}")
c5.metric("AI-driven Hits", f"{ai_hits:,}")

st.markdown("---")

# ------------------------------------
# Top Bots
# ------------------------------------
st.subheader("Top Bots by Hits")
bot_hits = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
fig_bot = px.bar(
    bot_hits.head(top_n),
    x="hits",
    y="bot_normalized",
    orientation="h",
    title=f"Top {top_n} Bots",
    labels={"bot_normalized": "", "hits": "Hits"},
)
fig_bot.update_layout(yaxis=dict(autorange="reversed"), height=420, plot_bgcolor="white")
st.plotly_chart(fig_bot, use_container_width=True)

# ------------------------------------
# Crawl Activity: AI vs Traditional
# ------------------------------------
st.subheader("Crawl Activity Trend: AI vs Traditional")
trend = df.groupby(["date_only", "is_ai_bot"]).size().reset_index(name="hits")
trend["date_only"] = pd.to_datetime(trend["date_only"])
fig_trend = px.line(
    trend,
    x="date_only",
    y="hits",
    color=trend["is_ai_bot"].map({True: "AI Bots", False: "Traditional Crawlers"}),
    markers=True,
    title="Daily Crawl Volume: AI vs Traditional",
)
fig_trend.update_layout(height=420, xaxis_title="Date", yaxis_title="Hits", plot_bgcolor="white")
st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------------
# Status Distribution
# ------------------------------------
st.subheader("HTTP Status Distribution (selected bots)")
status_summary = df.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
status_summary["percent"] = status_summary["count"] / status_summary.groupby("bot_normalized")["count"].transform("sum") * 100
top_bots = bot_hits.head(top_n)["bot_normalized"].tolist()
plot_data = status_summary[status_summary["bot_normalized"].isin(top_bots)]
fig_status = px.bar(
    plot_data,
    x="bot_normalized",
    y="percent",
    color="status",
    title="Status Code Share per Bot",
    labels={"bot_normalized": "Bot", "percent": "Share (%)"},
)
fig_status.update_layout(barmode="stack", height=520, xaxis_tickangle=-45, plot_bgcolor="white")
st.plotly_chart(fig_status, use_container_width=True)

# ------------------------------------
# Top Pages Crawled
# ------------------------------------
st.subheader("Top Pages Crawled (by bot)")
top_urls = df.groupby(["bot_normalized", "url"]).size().reset_index(name="hits")
top_urls = top_urls.groupby("bot_normalized").apply(lambda x: x.nlargest(5, "hits")).reset_index(drop=True)
fig_treemap = px.treemap(top_urls, path=["bot_normalized", "url"], values="hits", title=f"Top URLs per Bot")
fig_treemap.update_layout(height=700, plot_bgcolor="white")
st.plotly_chart(fig_treemap, use_container_width=True)

# ------------------------------------
# Deep Dive
# ------------------------------------
st.subheader("Deep Dive: Per-bot Breakdown")
bot_choice = st.selectbox("Select bot", sorted(df["bot_normalized"].unique()))
df_bot = df[df["bot_normalized"] == bot_choice]
bot_summary = df_bot.groupby(["url", "status"]).size().reset_index(name="hits").sort_values("hits", ascending=False)
st.dataframe(bot_summary, use_container_width=True, height=400)
csv = bot_summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Bot Breakdown",
    csv,
    f"{bot_choice}_breakdown_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv",
    "text/csv"
)

st.markdown("---")
st.subheader("Actionable Insights")
st.markdown(
    """
- Focus on 200 and 304 statuses for visibility.
- Separate AI bot analysis (GPTBot, ClaudeBot, PerplexityBot) to track AI search exposure.
- Use per-bot tables to detect redundant crawling or underperforming sections.
- Monitor daily trend divergence between AI and traditional crawlers.
"""
)

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime

st.set_page_config(page_title="Bot Traffic Intelligence", layout="wide")

# -------------------------------
# Utilities and Bot Signatures
# -------------------------------
# Extended signatures (includes tokens from slides like ChatGPT-User, GPTBot, Perplexity, Claude, OAI-SearchBot, etc.)
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
    "Unclassified Bot": [r""],  # fallback
}

# Pre-compile regex for speed
_COMPILED_PATTERNS = []
for bot, patterns in BOT_SIGNATURES.items():
    compiled = [re.compile(p, flags=re.I) for p in patterns]
    _COMPILED_PATTERNS.append((bot, compiled))


@st.cache_data
def identify_bot(ua: str) -> str:
    if pd.isna(ua):
        return "Unclassified Bot"
    ua = str(ua)
    for bot, pat_list in _COMPILED_PATTERNS:
        for p in pat_list:
            if p.search(ua):
                return bot
    return "Unclassified Bot"


def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce", utc=False)


# -------------------------------
# Header and instructions
# -------------------------------
st.title("Unified Bot Intelligence")
st.caption("Log-file-driven visibility and AI Search analysis — meaningful metrics, not decoration.")

# -------------------------------
# File upload
# -------------------------------
uploaded = st.file_uploader("Upload log file (Excel or CSV). Required columns: user-agent, url, status, date", type=["xlsx", "xls", "csv"])
optional_pages_upload = st.file_uploader("Optional: Upload known_pages.csv (one column 'url') to compute site coverage", type=["csv"])

if not uploaded:
    st.info("Upload an input file to begin. Example file expects columns: user-agent, url, status, date (optionally clicks, country, language).")
    st.stop()

# -------------------------------
# Read file robustly
# -------------------------------
def read_input_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
    return df

df = read_input_file(uploaded)

# Normalize columns
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# Required columns check (flexible names handled)
required_candidates = ["user_agent", "user-agent", "useragent", "ua"]
url_candidates = ["url", "page", "request_uri"]
status_candidates = ["status", "http_status", "status_code"]
date_candidates = ["date", "timestamp", "time", "datetime"]

def find_col(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    return None

user_col = find_col(required_candidates, df.columns)
url_col = find_col(url_candidates, df.columns)
status_col = find_col(status_candidates, df.columns)
date_col = find_col(date_candidates, df.columns)

missing = []
if not user_col:
    missing.append("user-agent (user_agent / ua)")
if not url_col:
    missing.append("url (url / page)")
if not status_col:
    missing.append("status (status_code / http_status)")
if not date_col:
    missing.append("date (date / timestamp)")

if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

# Rename for internal consistency
df = df.rename(columns={user_col: "user_agent", url_col: "url", status_col: "status", date_col: "date"})

# Enforce types and clean
df["user_agent"] = df["user_agent"].astype(str)
df["url"] = df["url"].astype(str)
df["status"] = df["status"].astype(str).str.strip()
df["date"] = safe_to_datetime(df["date"])

# Drop rows without URL or date if essential
df = df[~df["url"].isna()]
# If large file, sample warning but continue
if df.shape[0] == 0:
    st.error("No valid rows found after basic cleaning.")
    st.stop()

# Optional known pages
known_pages = None
if optional_pages_upload:
    try:
        kp = pd.read_csv(optional_pages_upload)
        # normalize
        if kp.shape[1] >= 1:
            kp_col = kp.columns[0]
            known_pages = kp[kp_col].astype(str).str.strip().dropna().unique().tolist()
    except Exception:
        known_pages = None

# -------------------------------
# Enrichment
# -------------------------------
# Identify bots
df["bot_normalized"] = df["user_agent"].apply(identify_bot)

# Tag AI-driven vs Traditional
ai_bot_keys = ["gpt", "chatgpt", "claude", "perplexity", "oai", "openai"]
df["is_ai_bot"] = df["bot_normalized"].str.contains("|".join(ai_bot_keys), case=False, na=False)

# Only-count visible statuses for "visibility" metrics (slides recommend 200 & 304)
VISIBLE_STATUS = ["200", "304"]
df["is_visible"] = df["status"].isin(VISIBLE_STATUS)

# Extract date-only for daily aggregates
df["date_only"] = df["date"].dt.date

# -------------------------------
# Sidebar filters
# -------------------------------
st.sidebar.header("Filters")
min_date = df["date"].min().date() if pd.notna(df["date"].min()) else None
max_date = df["date"].max().date() if pd.notna(df["date"].max()) else None
if min_date and max_date:
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df["date_only"] >= start_date) & (df["date_only"] <= end_date)]
else:
    start_date, end_date = None, None

bot_options = sorted(df["bot_normalized"].unique())
bot_filter = st.sidebar.multiselect("Bots to include", bot_options, default=bot_options[:10] if len(bot_options) > 0 else bot_options)
if bot_filter:
    df = df[df["bot_normalized"].isin(bot_filter)]

status_options = sorted(df["status"].unique())
status_selected = st.sidebar.multiselect("Status codes", status_options, default=VISIBLE_STATUS if set(VISIBLE_STATUS).issubset(status_options) else status_options)
if status_selected:
    df = df[df["status"].isin(status_selected)]

top_n = st.sidebar.number_input("Top N bots/URLs", min_value=5, max_value=100, value=15)

# -------------------------------
# Key Metrics (KPI)
# -------------------------------
total_hits = int(len(df))
unique_bots = int(df["bot_normalized"].nunique())
unique_urls = int(df["url"].nunique())
visible_hits = int(df[df["is_visible"]].shape[0])
ai_hits = int(df[df["is_ai_bot"]].shape[0])
# Pages crawled by AI bots (unique)
ai_pages = int(df[df["is_ai_bot"]]["url"].nunique())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Hits", f"{total_hits:,}")
col2.metric("Unique Bots", f"{unique_bots}")
col3.metric("Unique URLs", f"{unique_urls}")
col4.metric("Visible hits (200/304)", f"{visible_hits:,}")
col5.metric("AI-driven hits", f"{ai_hits:,}")

st.markdown("---")

# -------------------------------
# Top Bots (bar)
# -------------------------------
st.subheader("Top Bots by Hits")
bot_hits = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
fig_bot = px.bar(bot_hits.head(top_n), x="hits", y="bot_normalized", orientation="h", title=f"Top {top_n} Bots", labels={"bot_normalized": "", "hits": "Hits"})
fig_bot.update_layout(yaxis=dict(autorange="reversed"), height=420, plot_bgcolor="white")
st.plotly_chart(fig_bot, use_container_width=True)

# -------------------------------
# Visibility: AI vs Traditional (trend & share)
# -------------------------------
# Trend daily
st.subheader("Crawl Activity Trend: AI vs Traditional")
trend = df.groupby(["date_only", "is_ai_bot"]).size().reset_index(name="hits")
trend["date_only"] = pd.to_datetime(trend["date_only"])
fig_trend = px.line(trend, x="date_only", y="hits", color=trend["is_ai_bot"].map({True: "AI Bots", False: "Traditional Crawlers"}), markers=True, title="Daily Crawl Volume: AI vs Traditional")
fig_trend.update_layout(height=420, xaxis_title="Date", yaxis_title="Hits", plot_bgcolor="white")
st.plotly_chart(fig_trend, use_container_width=True)

# Share pie
st.subheader("Traffic Share: AI vs Traditional")
share = df["is_ai_bot"].value_counts().rename(index={True: "AI Bots", False: "Traditional Crawlers"})
fig_share = px.pie(values=share.values, names=share.index, title="Traffic Share")
fig_share.update_traces(textinfo="percent+label")
st.plotly_chart(fig_share, use_container_width=True)

# -------------------------------
# HTTP Status Efficiency (stacked / percent)
# -------------------------------
st.subheader("HTTP Status Distribution (selected bots)")
status_summary = (
    df.groupby(["bot_normalized", "status"])
    .size()
    .reset_index(name="count")
)
# convert to percent within each bot
status_summary["percent"] = status_summary["count"] / status_summary.groupby("bot_normalized")["count"].transform("sum") * 100

# Choose top bots for clarity
top_bots_list = bot_hits.head(min(len(bot_hits), top_n))["bot_normalized"].tolist()
status_plot = status_summary[status_summary["bot_normalized"].isin(top_bots_list)]
fig_status = px.bar(status_plot, x="bot_normalized", y="percent", color="status", title="Status Code Share per Bot (percent stacked)", labels={"bot_normalized": "Bot", "percent": "Share (%)"})
fig_status.update_layout(barmode="stack", height=520, xaxis_tickangle=-45, plot_bgcolor="white")
st.plotly_chart(fig_status, use_container_width=True)

# -------------------------------
# Top pages crawled (treemap/sunburst)
# -------------------------------
st.subheader("Top Pages Crawled (by bot)")
top_urls = df.groupby(["bot_normalized", "url"]).size().reset_index(name="hits")
# Filter to selected top bots and take top N urls per bot
def top_n_per_group(df_in, groupcol, valuecol, n=5):
    return df_in.groupby(groupcol).apply(lambda x: x.nlargest(n, valuecol)).reset_index(drop=True)

top_urls = top_n_per_group(top_urls[top_urls["bot_normalized"].isin(top_bots_list)], "bot_normalized", "hits", n=top_n)
fig_treemap = px.treemap(top_urls, path=["bot_normalized", "url"], values="hits", title=f"Top {top_n} URLs per Bot (Treemap)")
fig_treemap.update_layout(height=700, plot_bgcolor="white")
st.plotly_chart(fig_treemap, use_container_width=True)

# -------------------------------
# Visibility metric and site coverage
# -------------------------------
st.subheader("AI Search Visibility & Site Coverage (recommended metrics from deck)")
# Visibility = (AI bot visible hits) / (AI bot hits) * 100  -- shows proportion of AI hits that are 'visible' statuses
ai_df = df[df["is_ai_bot"]]
ai_total = ai_df.shape[0]
ai_visible = ai_df[ai_df["is_visible"]].shape[0]
visibility_pct = (ai_visible / ai_total * 100) if ai_total > 0 else np.nan

# Pages crawled vs known pages
crawled_pages = df[df["is_ai_bot"]]["url"].nunique() if not df[df["is_ai_bot"]].empty else 0
known_pages_count = len(known_pages) if known_pages else np.nan
coverage_pct = (crawled_pages / known_pages_count * 100) if known_pages_count and known_pages_count > 0 else np.nan

col_a, col_b = st.columns(2)
col_a.metric("AI Search Visibility (AI visible hits / AI hits)", f"{visibility_pct:.1f}%" if not np.isnan(visibility_pct) else "N/A")
col_b.metric("AI Pages Crawled", f"{crawled_pages:,}" + (f" / {known_pages_count:,} known" if not np.isnan(known_pages_count) else ""))

# -------------------------------
# Citation missed opportunities
# -------------------------------
st.subheader("Citation / Missed Click Opportunities")
# If 'clicks' or 'ga_clicks' or 'impressions' exist in input, calculate ratio (Clicks / Bot hits) * 100
click_candidates = [c for c in df.columns if "click" in c or "impression" in c or "gsc_clicks" in c or "clicks" in c]
click_col = click_candidates[0] if click_candidates else None

if click_col:
    # aggregate by url
    agg = df.groupby("url").agg(bot_hits=("url", "size"), clicks=(click_col, "sum")).reset_index()
    agg["clicks"] = agg["clicks"].fillna(0)
    agg["clicks_perc_of_hits"] = agg["clicks"] / agg["bot_hits"] * 100
    missed = agg[(agg["bot_hits"] > 0) & (agg["clicks"] == 0)].sort_values("bot_hits", ascending=False).head(50)
    st.markdown("Pages heavily crawled by bots but getting zero clicks (top 50).")
    st.dataframe(missed[["url", "bot_hits"]].rename(columns={"bot_hits": "bot_hits (no clicks)"}), use_container_width=True, height=350)
else:
    st.markdown("No 'clicks' or 'impression' column found in uploaded file. To compute missed click opportunities, upload a file with clicks/impressions or join with Search Console data.")

# -------------------------------
# Deep Dive: per-bot detail and export
# -------------------------------
st.subheader("Deep Dive: Per-bot Drilldown")
bot_choice = st.selectbox("Select bot", options=sorted(df["bot_normalized"].unique()), index=0)
df_bot = df[df["bot_normalized"] == bot_choice]

# Summary table for selected bot
bot_summary = (
    df_bot.groupby(["url", "status"])
    .size()
    .reset_index(name="hits")
    .sort_values("hits", ascending=False)
)
st.dataframe(bot_summary, use_container_width=True, height=400)
csv = bot_summary.to_csv(index=False).encode("utf-8")
st.download_button("Download bot breakdown CSV", csv, f"{bot_choice}_breakdown_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv", "text/csv")

# -------------------------------
# Notes & recommended actions (derived from deck)
# -------------------------------
st.markdown("---")
st.subheader("Actionable takeaways (from log analysis)")
st.markdown(
    """
- Focus on `200` and `304` responses for visibility metrics (filter out client/server errors and redirects unless they are relevant).
- Track AI bots separately: measure how many unique pages AI bots visit and which pages are heavily crawled but never get clicks (missed citation opportunity).
- If you can upload a 'known pages' list, compute coverage (crawled pages / known pages) to monitor AI search indexing progress.
- Use the per-URL export to identify pages to protect (paywalls, private content) or improve (high-crawl, low-click pages).
"""
)

# End

# app.py — Robust Log Visualizer (Final rebuild)
# - Local-time parsing (Asia/Kolkata) without UTC day-shift
# - Keeps all rows (marks invalid timestamps; does not drop)
# - Case-insensitive header mapping for your schema
# - Clear raw vs filtered metrics and diagnostics
# - Downsampling for responsive plotting
# - Export excluded rows
#
# Tested against the example schema you provided:
# File, LineNo, ClientIP, Time, Time_parsed, Date, HourBucket, Method, Path, PathClean, Query, QueryParams,
# Status, StatusClass, Bytes, Referer, User-Agent, Bot Type, IsStatic, IsMobile, Section, SessionID, HTTP_Version

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
from typing import Optional

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="AI Search Log Intelligence (Stable)", layout="wide")
st.title("AI Search Log Intelligence — Stable Rebuild")
st.caption("Local-time parsing (Asia/Kolkata). Raw totals preserved. Filters apply only to visuals.")

# -------------------------
# Bot signatures (expandable)
# -------------------------
BOT_SIGNATURES = {
    "Googlebot": [r"googlebot"],
    "Bingbot": [r"bingbot|msnbot"],
    "GPTBot": [r"\bgptbot\b"],
    "ChatGPT-User": [r"chatgpt[-_ ]?user"],
    "ClaudeBot": [r"claudebot"],
    "PerplexityBot": [r"perplexity(bot|ai|search)"],
    "OAI-SearchBot": [r"oai[-_ ]?search|openai[-_ ]?search"],
    "Applebot": [r"applebot"],
    "AhrefsBot": [r"ahrefs"],
    "SemrushBot": [r"semrush"],
    "Bytespider": [r"bytespider"],
    "Unclassified": [r".*"],
}
_COMPILED = [(name, [re.compile(p, re.I) for p in pats]) for name, pats in BOT_SIGNATURES.items()]

def identify_bot(ua: Optional[str]) -> str:
    if pd.isna(ua):
        return "Unclassified"
    s = str(ua)
    for bot, pats in _COMPILED:
        for p in pats:
            if p.search(s):
                return bot
    return "Unclassified"

def is_ai_bot(bot_name: Optional[str]) -> bool:
    if pd.isna(bot_name):
        return False
    b = str(bot_name).lower()
    return any(k in b for k in ("gpt", "claude", "perplexity", "oai", "bytespider", "applebot"))

# -------------------------
# Helpers
# -------------------------
ASSET_PATTERN = re.compile(r"\.(?:js|css|png|jpe?g|gif|svg|ico|woff2?|ttf)(?:\?|$)", re.I)

def is_content_url(url: Optional[str]) -> bool:
    if pd.isna(url):
        return False
    return not bool(ASSET_PATTERN.search(str(url).lower()))

def downsample(df: pd.DataFrame, max_points: int = 1000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()

def find_time_candidates(columns):
    """
    Return an ordered list of plausible time columns (existing in dataframe),
    preferring canonical 'time' then 'time_parsed' then 'hourbucket' then 'date'.
    """
    preferred = ["time", "time_parsed", "hourbucket", "date", "timestamp", "datetime"]
    cols_lower = [c.lower() for c in columns]
    picks = []
    for p in preferred:
        for i, c in enumerate(cols_lower):
            if p == c:
                picks.append(columns[i])
    # also include any column that contains 'time' or 'date' as fallback (preserve original order)
    for i, c in enumerate(cols_lower):
        if ("time" in c or "date" in c) and columns[i] not in picks:
            picks.append(columns[i])
    return picks

def to_local_naive(series: pd.Series, tz_name: str = "Asia/Kolkata") -> pd.Series:
    """
    Convert a series of timestamps to naive datetimes in local tz (tz_name).
    Rules:
      - Parse with pd.to_datetime (no forced utc)
      - If parsed values are tz-naive: localize to tz_name (assume they are local)
      - If parsed values are tz-aware (contain offsets): convert to tz_name
      - Finally return tz-naive datetimes representing local wall time
    Conservative: ambiguous/nonexistent times will become NaT (pandas behavior).
    """
    parsed = pd.to_datetime(series, errors="coerce")
    # If all parsed are NaT, return as-is
    if parsed.dropna().empty:
        return parsed
    # If dtype has tz info (pandas stores tz-aware as dtype 'datetime64[ns, tz]')
    if parsed.dt.tz is None:
        # tz-naive: assume local timestamps -> localize to tz_name
        try:
            localized = parsed.dt.tz_localize(tz_name, ambiguous="NaT", nonexistent="NaT")
            return localized.dt.tz_convert(tz_name).dt.tz_localize(None)
        except Exception:
            # fallback: return naive parsed values
            return parsed
    else:
        # tz-aware: convert to local timezone then remove tzinfo
        try:
            converted = parsed.dt.tz_convert(tz_name)
            return converted.dt.tz_localize(None)
        except Exception:
            return parsed.dt.tz_convert(tz_name).dt.tz_localize(None)

# -------------------------
# Upload & Load
# -------------------------
uploaded = st.file_uploader("Upload log file (CSV or Excel). Expected headers similar to sample (PathClean/Path, Time, Status, User-Agent).", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file, low_memory=False)
    else:
        df = pd.read_excel(file)
    # normalize headers: strip, lowercase, replace spaces and hyphens with underscore
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    # drop duplicate columns (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

try:
    df_raw = load_file(uploaded)
except Exception as e:
    st.error(f"Failed to read uploaded file: {e}")
    st.stop()

# show detected columns
st.subheader("Detected columns (normalized)")
st.write(list(df_raw.columns))

# -------------------------
# Map canonical columns
# -------------------------
# Map common source column names to internal canonical names (url, status, user_agent)
rename_map = {}
cols = set(df_raw.columns)

# url: prefer pathclean, fallback to path
if "pathclean" in cols:
    rename_map["pathclean"] = "url"
elif "path" in cols:
    rename_map["path"] = "url"

# user agent
if "user_agent" not in cols and "user-agent" in cols:
    rename_map["user-agent"] = "user_agent"

# status (case-insensitive already)
# ensure 'status' exists; if not try 'statuscode' or 'http_status' common variants
if "status" not in cols:
    for alt in ("status_code", "http_status", "statuscode"):
        if alt in cols:
            rename_map[alt] = "status"
            break

if rename_map:
    df_raw = df_raw.rename(columns=rename_map)
    cols = set(df_raw.columns)

# Validate required columns exist
required = ["url", "status", "user_agent"]
missing = [c for c in required if c not in cols]
if missing:
    st.error(f"Missing required columns after normalization: {missing}. The app requires these columns (or their variants). Aborting.")
    st.stop()

# -------------------------
# Timestamp selection & parsing
# -------------------------
time_candidates = find_time_candidates(df_raw.columns)
if not time_candidates:
    st.error("No plausible time/date column found. Expected names containing 'time', 'time_parsed', 'hourbucket', or 'date'. Aborting.")
    st.stop()

# Offer the user a choice if multiple options detected (default to first/best)
st.sidebar.header("Timestamp column selection")
selected_time_col = st.sidebar.selectbox("Choose timestamp column (used for date filtering)", options=time_candidates, index=0, help="If your preferred timestamp column is present, pick it here. The app parses to Asia/Kolkata local time.")
st.sidebar.markdown("Detected candidates (best-first): " + ", ".join(time_candidates))

# Parse timestamps conservatively into local naive datetimes (Asia/Kolkata)
df_raw["date"] = to_local_naive(df_raw[selected_time_col], tz_name="Asia/Kolkata")
df_raw["date_bucket"] = df_raw["date"].dt.floor("D")
df_raw["has_valid_date"] = ~df_raw["date"].isna()

# -------------------------
# Enrichment (no row drops)
# -------------------------
df_raw["is_content_page"] = df_raw["url"].apply(is_content_url)
df_raw["status"] = df_raw["status"].astype(str)
df_raw["bot_normalized"] = df_raw["user_agent"].apply(identify_bot)
df_raw["is_ai_bot"] = df_raw["bot_normalized"].apply(is_ai_bot)
df_raw["is_visible"] = df_raw["status"].isin(["200", "304"])

# -------------------------
# Diagnostics: raw counts
# -------------------------
total_rows = len(df_raw)
valid_dates = int(df_raw["has_valid_date"].sum())
invalid_dates = total_rows - valid_dates

st.success(f"Loaded {total_rows:,} rows. {valid_dates:,} rows have parseable timestamps; {invalid_dates:,} have invalid/missing timestamps and are retained (not dropped).")

# Show date bucket distribution (stringified to include 'NaT')
st.write("Date buckets (local Asia/Kolkata) — 'NaT' indicates unparseable timestamps:")
date_counts = df_raw["date_bucket"].astype(str).value_counts(dropna=False).sort_index()
st.dataframe(date_counts.rename_axis("date_bucket").reset_index(name="count"))

# -------------------------
# Sidebar filters (date range based on parseable dates)
# -------------------------
st.sidebar.header("Filters")
if valid_dates > 0:
    min_date = df_raw.loc[df_raw["has_valid_date"], "date_bucket"].min().date()
    max_date = df_raw.loc[df_raw["has_valid_date"], "date_bucket"].max().date()
else:
    min_date = pd.Timestamp.now().date()
    max_date = min_date

date_range = st.sidebar.date_input("Date range (applies only to rows with parseable timestamps)", value=(min_date, max_date), min_value=min_date, max_value=max_date)
show_content_only = st.sidebar.checkbox("Show only content pages (exclude static assets)", value=False)
top_n = st.sidebar.number_input("Top N bots / URLs", min_value=5, max_value=100, value=15)

# -------------------------
# Build filtered dataset (for visuals only)
# - rows without parseable date are excluded from date filtering (they remain in raw totals)
# -------------------------
df_filtered = df_raw.copy()
start = pd.to_datetime(date_range[0]).tz_localize(None)
end = pd.to_datetime(date_range[1]).tz_localize(None)
mask_date = (df_filtered["has_valid_date"]) & (df_filtered["date_bucket"] >= start) & (df_filtered["date_bucket"] <= end)
df_filtered = df_filtered[mask_date]

if show_content_only:
    df_filtered = df_filtered[df_filtered["is_content_page"]]

filtered_rows = len(df_filtered)
excluded_rows = total_rows - filtered_rows

# -------------------------
# Totals: raw vs filtered
# -------------------------
st.subheader("Totals and Filtered Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Hits (raw file)", f"{total_rows:,}")
c2.metric("Unique Bots (raw)", df_raw["bot_normalized"].nunique())
c3.metric("AI-driven Hits (raw)", f"{df_raw[df_raw['is_ai_bot']].shape[0]:,}")
c4.metric("Visible Hits (raw, 200/304)", f"{df_raw[df_raw['is_visible']].shape[0]:,}")
c5.metric("Content-page Hits (raw)", f"{df_raw[df_raw['is_content_page']].shape[0]:,}")

fc1, fc2, fc3, fc4, fc5 = st.columns(5)
fc1.metric("Filtered Hits (for visuals)", f"{filtered_rows:,}")
fc2.metric("Unique Bots (filtered)", df_filtered["bot_normalized"].nunique() if filtered_rows else 0)
fc3.metric("AI-driven Hits (filtered)", f"{df_filtered[df_filtered['is_ai_bot']].shape[0]:,}" if filtered_rows else "0")
fc4.metric("Visible Hits (filtered)", f"{df_filtered[df_filtered['is_visible']].shape[0]:,}" if filtered_rows else "0")
fc5.metric("Content-page Hits (filtered)", f"{df_filtered[df_filtered['is_content_page']].shape[0]:,}" if filtered_rows else "0")

st.info(f"Rows excluded from visuals: {excluded_rows:,} (rows with no parseable timestamp or outside date range and/or filtered by content).")

# Provide sample of excluded rows for debugging (if any)
if excluded_rows > 0:
    st.subheader("Sample excluded rows (first 10) — these are in raw data but not in filtered visuals")
    excluded_idx = df_raw.index.difference(df_filtered.index)
    st.dataframe(df_raw.loc[excluded_idx].head(10))

# -------------------------
# If no filtered rows, show warning
# -------------------------
if filtered_rows == 0:
    st.warning("No rows in filtered dataset. Adjust date range or uncheck content-only filter.")
else:
    # -------------------------
    # Crawl Trend by Bot
    # -------------------------
    st.subheader("Crawl Trend by Bot (filtered)")
    trend = df_filtered.groupby(["date_bucket", "bot_normalized"]).size().reset_index(name="hits")
    trend = trend.sort_values(["date_bucket", "hits"], ascending=[True, False])
    trend_plot = downsample(trend, max_points=2000)
    fig_trend = px.line(trend_plot, x="date_bucket", y="hits", color="bot_normalized", title="Daily Crawl Volume by Bot (filtered)", template="plotly_dark")
    fig_trend.update_layout(legend_title_text="Bot", height=420)
    st.plotly_chart(fig_trend, use_container_width=True)

    # -------------------------
    # AI vs Traditional
    # -------------------------
    st.subheader("AI vs Traditional (filtered)")
    ai_trend = df_filtered.groupby(["date_bucket", "is_ai_bot"]).size().reset_index(name="hits")
    ai_trend["bot_type"] = np.where(ai_trend["is_ai_bot"], "AI Bots", "Traditional")
    ai_trend_plot = downsample(ai_trend, max_points=2000)
    fig_ai = px.area(ai_trend_plot, x="date_bucket", y="hits", color="bot_type", title="AI vs Traditional Crawl Volume", template="plotly_dark")
    fig_ai.update_layout(height=360)
    st.plotly_chart(fig_ai, use_container_width=True)

    # -------------------------
    # Top crawlers
    # -------------------------
    st.subheader(f"Top {top_n} Crawlers (filtered)")
    bot_summary = df_filtered.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
    fig_bot = px.bar(bot_summary.head(top_n), x="hits", y="bot_normalized", orientation="h", title="Top Crawlers", template="plotly_dark", color="bot_normalized")
    fig_bot.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
    st.plotly_chart(fig_bot, use_container_width=True)

    # -------------------------
    # Top AI URLs
    # -------------------------
    st.subheader("Top AI-Crawled URLs (filtered)")
    ai_urls = df_filtered[df_filtered["is_ai_bot"]].groupby("url").size().reset_index(name="hits").sort_values("hits", ascending=False)
    fig_urls = px.bar(ai_urls.head(25), x="hits", y="url", orientation="h", title="Top AI-Crawled URLs", template="plotly_dark")
    fig_urls.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=220))
    st.plotly_chart(fig_urls, use_container_width=True)

    # -------------------------
    # Status distribution
    # -------------------------
    st.subheader("Status Code Distribution per Bot (filtered)")
    status_summary = df_filtered.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
    status_summary["percent"] = status_summary["count"] / status_summary.groupby("bot_normalized")["count"].transform("sum") * 100
    fig_status = px.bar(status_summary, x="bot_normalized", y="percent", color="status", title="Status Share per Bot", template="plotly_dark")
    fig_status.update_layout(barmode="stack", xaxis_tickangle=-45)
    st.plotly_chart(fig_status, use_container_width=True)

# -------------------------
# Export excluded rows
# -------------------------
st.markdown("---")
st.subheader("Export / Debug")
if excluded_rows > 0:
    excluded_df = df_raw.loc[excluded_idx]
    csv_bytes = excluded_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download excluded rows (CSV)", data=csv_bytes, file_name="excluded_rows.csv", mime="text/csv")
else:
    st.write("No excluded rows to export (current filters include all parsed rows).")

st.caption("Notes: timestamps are interpreted/lifted to Asia/Kolkata local wall time to avoid day-shift. Rows with invalid timestamps are retained in raw totals but are excluded from date-filtered visuals. Adjust timestamp column selection if you want to use a different source of dates (Time, Time_parsed, HourBucket, etc.).")

# Log_App.py — Final: Explicit totals + filtered metrics + diagnostics
# Stable, defensive, and transparent: totals always match raw file; filtered metrics shown separately.
# Use this as your Streamlit app.

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="AI Search Log Intelligence", layout="wide")
st.title("AI Search Log Intelligence")
st.caption("Totals always reflect the uploaded file. Filtered metrics are shown separately with diagnostics.")

# -----------------------
# Bot signatures
# -----------------------
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
_COMPILED = [(k, [re.compile(p, re.I) for p in v]) for k, v in BOT_SIGNATURES.items()]

def identify_bot(ua: str) -> str:
    if pd.isna(ua):
        return "Unclassified"
    ua = str(ua)
    for bot, pats in _COMPILED:
        for p in pats:
            if p.search(ua):
                return bot
    return "Unclassified"

def is_ai_bot(bot_name: str) -> bool:
    if pd.isna(bot_name):
        return False
    b = str(bot_name).lower()
    return any(x in b for x in ("gpt", "claude", "perplexity", "oai", "bytespider"))

# -----------------------
# Helpers
# -----------------------
ASSET_PATTERN = re.compile(
    r"\.(?:js|css|png|jpg|jpeg|gif|ico|woff2?|ttf|svg)(?:\?|$)", re.IGNORECASE
)

def is_content_url(url: str) -> bool:
    if pd.isna(url):
        return False
    return not bool(ASSET_PATTERN.search(str(url).lower()))

def find_time_column(cols):
    # fuzzy search for likely time columns (prefer full ISO-like 'time')
    candidates = ["time", "time_parsed", "timestamp", "datetime", "date", "hourbucket"]
    for cand in candidates:
        for c in cols:
            if cand in c:
                return c
    # fallback to first column that contains 'time'/'date' substring
    for c in cols:
        if "time" in c or "date" in c:
            return c
    return None

def normalize_datetime(series):
    s = pd.to_datetime(series, errors="coerce", utc=True)
    # keep NaT for truly invalid values; do NOT forward fill here
    return s.dt.tz_localize(None)

def downsample(df, max_points=1000):
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step, :].copy()

# -----------------------
# File upload
# -----------------------
uploaded = st.file_uploader("Upload your log file (.csv or .xlsx)", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(file):
    name = file.name.lower()
    df = pd.read_csv(file, low_memory=False) if name.endswith(".csv") else pd.read_excel(file)
    # normalize column names (strip, lowercase, replace spaces/hyphens)
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    # drop duplicate column names (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

try:
    df_raw = load_data(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# show detected columns for transparency
st.subheader("File schema (detected columns)")
st.write(list(df_raw.columns))

# -----------------------
# Ensure required fields exist (or map)
# -----------------------
# map common names to canonical names if present
rename_map = {}
if "pathclean" in df_raw.columns:
    rename_map["pathclean"] = "url"
elif "path" in df_raw.columns:
    rename_map["path"] = "url"

if "user_agent" not in df_raw.columns and "user-agent" in df_raw.columns:
    rename_map["user-agent"] = "user_agent"
if rename_map:
    df_raw = df_raw.rename(columns=rename_map)

# check for minimal required columns
required = ["url", "status", "user_agent"]
missing_required = [c for c in required if c not in df_raw.columns]
if missing_required:
    st.error(f"Missing required columns in uploaded file: {missing_required}. The app requires these columns.")
    st.stop()

# -----------------------
# Timestamp parsing (do not drop rows)
# -----------------------
time_col = find_time_column(df_raw.columns)
if not time_col:
    st.error("No recognizable time/date column found. Expected names containing: time, time_parsed, timestamp, datetime, date, hourbucket.")
    st.stop()

# create parsed date column (may contain NaT)
df_raw["date"] = normalize_datetime(df_raw[time_col])
# date_bucket for day-level grouping (NaT remains NaT)
df_raw["date_bucket"] = df_raw["date"].dt.floor("D")

# mark which rows have valid date for filtering/plots
df_raw["has_valid_date"] = ~df_raw["date"].isna()

# -----------------------
# Enrichment (no drops)
# -----------------------
df_raw["is_content_page"] = df_raw["url"].apply(is_content_url)
df_raw["status"] = df_raw["status"].astype(str)
df_raw["bot_normalized"] = df_raw["user_agent"].apply(identify_bot)
df_raw["is_ai_bot"] = df_raw["bot_normalized"].apply(is_ai_bot)
df_raw["is_visible"] = df_raw["status"].isin(["200", "304"])

# -----------------------
# Diagnostics: raw counts
# -----------------------
total_rows = len(df_raw)
rows_with_date = df_raw["has_valid_date"].sum()
rows_without_date = total_rows - rows_with_date

st.success(f"Loaded {total_rows:,} rows. {rows_with_date:,} rows contain parseable timestamps; {rows_without_date:,} rows have invalid/missing timestamps (kept).")

# show distribution of date buckets (helpful to spot rows outside slider)
st.write("Date buckets (counts) — NaT indicates missing/unparseable timestamps:")
date_counts = df_raw["date_bucket"].astype(str).value_counts(dropna=False).sort_index()
st.dataframe(date_counts.rename_axis("date_bucket").reset_index(name="count"))

# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.header("Filters")

# build date slider domain from only parseable dates
if rows_with_date > 0:
    min_date = df_raw.loc[df_raw["has_valid_date"], "date_bucket"].min().date()
    max_date = df_raw.loc[df_raw["has_valid_date"], "date_bucket"].max().date()
else:
    # no valid dates—default to today
    min_date = pd.Timestamp.utcnow().date()
    max_date = min_date

date_range = st.sidebar.date_input("Date Range (applies only to rows with parseable timestamps)", (min_date, max_date), min_value=min_date, max_value=max_date)
show_content_only = st.sidebar.checkbox("Show only content pages (exclude assets)", value=False)
top_n_bots = st.sidebar.number_input("Top N bots/URLs (for bar charts)", min_value=5, max_value=100, value=15)

# -----------------------
# Build filtered dataframe (only for visuals/filtered metrics)
# - Do NOT mutate df_raw.
# - Rows with no parseable date are excluded from date filtering, but they remain in raw totals.
# -----------------------
df_filtered = df_raw.copy()

# apply date filter only to rows that have valid dates; invalid-date rows are excluded from visual dataset unless we choose to include them separately.
start_date, end_date = None, None
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0]).tz_localize(None)
    end_date = pd.to_datetime(date_range[1]).tz_localize(None)
    # mask: has_valid_date and within range
    mask_date = (df_filtered["has_valid_date"]) & (df_filtered["date_bucket"] >= start_date) & (df_filtered["date_bucket"] <= end_date)
    df_filtered = df_filtered[mask_date]
else:
    # single date selected case
    sel = pd.to_datetime(date_range).tz_localize(None)
    mask_date = (df_filtered["has_valid_date"]) & (df_filtered["date_bucket"].dt.date == sel.date())
    df_filtered = df_filtered[mask_date]

# apply content filter if enabled
if show_content_only:
    df_filtered = df_filtered[df_filtered["is_content_page"]]

# counts for diagnostics
filtered_rows = len(df_filtered)
excluded_by_date = df_raw[~((df_raw["has_valid_date"]) & (df_raw["date_bucket"] >= (start_date if start_date is not None else min_date)) & (df_raw["date_bucket"] <= (end_date if end_date is not None else max_date)))].shape[0] if start_date is not None else 0
# Better breakdown:
excluded_date_count = df_raw.shape[0] - df_raw[(df_raw["has_valid_date"]) & (df_raw["date_bucket"] >= start_date) & (df_raw["date_bucket"] <= end_date)].shape[0] if start_date is not None else 0
excluded_by_content = df_raw.shape[0] - df_raw[df_raw["is_content_page"]].shape[0] if show_content_only else 0

# -----------------------
# Show transparent summary (All vs Filtered)
# -----------------------
st.subheader("Totals & Filtered Metrics (transparent)")

col1, col2, col3, col4, col5 = st.columns(5)

# Totals derived from raw file (never filtered)
col1.metric("Total Hits (All Data, unfiltered)", f"{total_rows:,}")
col2.metric("Unique Bots (All Data)", df_raw["bot_normalized"].nunique())
col3.metric("Visible Hits (All Data, 200/304)", f"{df_raw[df_raw['is_visible']].shape[0]:,}")
col4.metric("AI-driven Hits (All Data)", f"{df_raw[df_raw['is_ai_bot']].shape[0]:,}")
col5.metric("Content-page Hits (All Data)", f"{df_raw[df_raw['is_content_page']].shape[0]:,}")

st.markdown("---")

# Filtered metrics (based on sidebar)
st.subheader("Filtered Metrics (used for charts / diagnostics)")

fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
fcol1.metric("Filtered Hits (in selected date range / content filter)", f"{filtered_rows:,}")
fcol2.metric("Unique Bots (Filtered)", df_filtered["bot_normalized"].nunique() if filtered_rows>0 else 0)
fcol3.metric("Visible Hits (Filtered, 200/304)", f"{df_filtered[df_filtered['is_visible']].shape[0]:,}" if filtered_rows>0 else "0")
fcol4.metric("AI-driven Hits (Filtered)", f"{df_filtered[df_filtered['is_ai_bot']].shape[0]:,}" if filtered_rows>0 else "0")
fcol5.metric("Content-page Hits (Filtered)", f"{df_filtered[df_filtered['is_content_page']].shape[0]:,}" if filtered_rows>0 else "0")

# diagnostics block: how many rows excluded and why
st.info(
    f"Excluded rows summary: total excluded = {total_rows - filtered_rows:,} "
    f"(rows without parseable timestamps or outside date range: {excluded_date_count if start_date is not None else 0:,}; "
    f"excluded by content filter (if enabled): {excluded_by_content:,})."
)

# show small sample of excluded rows for user debugging
if total_rows - filtered_rows > 0:
    st.subheader("Sample of rows excluded from filtered dataset (first 10)")
    # produce excluded set = rows in raw but not in filtered (match by index)
    excluded_idx = df_raw.index.difference(df_filtered.index)
    st.dataframe(df_raw.loc[excluded_idx].head(10))

# -----------------------
# Charts (use df_filtered)
# -----------------------
st.markdown("---")
st.subheader("Crawl Trend by Bot Type (Filtered)")

if filtered_rows == 0:
    st.warning("No rows in filtered dataset. Adjust Date Range or uncheck content filter.")
else:
    trend = df_filtered.groupby(["date_bucket", "bot_normalized"]).size().reset_index(name="hits")
    trend = downsample(trend, max_points=1000)
    fig_trend = px.line(trend, x="date_bucket", y="hits", color="bot_normalized",
                        color_discrete_sequence=px.colors.qualitative.Dark2,
                        markers=False)
    fig_trend.update_layout(template="plotly_dark", height=420)
    fig_trend.update_xaxes(title="date")
    fig_trend.update_yaxes(title="hits")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("AI vs Traditional (Filtered)")
    ai_trend = df_filtered.groupby(["date_bucket", "is_ai_bot"]).size().reset_index(name="hits")
    ai_trend["bot_type"] = np.where(ai_trend["is_ai_bot"], "AI Bots", "Traditional")
    ai_trend = downsample(ai_trend, max_points=1000)
    fig_ai = px.area(ai_trend, x="date_bucket", y="hits", color="bot_type", template="plotly_dark")
    fig_ai.update_layout(height=360)
    st.plotly_chart(fig_ai, use_container_width=True)

    st.subheader("Top Crawlers by Hit Volume (Filtered)")
    bot_summary = df_filtered.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
    fig_bot = px.bar(bot_summary.head(int(top_n_bots)), x="hits", y="bot_normalized", orientation="h",
                     color="bot_normalized", template="plotly_dark")
    fig_bot.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_bot, use_container_width=True)

    st.subheader("Top AI-Crawled URLs (Filtered)")
    ai_urls = df_filtered[df_filtered["is_ai_bot"]].groupby("url").size().reset_index(name="hits").sort_values("hits", ascending=False)
    fig_urls = px.bar(ai_urls.head(25), x="hits", y="url", orientation="h", template="plotly_dark")
    fig_urls.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=220))
    st.plotly_chart(fig_urls, use_container_width=True)

    st.subheader("Status Code Distribution per Bot (Filtered)")
    status_summary = df_filtered.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
    status_summary["percent"] = status_summary["count"] / status_summary.groupby("bot_normalized")["count"].transform("sum") * 100
    fig_status = px.bar(status_summary, x="bot_normalized", y="percent", color="status", template="plotly_dark")
    fig_status.update_layout(barmode="stack", xaxis_tickangle=-45)
    st.plotly_chart(fig_status, use_container_width=True)

# -----------------------
# Exports & Debug
# -----------------------
st.markdown("---")
st.subheader("Debug & Export")
st.write("If you want to inspect excluded rows in full, download them for offline analysis.")
if total_rows - filtered_rows > 0:
    excluded_df = df_raw.loc[df_raw.index.difference(df_filtered.index)]
    csv = excluded_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download excluded rows (CSV)", csv, "excluded_rows.csv", "text/csv")

st.caption("Notes: 'Total Hits (All Data)' comes from the raw uploaded file and never changes with filters. Charts and filtered metrics respect the sidebar controls and exclude rows without parseable timestamps from the date filtering step.")

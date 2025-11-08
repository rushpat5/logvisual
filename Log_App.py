# app.py — Clean rebuild: AI Search Log Intelligence
# - Correct local-time parsing (Asia/Kolkata)
# - Raw totals never change
# - Filters apply only to visuals/filtered metrics
# - Invalid timestamps flagged, not dropped

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px

st.set_page_config(page_title="AI Search Log Intelligence", layout="wide")
st.title("AI Search Log Intelligence — Fresh Build")
st.caption("Accurate local-time parsing (Asia/Kolkata). Raw totals preserved. Filters affect only visuals.")

# ---------------------------
# Bot signatures
# ---------------------------
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
    s = str(ua)
    for bot, pats in _COMPILED:
        for p in pats:
            if p.search(s):
                return bot
    return "Unclassified"

def is_ai_bot(bot_name: str) -> bool:
    if pd.isna(bot_name):
        return False
    b = str(bot_name).lower()
    return any(x in b for x in ("gpt", "claude", "perplexity", "oai", "bytespider", "applebot"))

# ---------------------------
# Helpers
# ---------------------------
ASSET_PATTERN = re.compile(r"\.(?:js|css|png|jpe?g|gif|svg|ico|woff2?|ttf)(?:\?|$)", re.I)
def is_content_url(url: str) -> bool:
    if pd.isna(url):
        return False
    return not bool(ASSET_PATTERN.search(str(url).lower()))

def find_time_column(cols):
    candidates = ["time", "time_parsed", "timestamp", "datetime", "date", "hourbucket"]
    cols_l = [c.lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(cols_l):
            if cand in c:
                return list(cols)[i]
    # fallback heuristic: first column containing 'time' or 'date'
    for i, c in enumerate(cols_l):
        if "time" in c or "date" in c:
            return list(cols)[i]
    return None

def to_local_naive(series, tz_name="Asia/Kolkata"):
    """
    Convert a series of timestamps to naive datetimes in local tz (tz_name).
    Rules:
     - parse without forcing UTC (pd.to_datetime(...))
     - if parsed values are tz-naive: localize to tz_name (assume they are local)
     - if parsed values are tz-aware (contain offsets like +05:30): convert to tz_name
     - finally, remove tz info and return naive datetimes representing local wall time
    """
    s = pd.to_datetime(series, errors="coerce")
    # if series is entirely tz-naive, .dt.tz is None
    try:
        if s.dt.tz is None:
            # treat as local (tz-naive) timestamps -> assign Asia/Kolkata
            s_loc = s.dt.tz_localize(tz_name, ambiguous='NaT', nonexistent='NaT')
            # drop tz to return naive local datetimes
            return s_loc.dt.tz_convert(tz_name).dt.tz_localize(None)
        else:
            # tz-aware values -> convert to local timezone, then drop tz
            s_conv = s.dt.tz_convert(tz_name)
            return s_conv.dt.tz_localize(None)
    except Exception:
        # fallback: return parsed naive datetimes (may contain UTC-shifted values)
        return pd.to_datetime(series, errors="coerce")

def downsample(df, max_points=1000):
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()

# ---------------------------
# Upload
# ---------------------------
uploaded = st.file_uploader("Upload log (.csv or .xlsx) — expected headers similar to provided sample", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.stop()

@st.cache_data(show_spinner=False)
def load_raw(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file, low_memory=False)
    else:
        df = pd.read_excel(file)
    # normalize headers for ease
    df.columns = [str(c).strip() for c in df.columns]
    return df

try:
    df_raw = load_raw(uploaded)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

# show detected columns
st.subheader("Detected columns")
st.write(list(df_raw.columns))

# ---------------------------
# Basic mapping / required checks
# ---------------------------
# map obvious names to canonical names
rename_map = {}
cols_lower = {c.lower(): c for c in df_raw.columns}
# url
if "pathclean" in cols_lower:
    rename_map[cols_lower["pathclean"]] = "url"
elif "path" in cols_lower:
    rename_map[cols_lower["path"]] = "url"
# user agent
if "user-agent" in cols_lower:
    rename_map[cols_lower["user-agent"]] = "user_agent"
elif "user_agent" in cols_lower:
    rename_map[cols_lower["user_agent"]] = "user_agent"
# time candidates will be handled later
if rename_map:
    df_raw = df_raw.rename(columns=rename_map)

required = ["url", "status", "user_agent"]
missing = [r for r in required if r not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}. App requires these columns. Aborting.")
    st.stop()

# ---------------------------
# Timestamp parsing (local)
# ---------------------------
time_col = find_time_column(df_raw.columns)
if not time_col:
    st.error("No recognizable time/date column found (expected names like Time, Time_parsed, Date, HourBucket). Aborting.")
    st.stop()

# create local-naive datetime (Asia/Kolkata) and day bucket
df_raw["date"] = to_local_naive(df_raw[time_col], tz_name="Asia/Kolkata")
df_raw["date_bucket"] = df_raw["date"].dt.floor("D")
df_raw["has_valid_date"] = ~df_raw["date"].isna()

# ---------------------------
# Enrichment (no drops)
# ---------------------------
df_raw["is_content_page"] = df_raw["url"].apply(is_content_url)
df_raw["status"] = df_raw["status"].astype(str)
df_raw["bot_normalized"] = df_raw["user_agent"].apply(identify_bot)
df_raw["is_ai_bot"] = df_raw["bot_normalized"].apply(is_ai_bot)
df_raw["is_visible"] = df_raw["status"].isin(["200", "304"])

# ---------------------------
# Diagnostics + totals
# ---------------------------
total_rows = len(df_raw)
valid_dates = df_raw["has_valid_date"].sum()
invalid_dates = total_rows - valid_dates

st.success(f"Loaded {total_rows:,} rows. {valid_dates:,} rows have parseable timestamps; {invalid_dates:,} rows have missing/invalid timestamps and are retained (but excluded from date-filtered visuals).")

st.write("Date bucket counts (after localizing to Asia/Kolkata):")
st.dataframe(df_raw["date_bucket"].astype(str).value_counts().sort_index().rename_axis("date_bucket").reset_index(name="count"))

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filters")
# date slider - domain uses only parseable dates
if valid_dates > 0:
    min_date = df_raw.loc[df_raw["has_valid_date"], "date_bucket"].min().date()
    max_date = df_raw.loc[df_raw["has_valid_date"], "date_bucket"].max().date()
else:
    min_date = pd.Timestamp.utcnow().date()
    max_date = min_date

date_range = st.sidebar.date_input("Date Range (applies to parseable timestamps)", (min_date, max_date), min_value=min_date, max_value=max_date)
show_content_only = st.sidebar.checkbox("Show only content pages (exclude assets)", value=False)
top_n = st.sidebar.number_input("Top N (bars)", min_value=5, max_value=100, value=15)

# ---------------------------
# Build filtered dataset (for visuals)
# ---------------------------
df_filtered = df_raw.copy()

# apply date filter only to rows that have parseable dates
start = pd.to_datetime(date_range[0]).tz_localize(None)
end = pd.to_datetime(date_range[1]).tz_localize(None)
mask_date = (df_filtered["has_valid_date"]) & (df_filtered["date_bucket"] >= start) & (df_filtered["date_bucket"] <= end)
df_filtered = df_filtered[mask_date]

# content filter
if show_content_only:
    df_filtered = df_filtered[df_filtered["is_content_page"]]

filtered_rows = len(df_filtered)

st.info(f"Raw rows: {total_rows:,} — Filtered rows (for visuals): {filtered_rows:,} — Rows excluded: {total_rows - filtered_rows:,}")

# ---------------------------
# Top-level totals (raw) and filtered metrics
# ---------------------------
st.subheader("Totals — Raw vs Filtered")

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Total Hits (raw file)", f"{total_rows:,}")
r2.metric("Unique Bots (raw)", df_raw["bot_normalized"].nunique())
r3.metric("Visible Hits (raw, 200/304)", f"{df_raw[df_raw['is_visible']].shape[0]:,}")
r4.metric("AI-driven Hits (raw)", f"{df_raw[df_raw['is_ai_bot']].shape[0]:,}")
r5.metric("Content-page Hits (raw)", f"{df_raw[df_raw['is_content_page']].shape[0]:,}")

f1, f2, f3, f4, f5 = st.columns(5)
f1.metric("Filtered Hits (visuals)", f"{filtered_rows:,}")
f2.metric("Unique Bots (filtered)", df_filtered["bot_normalized"].nunique() if filtered_rows else 0)
f3.metric("Visible Hits (filtered)", f"{df_filtered[df_filtered['is_visible']].shape[0]:,}" if filtered_rows else "0")
f4.metric("AI-driven Hits (filtered)", f"{df_filtered[df_filtered['is_ai_bot']].shape[0]:,}" if filtered_rows else "0")
f5.metric("Content-page Hits (filtered)", f"{df_filtered[df_filtered['is_content_page']].shape[0]:,}" if filtered_rows else "0")

st.markdown("---")

# ---------------------------
# If nothing to show after filters, warn
# ---------------------------
if filtered_rows == 0:
    st.warning("No rows in filtered dataset. Adjust the date range or uncheck 'content only' filter.")
else:
    # Crawl trend
    st.subheader("Crawl Trend by Bot (filtered)")
    trend = df_filtered.groupby(["date_bucket", "bot_normalized"]).size().reset_index(name="hits")
    trend = downsample(trend, max_points=1000)
    fig = px.line(trend, x="date_bucket", y="hits", color="bot_normalized", title="Daily Crawl Volume by Bot (filtered)", template="plotly_dark")
    fig.update_layout(legend_title_text="Bot")
    st.plotly_chart(fig, use_container_width=True, height=420)

    # AI vs Traditional
    st.subheader("AI vs Traditional (filtered)")
    ai_trend = df_filtered.groupby(["date_bucket", "is_ai_bot"]).size().reset_index(name="hits")
    ai_trend["type"] = np.where(ai_trend["is_ai_bot"], "AI Bots", "Traditional")
    ai_trend = downsample(ai_trend, max_points=1000)
    fig2 = px.area(ai_trend, x="date_bucket", y="hits", color="type", title="AI vs Traditional Crawl Volume", template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True, height=360)

    # Top crawlers
    st.subheader(f"Top {top_n} Crawlers by Hits (filtered)")
    bot_summary = df_filtered.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
    fig3 = px.bar(bot_summary.head(top_n), x="hits", y="bot_normalized", orientation="h", title="Top Crawlers (filtered)", template="plotly_dark")
    fig3.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)

    # Top AI URLs
    st.subheader("Top AI-crawled URLs (filtered)")
    ai_urls = df_filtered[df_filtered["is_ai_bot"]].groupby("url").size().reset_index(name="hits").sort_values("hits", ascending=False)
    fig4 = px.bar(ai_urls.head(25), x="hits", y="url", orientation="h", title="Top AI-Crawled URLs", template="plotly_dark")
    fig4.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=240))
    st.plotly_chart(fig4, use_container_width=True)

    # Status distribution
    st.subheader("Status Code Distribution per Bot (filtered)")
    status_summary = df_filtered.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
    status_summary["percent"] = status_summary["count"] / status_summary.groupby("bot_normalized")["count"].transform("sum") * 100
    fig5 = px.bar(status_summary, x="bot_normalized", y="percent", color="status", title="Status Share per Bot", template="plotly_dark")
    fig5.update_layout(barmode="stack", xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)

# ---------------------------
# Debug / Export excluded rows
# ---------------------------
st.markdown("---")
st.subheader("Excluded Rows (from filtered dataset)")
excluded_idx = df_raw.index.difference(df_filtered.index)
st.write(f"Excluded rows count: {len(excluded_idx):,} (these are rows that were outside the selected date range or excluded by content filter).")
if len(excluded_idx) > 0:
    st.dataframe(df_raw.loc[excluded_idx].head(10))
    csv = df_raw.loc[excluded_idx].to_csv(index=False).encode("utf-8")
    st.download_button("Download excluded rows (CSV)", data=csv, file_name="excluded_rows.csv", mime="text/csv")

st.markdown("---")
st.caption("Notes: timestamps are parsed and normalized to Asia/Kolkata local wall time to avoid day-shift caused by treating offsets as UTC. Rows with invalid timestamps are retained in raw totals but excluded from date-filtered visuals.")

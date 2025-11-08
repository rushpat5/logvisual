# app.py — AI Search Log Intelligence (Final Stable Build)
# Compatible with detailed_hits.csv structure
# Fixes:
#  - Case-insensitive column mapping
#  - Accurate local-time parsing (Asia/Kolkata)
#  - No UTC day shift
#  - Raw totals preserved
#  - Filters affect visuals only
#  - Optimized plotting & stability

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px

st.set_page_config(page_title="AI Search Log Intelligence", layout="wide")
st.title("AI Search Log Intelligence — Final Stable Build")
st.caption("Accurate local-time parsing (Asia/Kolkata). Raw totals preserved. Filters affect only visuals.")

# -------------------------------------------
# Bot Signatures
# -------------------------------------------
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

# -------------------------------------------
# Helpers
# -------------------------------------------
ASSET_PATTERN = re.compile(r"\.(js|css|png|jpg|jpeg|gif|svg|ico|woff2?|ttf)(\?|$)", re.I)
def is_content_url(url: str) -> bool:
    if pd.isna(url):
        return False
    return not bool(ASSET_PATTERN.search(str(url).lower()))

def find_time_column(cols):
    candidates = ["time", "time_parsed", "timestamp", "datetime", "hourbucket", "date"]
    for c in cols:
        if any(cand in c for cand in candidates):
            return c
    return None

def to_local_naive(series, tz_name="Asia/Kolkata"):
    s = pd.to_datetime(series, errors="coerce")
    try:
        if s.dt.tz is None:
            s_loc = s.dt.tz_localize(tz_name, ambiguous='NaT', nonexistent='NaT')
            return s_loc.dt.tz_convert(tz_name).dt.tz_localize(None)
        else:
            return s.dt.tz_convert(tz_name).dt.tz_localize(None)
    except Exception:
        return s

def downsample(df, max_points=1000):
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()

# -------------------------------------------
# Upload Section
# -------------------------------------------
uploaded = st.file_uploader("Upload log (.csv or .xlsx)", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.stop()

@st.cache_data(show_spinner=False)
def load_raw(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file, low_memory=False)
    else:
        df = pd.read_excel(file)
    # normalize headers for matching
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

try:
    df_raw = load_raw(uploaded)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

st.subheader("Detected columns")
st.write(list(df_raw.columns))

# -------------------------------------------
# Column Mapping
# -------------------------------------------
rename_map = {}
cols_lower = set(df_raw.columns)

# URL field
if "pathclean" in cols_lower:
    rename_map["pathclean"] = "url"
elif "path" in cols_lower:
    rename_map["path"] = "url"

# User agent
if "user-agent" in cols_lower:
    rename_map["user-agent"] = "user_agent"
elif "user_agent" in cols_lower:
    rename_map["user_agent"] = "user_agent"

if rename_map:
    df_raw = df_raw.rename(columns=rename_map)

required = ["url", "status", "user_agent"]
missing = [r for r in required if r not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}.")
    st.stop()

# -------------------------------------------
# Timestamp Parsing (Local)
# -------------------------------------------
time_col = find_time_column(df_raw.columns)
if not time_col:
    st.error("No recognizable time/date column found.")
    st.stop()

df_raw["date"] = to_local_naive(df_raw[time_col], tz_name="Asia/Kolkata")
df_raw["date_bucket"] = df_raw["date"].dt.floor("D")
df_raw["has_valid_date"] = ~df_raw["date"].isna()

# -------------------------------------------
# Enrichment
# -------------------------------------------
df_raw["is_content_page"] = df_raw["url"].apply(is_content_url)
df_raw["status"] = df_raw["status"].astype(str)
df_raw["bot_normalized"] = df_raw["user_agent"].apply(identify_bot)
df_raw["is_ai_bot"] = df_raw["bot_normalized"].apply(is_ai_bot)
df_raw["is_visible"] = df_raw["status"].isin(["200", "304"])

# -------------------------------------------
# Totals
# -------------------------------------------
total_rows = len(df_raw)
valid_dates = df_raw["has_valid_date"].sum()
invalid_dates = total_rows - valid_dates

st.success(f"Loaded {total_rows:,} rows. {valid_dates:,} valid timestamps. {invalid_dates:,} invalid timestamps retained.")

st.write("Date distribution (Asia/Kolkata):")
st.dataframe(
    df_raw["date_bucket"].astype(str).value_counts().sort_index().rename_axis("date_bucket").reset_index(name="count")
)

# -------------------------------------------
# Sidebar Filters
# -------------------------------------------
st.sidebar.header("Filters")
if valid_dates > 0:
    min_date = df_raw.loc[df_raw["has_valid_date"], "date_bucket"].min().date()
    max_date = df_raw.loc[df_raw["has_valid_date"], "date_bucket"].max().date()
else:
    min_date = pd.Timestamp.utcnow().date()
    max_date = min_date

date_range = st.sidebar.date_input("Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)
show_content_only = st.sidebar.checkbox("Show only content pages", value=False)
top_n = st.sidebar.number_input("Top N (bars)", min_value=5, max_value=100, value=15)

# -------------------------------------------
# Filtered Data
# -------------------------------------------
df_filtered = df_raw.copy()

start = pd.to_datetime(date_range[0]).tz_localize(None)
end = pd.to_datetime(date_range[1]).tz_localize(None)
mask = (df_filtered["has_valid_date"]) & (df_filtered["date_bucket"] >= start) & (df_filtered["date_bucket"] <= end)
df_filtered = df_filtered[mask]

if show_content_only:
    df_filtered = df_filtered[df_filtered["is_content_page"]]

filtered_rows = len(df_filtered)
excluded_rows = total_rows - filtered_rows

st.info(f"Raw: {total_rows:,}  |  Filtered: {filtered_rows:,}  |  Excluded: {excluded_rows:,}")

# -------------------------------------------
# Metrics
# -------------------------------------------
st.subheader("Key Metrics")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Total Hits", f"{total_rows:,}")
r2.metric("Unique Bots", df_raw["bot_normalized"].nunique())
r3.metric("AI-driven Hits", f"{df_raw[df_raw['is_ai_bot']].shape[0]:,}")
r4.metric("Visible Hits (200/304)", f"{df_raw[df_raw['is_visible']].shape[0]:,}")
r5.metric("Content Pages", f"{df_raw[df_raw['is_content_page']].shape[0]:,}")

f1, f2, f3, f4, f5 = st.columns(5)
f1.metric("Filtered Hits", f"{filtered_rows:,}")
f2.metric("Unique Bots (filtered)", df_filtered["bot_normalized"].nunique() if filtered_rows else 0)
f3.metric("AI-driven Hits (filtered)", f"{df_filtered[df_filtered['is_ai_bot']].shape[0]:,}")
f4.metric("Visible Hits (filtered)", f"{df_filtered[df_filtered['is_visible']].shape[0]:,}")
f5.metric("Content Pages (filtered)", f"{df_filtered[df_filtered['is_content_page']].shape[0]:,}")

st.markdown("---")

# -------------------------------------------
# Visuals
# -------------------------------------------
if filtered_rows == 0:
    st.warning("No rows after filters. Adjust date or content filter.")
else:
    # Trend
    st.subheader("Daily Crawl Volume by Bot (filtered)")
    trend = df_filtered.groupby(["date_bucket", "bot_normalized"]).size().reset_index(name="hits")
    trend = downsample(trend)
    fig = px.line(trend, x="date_bucket", y="hits", color="bot_normalized", template="plotly_dark", title="Crawl Trend")
    fig.update_layout(legend_title_text="Bot", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # AI vs Traditional
    st.subheader("AI vs Traditional (filtered)")
    ai_trend = df_filtered.groupby(["date_bucket", "is_ai_bot"]).size().reset_index(name="hits")
    ai_trend["type"] = np.where(ai_trend["is_ai_bot"], "AI Bots", "Traditional")
    fig2 = px.area(ai_trend, x="date_bucket", y="hits", color="type", template="plotly_dark", title="AI vs Traditional Trend")
    st.plotly_chart(fig2, use_container_width=True)

    # Top Bots
    st.subheader(f"Top {top_n} Bots (filtered)")
    bot_summary = df_filtered.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
    fig3 = px.bar(bot_summary.head(top_n), x="hits", y="bot_normalized", orientation="h", template="plotly_dark", title="Top Bots")
    fig3.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)

    # Top AI URLs
    st.subheader("Top AI-Crawled URLs (filtered)")
    ai_urls = df_filtered[df_filtered["is_ai_bot"]].groupby("url").size().reset_index(name="hits").sort_values("hits", ascending=False)
    fig4 = px.bar(ai_urls.head(25), x="hits", y="url", orientation="h", template="plotly_dark", title="Top AI URLs")
    fig4.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=250))
    st.plotly_chart(fig4, use_container_width=True)

    # Status Distribution
    st.subheader("Status Code Distribution per Bot (filtered)")
    status_summary = df_filtered.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
    status_summary["percent"] = status_summary["count"] / status_summary.groupby("bot_normalized")["count"].transform("sum") * 100
    fig5 = px.bar(status_summary, x="bot_normalized", y="percent", color="status", barmode="stack", template="plotly_dark", title="Status Code Share")
    fig5.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")
st.caption("Timestamps are parsed in Asia/Kolkata local time. Raw totals are preserved. Filters affect visuals only.")

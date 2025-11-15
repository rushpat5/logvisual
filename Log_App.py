# Log_App.py
# Streamlit Bot & UA Log Analyzer — revised
# Single-file version that fixes the dt.strftime error, removes the "Column mapping (auto-suggested)" heading,
# and classifies requests by a single user-agent name token.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import re
from datetime import timedelta

st.set_page_config(page_title="Bot & UA Log Analyzer", layout="wide")

# ----------------------------
# CONFIG / SIGNATURES
# ----------------------------
BOT_SIGNALS = [
    "gptbot", "chatgpt-user", "openai", "oai-searchbot", "perplexity", "perplexitybot",
    "claude", "claudebot", "anthropic", "mistral", "bytespider", "ccbot", "serpapi",
    "copilot", "gpt-4o", "bingbot", "googlebot", "yandex", "duckduckbot", "baiduspider",
    "slurp", "ahrefsbot", "semrushbot", "bingpreview", "sogou", "curl", "wget",
    "python-requests"
]
# common browser tokens to extract simple browser name
BROWSER_TOKENS = ["chrome", "firefox", "safari", "edge", "msie", "trident", "opera", "opr", "mozilla"]

STATIC_DEFAULT_EXTS = [".css", ".js", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".map", ".eot", ".webp"]

# ----------------------------
# HELPERS
# ----------------------------
def normalize_timestamp(series):
    """Return tz-naive UTC datetimes (or NaT) for a pandas Series."""
    s = pd.to_datetime(series, errors="coerce", utc=True)
    # drop tzinfo to make grouping easier (hour etc.)
    try:
        return s.dt.tz_convert(None)
    except Exception:
        # if series contains NaT or cannot tz-convert, just return as-is but ensure dtype
        return pd.to_datetime(s, errors="coerce").dt.tz_localize(None) if s.dtype == object else s

def detect_static(path, extra_exts=None, regex=None):
    if not isinstance(path, str) or path.strip() == "":
        return False
    p = path.split("?", 1)[0].lower()
    exts = list(STATIC_DEFAULT_EXTS)
    if extra_exts:
        exts += [e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower() for e in extra_exts.split(",") if e.strip()]
    for ext in exts:
        if p.endswith(ext):
            return True
    if regex:
        try:
            if re.search(regex, p):
                return True
        except Exception:
            pass
    return False

def extract_agent_token(ua):
    """
    Return a single canonical token for the UA string:
    - first match from BOT_SIGNALS (exact substring)
    - else first browser token match (chrome, firefox, safari, edge, etc.)
    - else 'unknown' or 'other' depending on content
    """
    if not isinstance(ua, str) or ua.strip() == "":
        return "unknown"
    s = ua.lower()
    # prefer exact signal match
    for token in BOT_SIGNALS:
        if token in s:
            return token
    # browsers: find first browser token
    for b in BROWSER_TOKENS:
        if b in s:
            # normalize some tokens: 'opr' -> opera, 'msie'/'trident' -> ie
            if b in ("opr", "opera"):
                return "opera"
            if b in ("msie", "trident"):
                return "ie"
            if b == "mozilla" and "firefox" not in s and "chrome" not in s:
                # generic 'mozilla' without firefox/chrome is ambiguous; treat as 'mozilla'
                return "mozilla"
            return b
    # additional heuristics: if contains "bot" anywhere return 'bot' (generic)
    if "bot" in s or "spider" in s or "crawler" in s or "scraper" in s:
        return "bot"
    # else attempt short token from first token before slash or space
    first_token = s.split()[0].split("/")[0]
    if len(first_token) <= 30 and re.match(r"^[a-z0-9\-\_\.]+$", first_token):
        return first_token
    return "other"

def safe_format_datetime(series, fmt="%Y-%m-%d %H:%M"):
    """Format datetimes in a Series safely; return strings (empty for NaT)."""
    # Ensure datetime dtype
    sdt = pd.to_datetime(series, errors="coerce")
    # if dtype is datetime64, use dt.strftime; else fallback apply
    if hasattr(sdt.dt, "strftime"):
        return sdt.dt.strftime(fmt).fillna("")
    else:
        return sdt.apply(lambda x: x.strftime(fmt) if pd.notna(x) else "")

# ----------------------------
# SIDEBAR — UPLOAD & SETTINGS
# ----------------------------
st.sidebar.header("Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload CSV log file", type=["csv", "txt", "tsv"])

if not uploaded:
    st.info("Upload your CSV/TSV file to start analysis.")
    st.stop()

@st.cache_data
def load_csv(file):
    """Try multiple engines and delimiters to robustly load logs."""
    # read bytes
    raw = file.getvalue()
    # try to detect delimiter by first line
    try:
        sample = raw.decode("utf-8", errors="replace").splitlines()[0]
    except Exception:
        sample = ""
    delim = "\t" if "\t" in sample else ","
    try:
        return pd.read_csv(io.BytesIO(raw), sep=delim, low_memory=False)
    except Exception:
        # fallback: try python engine and allow bad lines
        try:
            return pd.read_csv(io.BytesIO(raw), sep=delim, engine="python", on_bad_lines="skip", low_memory=False)
        except Exception:
            # last resort: read as single-column and attempt to parse
            return pd.read_fwf(io.BytesIO(raw))

df = load_csv(uploaded)
st.sidebar.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

# Minimal column remapping UI (placed under same Upload & Settings area)
cols = df.columns.tolist()
def find_col(*names):
    for n in names:
        for c in cols:
            if n.lower() in c.lower():
                return c
    return None

col_time = st.sidebar.selectbox("Timestamp column", ["(none)"] + cols, index=(cols.index(find_col("time")) + 1 if find_col("time") else 0))
col_ua = st.sidebar.selectbox("User-Agent column", ["(none)"] + cols, index=(cols.index(find_col("user")) + 1 if find_col("user") else 0))
col_path = st.sidebar.selectbox("Path/URL column", ["(none)"] + cols, index=(cols.index(find_col("path", "uri", "url")) + 1 if find_col("path", "uri", "url") else 0))
col_status = st.sidebar.selectbox("Status column", ["(none)"] + cols, index=(cols.index(find_col("status", "code")) + 1 if find_col("status", "code") else 0))
col_ip = st.sidebar.selectbox("Client IP column (optional)", ["(none)"] + cols, index=(cols.index(find_col("ip", "client")) + 1 if find_col("ip", "client") else 0))
col_static = st.sidebar.selectbox("IsStatic column (optional)", ["(none)"] + cols, index=(cols.index(find_col("static")) + 1 if find_col("static") else 0))

# Static detection settings
st.sidebar.markdown("Static asset detection:")
extra_exts = st.sidebar.text_input("Extra extensions (comma-separated)", value="")
static_regex = st.sidebar.text_input("Static regex (optional)", value="")

# Preview
st.subheader("Preview (first 8 rows)")
st.dataframe(df.head(8))

# ----------------------------
# PREPARE DATA
# ----------------------------
# Parse timestamp if available
if col_time != "(none)":
    # Attempt robust parse including epochs
    s = df[col_time]
    # attempt numeric epoch detection
    try:
        numeric = pd.to_numeric(s, errors="coerce")
    except Exception:
        numeric = pd.Series([np.nan] * len(s))
    df["timestamp"] = pd.NaT
    if numeric.notna().any():
        # heuristics
        if (numeric > 1e12).any():
            df["timestamp"] = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
        elif (numeric > 1e9).any():
            df["timestamp"] = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
        # if many NaT remain, try parsing original strings
        if df["timestamp"].isna().mean() > 0.5:
            df["timestamp"] = pd.to_datetime(s, errors="coerce", utc=True, infer_datetime_format=True)
    else:
        df["timestamp"] = pd.to_datetime(s, errors="coerce", utc=True, infer_datetime_format=True)
else:
    df["timestamp"] = pd.NaT

# normalize timestamps
df["timestamp"] = normalize_timestamp(df["timestamp"])

# other fields
df["user_agent"] = df[col_ua].astype(str) if col_ua != "(none)" else ""
df["path"] = df[col_path].astype(str) if col_path != "(none)" else ""
df["status"] = df[col_status].astype(str).str.strip() if col_status != "(none)" else "unknown"
df["client_ip"] = df[col_ip].astype(str) if col_ip != "(none)" else ""
if col_static != "(none)":
    df["is_static"] = df[col_static].astype(str).str.upper().isin(["TRUE", "1", "YES"])
else:
    df["is_static"] = df["path"].apply(lambda p: detect_static(p, extra_exts, static_regex))

# New: simplified agent token extraction (single-name)
df["agent_token"] = df["user_agent"].apply(extract_agent_token)
# boolean bot flag (heuristic): token in bot signals or token == "bot"
df["is_bot"] = df["agent_token"].apply(lambda t: (t in BOT_SIGNALS) or (t == "bot"))

df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour

# ----------------------------
# FILTERS
# ----------------------------
st.sidebar.header("Filters")
agent_tokens = sorted(df["agent_token"].unique())
selected_agents = st.sidebar.multiselect("Agent token(s)", options=agent_tokens, default=agent_tokens[:6] if len(agent_tokens)>6 else agent_tokens)
all_statuses = sorted(df["status"].unique())
default_statuses = [s for s in ["200", "304", "404"] if s in all_statuses]
selected_statuses = st.sidebar.multiselect("Status Codes", options=all_statuses, default=default_statuses)
content_filter = st.sidebar.radio("URL Scope", ["All URLs", "Content-only (exclude static assets)"], index=1)
min_date, max_date = (df["date"].min(), df["date"].max()) if df["timestamp"].notna().any() else (None, None)
if min_date and max_date:
    date_range = st.sidebar.date_input("Date Range", (min_date, max_date))
else:
    date_range = None
top_n = st.sidebar.slider("Top N URLs", 5, 100, 25)

# Apply filters
df_f = df.copy()
if selected_agents:
    df_f = df_f[df_f["agent_token"].isin(selected_agents)]
if selected_statuses:
    df_f = df_f[df_f["status"].isin(selected_statuses)]
if content_filter == "Content-only (exclude static assets)":
    df_f = df_f[~df_f["is_static"]]
if date_range:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    df_f = df_f[(df_f["timestamp"] >= start) & (df_f["timestamp"] < end)]

# ----------------------------
# KPIs
# ----------------------------
st.title("Bot & UA Traffic Dashboard")
c1, c2, c3, c4 = st.columns(4)
total_hits = len(df_f)
bot_hits = df_f["is_bot"].sum()
unique_urls = df_f["path"].nunique()
unique_agents = df_f["agent_token"].nunique()

c1.metric("Total Hits", f"{total_hits:,}")
c2.metric("Bot Hits (heuristic)", f"{bot_hits:,}", f"{bot_hits/total_hits*100:.1f}%" if total_hits else "0%")
c3.metric("Unique URLs", f"{unique_urls:,}")
c4.metric("Unique Agents", f"{unique_agents:,}")

st.markdown("---")

# ----------------------------
# VISUALS
# ----------------------------
if not df_f.empty and df_f["timestamp"].notna().any():
    st.subheader("Traffic Over Time (Hourly)")
    ts = df_f.groupby([pd.Grouper(key="timestamp", freq="H"), "agent_token"]).size().reset_index(name="hits")
    fig = px.area(ts, x="timestamp", y="hits", color="agent_token", title="Hourly Traffic by Agent Token")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("HTTP Status Codes by Agent Token")
status_df = df_f.groupby(["status", "agent_token"]).size().reset_index(name="count")
if not status_df.empty:
    fig = px.bar(status_df, x="status", y="count", color="agent_token", barmode="group", title="Status Codes by Agent Token")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No status data in filtered set.")

# Top URLs
st.subheader(f"Top {top_n} URLs by Hits")
url_stats = (
    df_f.groupby("path")
    .agg(
        hits=("path", "count"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        common_status=("status", lambda x: x.mode().iloc[0] if not x.mode().empty else "")
    )
    .reset_index()
    .sort_values("hits", ascending=False)
    .head(top_n)
)

# Ensure first_seen/last_seen are datetimes before formatting; safe format
url_stats["first_seen"] = pd.to_datetime(url_stats["first_seen"], errors="coerce")
url_stats["last_seen"] = pd.to_datetime(url_stats["last_seen"], errors="coerce")
url_stats["first_seen_fmt"] = safe_format_datetime(url_stats["first_seen"], "%Y-%m-%d %H:%M")
url_stats["last_seen_fmt"] = safe_format_datetime(url_stats["last_seen"], "%Y-%m-%d %H:%M")

st.dataframe(url_stats[["path","hits","common_status","first_seen_fmt","last_seen_fmt"]].rename(columns={
    "first_seen_fmt":"first_seen",
    "last_seen_fmt":"last_seen"
}), use_container_width=True)

fig_urls = px.bar(url_stats, x="hits", y="path", orientation="h", title=f"Top {top_n} URLs")
st.plotly_chart(fig_urls, use_container_width=True)

# Hourly Heatmap
st.subheader("Hourly Heatmap (Agent Tokens)")
heat = df_f.groupby(["hour", "agent_token"]).size().reset_index(name="count")
if not heat.empty:
    heat_pivot = heat.pivot(index="hour", columns="agent_token", values="count").fillna(0)
    st.dataframe(heat_pivot)
    # plotly imshow expects numeric matrix-like; transpose to have agent tokens on y-axis
    fig_heat = px.imshow(heat_pivot.T, labels=dict(x="Hour", y="Agent Token", color="Hits"), title="Hourly Crawl Intensity (by token)")
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No hourly data available for heatmap.")

# Crawl depth proxy
st.subheader("Crawl Depth & Unique Pages per Agent Token")
crawl_depth = df_f.groupby("agent_token")["path"].nunique().reset_index().rename(columns={"path":"unique_urls"})
st.dataframe(crawl_depth.sort_values("unique_urls", ascending=False).head(200), use_container_width=True)

st.markdown("---")

# ----------------------------
# BASIC HEURISTICS / ANOMALY SIGNALS
# ----------------------------
st.subheader("Heuristics & Signals")

# 1) Burst detection by client_ip (requests per second)
if "client_ip" in df_f.columns and df_f["client_ip"].notna().any():
    bursts = df_f.groupby("client_ip").agg(
        hits=("client_ip","count"),
        unique_paths=("path","nunique"),
        first_seen=("timestamp","min"),
        last_seen=("timestamp","max")
    ).reset_index()
    bursts["duration_s"] = (bursts["last_seen"] - bursts["first_seen"]).dt.total_seconds().clip(lower=1)
    bursts["rps"] = bursts["hits"] / bursts["duration_s"]
    st.markdown("**Top clients by requests/sec**")
    st.dataframe(bursts.sort_values("rps", ascending=False).head(20)[["client_ip","hits","unique_paths","rps"]], use_container_width=True)
else:
    st.info("No client IP column selected; IP heuristics skipped.")

# 2) Agents requesting many distinct URLs
agent_url_stats = df_f.groupby("agent_token").agg(hits=("agent_token","count"), unique_paths=("path","nunique")).reset_index()
agent_url_stats["unique_per_hit"] = agent_url_stats["unique_paths"] / agent_url_stats["hits"]
st.markdown("**Agents with many unique pages (crawler-like)**")
st.dataframe(agent_url_stats.sort_values("unique_paths", ascending=False).head(50), use_container_width=True)

# 3) Error exposure (4xx/5xx)
st.markdown("**Bot exposure to errors (4xx/5xx)**")
err = df_f[df_f["status"].str.startswith(("4","5"), na=False)]
err_summary = err.groupby(["agent_token","status"]).size().reset_index(name="count").sort_values("count", ascending=False)
st.dataframe(err_summary, use_container_width=True)

st.markdown("---")

# ----------------------------
# EXPORT
# ----------------------------
st.subheader("Export Filtered Data")
export_cols = ["timestamp","path","status","user_agent","agent_token","is_static","client_ip"]
export_cols = [c for c in export_cols if c in df_f.columns]
df_export = df_f[export_cols].copy()
df_export["timestamp"] = safe_format_datetime(df_export.get("timestamp", pd.Series()), "%Y-%m-%d %H:%M:%S")
csv_bytes = df_export.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered CSV", csv_bytes, "filtered_bot_data.csv")

st.success("Analysis complete.")

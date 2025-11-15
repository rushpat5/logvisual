# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from utils import (
    guess_columns,
    parse_timestamp_column,
    classify_user_agent,
    detect_static_path,
    normalize_timestamp_series
)
import io

st.set_page_config(page_title="Bot & AI Log Analyzer (Revamped)", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar: upload & settings ---
st.sidebar.header("Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload CSV / TSV / TXT log file", type=["csv", "tsv", "txt"], accept_multiple_files=False)

st.sidebar.markdown("**Large file settings**")
chunk_read = st.sidebar.checkbox("Use chunked read (safer for huge files)", value=True)
sample_rows = st.sidebar.number_input("Preview rows", min_value=3, max_value=100, value=8)

if not uploaded:
    st.sidebar.info("Upload a CSV/TSV exported from your logs (detailed_hits.csv, etc.)")
    st.stop()

# --- Load file robustly ---
@st.cache_data(show_spinner=False)
def load_file_bytes(file_bytes, use_chunks=True):
    raw = file_bytes.getvalue()
    text = raw.decode("utf-8", errors="replace")
    # quick sniff for delimiter
    if "\t" in text.splitlines()[0]:
        delim = "\t"
    else:
        delim = ","
    if use_chunks:
        try:
            return pd.read_csv(io.BytesIO(raw), sep=delim, engine="python", low_memory=False)
        except Exception:
            return pd.read_csv(io.BytesIO(raw), sep=delim, engine="c", low_memory=False, on_bad_lines="skip")
    else:
        return pd.read_csv(io.BytesIO(raw), sep=delim, engine="c", low_memory=False, on_bad_lines="skip")

df = load_file_bytes(uploaded, use_chunks=chunk_read)
st.sidebar.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")

# --- Column guessing + mapping UI ---
st.sidebar.header("Column mapping (auto-suggested)")
guesses = guess_columns(list(df.columns))
# show mapping with selectboxes
def mapping_select(name, key, default):
    cols = ["(none)"] + list(df.columns)
    idx = 0
    if default and default in df.columns:
        idx = cols.index(default)
    return st.sidebar.selectbox(name, cols, index=idx, key=key)

col_time = mapping_select("Timestamp column", "map_time", guesses.get("timestamp"))
col_ua = mapping_select("User-Agent column", "map_ua", guesses.get("user_agent"))
col_path = mapping_select("Path/URL column", "map_path", guesses.get("path"))
col_status = mapping_select("Status column", "map_status", guesses.get("status"))
col_ip = mapping_select("Client IP column (optional)", "map_ip", guesses.get("ip"))
col_bytes = mapping_select("Response bytes (optional)", "map_bytes", guesses.get("bytes"))
col_section = mapping_select("Section / Site area (optional)", "map_section", guesses.get("section"))

# Provide preview
st.subheader("Preview uploaded data")
st.dataframe(df.head(sample_rows))

# --- Normalize & parse ---
working = df.copy()

# parse timestamp
if col_time != "(none)":
    working = parse_timestamp_column(working, col_time)
    if "timestamp" not in working.columns:
        working["timestamp"] = pd.to_datetime(working[col_time], errors="coerce")
else:
    # no timestamps -> create a placeholder and warn
    working["timestamp"] = pd.NaT
    st.warning("No timestamp column selected. Time-based charts will be unavailable.")

# enforce timezone normalization (naive UTC conversion)
working["timestamp"] = normalize_timestamp_series(working["timestamp"])

# coerce other columns
working["user_agent"] = working[col_ua].astype(str) if col_ua != "(none)" else ""
working["path"] = working[col_path].astype(str) if col_path != "(none)" else ""
working["status"] = working[col_status].astype(str).str.strip() if col_status != "(none)" else "unknown"
working["client_ip"] = working[col_ip].astype(str) if col_ip != "(none)" else ""
working["resp_bytes"] = pd.to_numeric(working[col_bytes], errors="coerce") if col_bytes != "(none)" else np.nan
working["section"] = working[col_section].astype(str) if col_section != "(none)" else ""

# is_static detection (allow override)
st.sidebar.header("Static asset detection")
custom_exts = st.sidebar.text_input("Extra static extensions (comma-separated)", value=".map,.webp")
static_extensions = [e.strip().lower() for e in (".css,.js,.svg,.png,.jpg,.jpeg,.gif,.ico,.woff,.woff2,.ttf,.map,.eot," + custom_exts).split(",") if e.strip()]
static_regex = st.sidebar.text_input("Static path regex (optional)", value="\\.(css|js|svg|png|jpg|jpeg|gif|ico|map|webp|woff2?)$")

working["is_static"] = working["path"].apply(lambda p: detect_static_path(p, static_extensions, static_regex))

# classify UA
def classify_row(ua):
    return classify_user_agent(ua)
working["agent_class"] = working["user_agent"].apply(classify_row)
# derive agent_group and canonical name
def agent_group_from_class(c):
    if c.startswith("ai_search_") or c.startswith("ai_user_") or c.lower().startswith("gptbot") or "openai" in c.lower():
        return "ai"
    if c.lower() in ["human"]:
        return "human"
    if "bot" in c.lower() or "crawler" in c.lower() or "spider" in c.lower() or c.startswith("bot_") or c.startswith("bot"):
        return "traditional"
    return "other"
working["agent_group"] = working["agent_class"].apply(agent_group_from_class)
working["is_bot"] = working["agent_group"] != "human"

# derive date/hour
working["date"] = working["timestamp"].dt.date
working["hour"] = working["timestamp"].dt.hour

# --- Filters UI ---
st.sidebar.header("Filters")
groups = st.sidebar.multiselect("Agent groups", options=sorted(working["agent_group"].unique()), default=["ai", "traditional"])
statuses = st.sidebar.multiselect("Status codes", options=sorted(working["status"].unique()), default=sorted(working["status"].unique())[:5])
content_scope = st.sidebar.radio("URL Scope", ["All URLs", "Content-only (exclude static)"], index=1)
date_min = working["date"].min() if working["timestamp"].notna().any() else None
date_max = working["date"].max() if working["timestamp"].notna().any() else None
if date_min and date_max:
    date_range = st.sidebar.date_input("Date range", (date_min, date_max))
else:
    date_range = None
top_n = st.sidebar.slider("Top N", 5, 100, 25)

# apply filters
df_f = working.copy()
if groups:
    df_f = df_f[df_f["agent_group"].isin(groups)]
if statuses:
    df_f = df_f[df_f["status"].isin(statuses)]
if content_scope == "Content-only (exclude static)":
    df_f = df_f[~df_f["is_static"]]
if date_range:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + timedelta(days=1)
    df_f = df_f[(df_f["timestamp"] >= start) & (df_f["timestamp"] < end)]

# --- KPIs ---
st.title("Bot & AI Traffic Dashboard — Revamped")
c1, c2, c3, c4 = st.columns(4)
total_hits = len(df_f)
bot_hits = df_f["is_bot"].sum()
ai_hits = (df_f["agent_group"] == "ai").sum()
unique_urls = df_f["path"].nunique()

c1.metric("Total hits", f"{total_hits:,}")
c2.metric("Bot hits", f"{bot_hits:,}", f"{bot_hits/total_hits*100:.1f}%" if total_hits else "0%")
c3.metric("AI hits", f"{ai_hits:,}", f"{ai_hits/total_hits*100:.1f}%" if total_hits else "0%")
c4.metric("Unique URLs", f"{unique_urls:,}")

st.markdown("---")

# --- Visuals: time series ---
if df_f["timestamp"].notna().any():
    st.subheader("Traffic over time (hourly)")
    ts = df_f.groupby([pd.Grouper(key="timestamp", freq="H"), "agent_group"]).size().reset_index(name="hits")
    fig = px.area(ts, x="timestamp", y="hits", color="agent_group", title="Hourly traffic by agent group")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No timestamps available for time-series charts.")

# status codes
st.subheader("HTTP status codes")
status_df = df_f.groupby(["status", "agent_group"]).size().reset_index(name="count")
fig2 = px.bar(status_df, x="status", y="count", color="agent_group", barmode="group", title="Status codes by agent group")
st.plotly_chart(fig2, use_container_width=True)

# top URLs
st.subheader(f"Top {top_n} URLs by hits")
url_stats = (
    df_f.groupby("path")
    .agg(hits=("path", "count"),
         first_seen=("timestamp", "min"),
         last_seen=("timestamp", "max"),
         common_status=("status", lambda x: x.mode().iloc[0] if not x.mode().empty else "")
         )
    .reset_index()
    .sort_values("hits", ascending=False)
    .head(top_n)
)
url_stats["first_seen"] = url_stats["first_seen"].dt.strftime("%Y-%m-%d %H:%M", na_rep="")
url_stats["last_seen"] = url_stats["last_seen"].dt.strftime("%Y-%m-%d %H:%M", na_rep="")
st.dataframe(url_stats, use_container_width=True)
fig_urls = px.bar(url_stats, x="hits", y="path", orientation="h", title=f"Top {top_n} URLs")
st.plotly_chart(fig_urls, use_container_width=True)

# crawl depth proxy
st.subheader("Crawl depth & unique pages per agent")
crawl_depth = df_f.groupby("agent_class")["path"].nunique().reset_index().rename(columns={"path": "unique_urls"})
st.dataframe(crawl_depth.sort_values("unique_urls", ascending=False).head(100), use_container_width=True)

# hourly heatmap
st.subheader("Hourly heatmap (agents)")
heat = df_f.groupby(["hour", "agent_group"]).size().reset_index(name="count")
heat_pivot = heat.pivot(index="hour", columns="agent_group", values="count").fillna(0)
st.dataframe(heat_pivot)
fig_heat = px.imshow(heat_pivot.T, labels=dict(x="Hour", y="Agent Group", color="Hits"), title="Hourly crawl intensity")
st.plotly_chart(fig_heat, use_container_width=True)

# --- Anomaly detection & heuristics ---
st.subheader("Heuristics & anomaly signals (basic)")
# 1) bursty IPs/agents (high request rate)
if "client_ip" in df_f.columns and df_f["client_ip"].notna().any():
    bursts = df_f.groupby("client_ip").agg(hits=("client_ip","count"),
                                           unique_paths=("path","nunique"),
                                           first_seen=("timestamp","min"),
                                           last_seen=("timestamp","max")).reset_index()
    bursts["duration_s"] = (bursts["last_seen"] - bursts["first_seen"]).dt.total_seconds().clip(lower=1)
    bursts["rps"] = bursts["hits"] / bursts["duration_s"]
    top_bursts = bursts.sort_values("rps", ascending=False).head(15)
    st.markdown("**Top fast clients (requests/sec)**")
    st.dataframe(top_bursts[["client_ip","hits","unique_paths","rps"]], use_container_width=True)
else:
    st.info("No client IP column selected; skipping IP heuristics.")

# 2) agents requesting many distinct URLs (crawler-like)
agent_url_stats = df_f.groupby("agent_class").agg(hits=("agent_class","count"),
                                                 unique_paths=("path","nunique")).reset_index()
agent_url_stats["unique_per_hit"] = agent_url_stats["unique_paths"] / agent_url_stats["hits"]
st.markdown("**Agents with many unique pages (crawler-like)**")
st.dataframe(agent_url_stats.sort_values("unique_paths", ascending=False).head(50), use_container_width=True)

# 3) repeated 4xx/5xx exposure to crawlers (important for SEO)
st.markdown("**Bot exposure to errors (4xx/5xx)**")
err = df_f[df_f["status"].str.startswith(("4","5"), na=False)]
err_summary = err.groupby(["agent_group","status"]).size().reset_index(name="count").sort_values("count", ascending=False)
st.dataframe(err_summary, use_container_width=True)

st.markdown("---")

# --- Export options ---
st.subheader("Export filtered data")
export_cols = ["timestamp","path","status","user_agent","agent_class","agent_group","is_static","client_ip","resp_bytes"]
export_cols = [c for c in export_cols if c in df_f.columns]
to_export = df_f[export_cols].copy()
to_export["timestamp"] = to_export["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S", na_rep="")

col1, col2, col3 = st.columns(3)
csv_bytes = to_export.to_csv(index=False).encode("utf-8")
parquet_bytes = None
try:
    parquet_buf = io.BytesIO()
    to_export.to_parquet(parquet_buf, index=False)
    parquet_bytes = parquet_buf.getvalue()
except Exception:
    parquet_bytes = None

with col1:
    st.download_button("Download CSV", csv_bytes, file_name="filtered_logs.csv", mime="text/csv")
with col2:
    if parquet_bytes:
        st.download_button("Download Parquet", parquet_bytes, file_name="filtered_logs.parquet", mime="application/octet-stream")
    else:
        st.button("Parquet not available (install pyarrow)")
with col3:
    st.download_button("Download JSON", to_export.to_json(orient="records").encode("utf-8"), file_name="filtered_logs.json", mime="application/json")

st.success("Analysis complete.")

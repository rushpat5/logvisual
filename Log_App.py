# app.py
# SEO Log Analyzer — full app with improved KPI panel (sparklines + percent-deltas)
# Single-file Streamlit app. No column-mapping UI.
# Requirements: streamlit, pandas, numpy, plotly

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import re
from datetime import timedelta

st.set_page_config(page_title="SEO Log Analyzer", layout="wide")
st.title("SEO Log Analyzer — Bot & Crawl Insights")

# ----------------------------
# Curated UA tokens (~60)
# ----------------------------
UA_TOKENS = sorted(set([
    "gptbot","chatgpt-user","oai-searchbot","openai","perplexity","perplexitybot",
    "claude","claudebot","anthropic","mistral","bytespider","ccbot","serpapi",
    "copilot","bingbot","googlebot","yandex","duckduckbot","baiduspider","slurp",
    "ahrefsbot","semrushbot","bingpreview","sogou","applebot","facebookexternalhit",
    "linkedinbot","duckduckgo","majestic","ia_archiver","facebot",
    "rogerbot","seznambot","rambler","sistrix","mj12bot","dotbot","surge",
    "screamingfrog","python-requests","curl","wget","httpclient",
    "phantomjs","headless","scrapy","crawler","spider","bot","robot","sitebot",
    "yeti","yisou","uptimebot","uptime-kuma","uptimerobot"
]))

BROWSER_TOKENS = ["chrome","firefox","safari","edge","msie","trident","opera","opr","mozilla"]

# ----------------------------
# Utilities: robust read + guess
# ----------------------------
def robust_read(uploaded_file):
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = ""
    first_line = text.splitlines()[0] if text else ""
    delim = "\t" if "\t" in first_line else ","
    try:
        return pd.read_csv(io.BytesIO(raw), sep=delim, engine="c", low_memory=False)
    except Exception:
        try:
            return pd.read_csv(io.BytesIO(raw), sep=delim, engine="python", on_bad_lines="skip", low_memory=False)
        except Exception:
            return pd.read_fwf(io.BytesIO(raw))

def guess_columns(columns):
    def find_any(keywords):
        for kw in keywords:
            for c in columns:
                if kw in c:
                    return c
        return None
    out = {}
    out["timestamp"] = find_any(["time_parsed","time","timestamp","date","datetime","ts"])
    out["path"]      = find_any(["pathclean","path","uri","request","url"])
    out["status"]    = find_any(["status","statuscode","status_code","statusclass"])
    out["ip"]        = find_any(["clientip","ip","remote_addr"])
    out["ua"]        = find_any(["user-agent","user_agent","useragent","ua"])
    out["bytes"]     = find_any(["bytes","size","response_size","content_length"])
    out["section"]   = find_any(["section"])
    return out

# ----------------------------
# Hardened timestamp parsing
# ----------------------------
def parse_timestamp_column(df, col):
    if col not in df.columns:
        df["timestamp"] = pd.NaT
        return df

    s = df[col].astype(str)
    df["_tmp_ts"] = pd.NaT

    with np.errstate(invalid='ignore'):
        num = pd.to_numeric(s, errors="coerce")

    if num.notna().any():
        # ms
        if (num > 1e12).any():
            df["_tmp_ts"] = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
        elif (num > 1e9).any():
            df["_tmp_ts"] = pd.to_datetime(num, unit="s", errors="coerce", utc=True)

    remaining = df["_tmp_ts"].isna()
    if remaining.any():
        try:
            parsed = pd.to_datetime(s[remaining], errors="coerce", utc=True, infer_datetime_format=True)
        except Exception:
            parsed = pd.to_datetime(s[remaining], errors="coerce", utc=True)
        df.loc[remaining, "_tmp_ts"] = parsed

    if df["_tmp_ts"].isna().mean() > 0.9:
        fmts = ("%d/%b/%Y:%H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S")
        for fmt in fmts:
            try:
                parsed = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
                if parsed.notna().any():
                    df["_tmp_ts"] = parsed
                    break
            except Exception:
                pass

    df["_tmp_ts"] = pd.to_datetime(df["_tmp_ts"], errors="coerce", utc=True)

    # convert to tz-naive UTC
    try:
        df["timestamp"] = df["_tmp_ts"].dt.tz_convert(None)
    except Exception:
        df["timestamp"] = pd.to_datetime(df["_tmp_ts"], errors="coerce").dt.tz_localize(None)

    df.drop(columns=["_tmp_ts"], inplace=True, errors="ignore")
    return df

# ----------------------------
# UA token extraction
# ----------------------------
def extract_agent_token(ua):
    if not isinstance(ua, str) or ua.strip() == "":
        return "unknown"
    s = ua.lower()
    for t in UA_TOKENS:
        if t in s:
            return t
    for b in BROWSER_TOKENS:
        if b in s:
            if b == "opr":
                return "opera"
            if b in ("msie", "trident"):
                return "ie"
            return b
    if re.search(r"(bot|spider|crawler|scraper|wget|curl|python-requests|headless|phantomjs)", s):
        return "bot"
    tok = s.split()[0].split("/")[0]
    if 0 < len(tok) < 40:
        return tok
    return "other"

# ----------------------------
# Sessionization helper
# ----------------------------
def sessionize(df, ip_col, agent_col, timeout_minutes=30):
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    last = {}
    sids = []
    counter = 0
    for idx, row in df_sorted.iterrows():
        ts = row["timestamp"]
        if pd.isna(ts):
            sids.append(f"nosess-{idx}")
            continue
        key = (row.get(ip_col, ""), row.get(agent_col, ""))
        if key not in last:
            counter += 1
            sid = f"s{counter}"
            last[key] = ts
            sids.append(sid)
        else:
            diff = (ts - last[key]).total_seconds() / 60.0
            if diff > timeout_minutes:
                counter += 1
                sid = f"s{counter}"
                last[key] = ts
                sids.append(sid)
            else:
                last[key] = ts
                sids.append(f"s{counter}")
    # map back in original order
    sids_series = pd.Series(sids, index=df_sorted.index)
    sids_series.index = df_sorted.index
    # create result aligned to original df order
    result = sids_series.sort_index()
    return result.values.tolist()

# ----------------------------
# Minimal sparkline helper
# ----------------------------
def tiny_sparkline(series):
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(width=2)))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=60)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

# ----------------------------
# Uploader (no mapping UI)
# ----------------------------
st.sidebar.info("Upload CSV/TSV. Auto-detection only; no manual mapping UI.")
uploaded = st.sidebar.file_uploader("Upload CSV/TSV", type=["csv","tsv","txt"])
sitemap_file = st.sidebar.file_uploader("Optional: sitemap.xml", type=["xml","txt"])
if not uploaded:
    st.info("Upload a CSV/TSV file to begin analysis.")
    st.stop()

df = robust_read(uploaded)

# ----------------------------
# Auto-detect columns & prepare data
# ----------------------------
cols_lower = [c.lower() for c in df.columns]
guesses = guess_columns(cols_lower)

def pick_column(df, guess_key):
    c = guesses.get(guess_key)
    if not c:
        return None
    # find original column case-insensitively
    for orig in df.columns:
        if orig.lower() == c:
            return orig
    return None

ts_col = pick_column(df, "timestamp")
if ts_col:
    df = parse_timestamp_column(df, ts_col)
else:
    df["timestamp"] = pd.NaT

path_col = pick_column(df, "path")
if path_col:
    df["_raw_path"] = df[path_col].astype(str)
else:
    candidate = None
    for c in df.columns:
        sample = df[c].astype(str).head(50).tolist()
        if any("/" in (x or "") for x in sample):
            candidate = c
            break
    df["_raw_path"] = df[candidate].astype(str) if candidate else df.iloc[:, 0].astype(str)

def canonicalize(p):
    if not isinstance(p, str):
        return ""
    p = p.strip()
    p = re.sub(r"^https?://[^/]+", "", p)
    return p if p else "/"

df["_canon"] = df["_raw_path"].apply(canonicalize)
def split_pq(p):
    if "?" in p:
        a,b = p.split("?",1); return a,b
    return p,""
df["_path_only"], df["_qs"] = zip(*df["_canon"].apply(split_pq))
df["_qs_count"] = pd.Series([0 if q=="" else len([x for x in re.split("[&;]", q) if x]) for q in df["_qs"]])

status_col = pick_column(df, "status")
df["_status"] = df[status_col].astype(str) if status_col else df.get("Status", "").astype(str)

ip_col = pick_column(df, "ip")
df["_ip"] = df[ip_col].astype(str) if ip_col else df.get("ClientIP", "").astype(str) if "ClientIP" in df.columns else ""

ua_col = pick_column(df, "ua")
df["_ua"] = df[ua_col].astype(str) if ua_col else df.get("User-Agent", "").astype(str) if "User-Agent" in df.columns else ""

bytes_col = pick_column(df, "bytes")
df["_bytes"] = pd.to_numeric(df[bytes_col], errors="coerce") if bytes_col else pd.to_numeric(df.get("Bytes", pd.Series([np.nan]*len(df))), errors="coerce")

section_col = pick_column(df, "section")
df["_section"] = df[section_col].astype(str) if section_col else df.get("Section", "").astype(str) if "Section" in df.columns else ""

# static detection
def detect_static(p):
    if not isinstance(p, str):
        return False
    p = p.split("?")[0].lower()
    exts = [".css",".js",".svg",".png",".jpg",".jpeg",".gif",".ico",".woff",".woff2",".ttf",".map",".eot",".webp"]
    return any(p.endswith(e) for e in exts)

df["_is_static"] = df.get("IsStatic", pd.NA)
if df["_is_static"].isna().all():
    df["_is_static"] = df["_path_only"].apply(detect_static)

# derived fields
df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
df["hour"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour

# agent & session
df["_agent"] = df["_ua"].apply(extract_agent_token)
# sessionize: returns list aligned to sorted df; we want aligned to original index so apply over sorted and map back
df_sorted_idx = df.sort_values("timestamp").index
session_ids = sessionize(df.loc[df_sorted_idx].reset_index(drop=True), "_ip", "_agent", timeout_minutes=30)
# place session ids into df in sorted order
df.loc[df_sorted_idx, "_session"] = session_ids

# optional sitemap paths
sitemap_paths = set()
if sitemap_file:
    try:
        txt = sitemap_file.getvalue().decode("utf-8", errors="ignore")
        locs = re.findall(r"<loc>(.*?)</loc>", txt, flags=re.IGNORECASE)
        sitemap_paths = {re.sub(r"^https?://[^/]+", "", u).split("?")[0] for u in locs}
    except Exception:
        sitemap_paths = set()

# ----------------------------
# Filters (safe date handling)
# ----------------------------
st.sidebar.header("Filters")
valid_ts_mask = df["timestamp"].notna()
if valid_ts_mask.any():
    min_d = df.loc[valid_ts_mask, "date"].min()
    max_d = df.loc[valid_ts_mask, "date"].max()
    date_range = st.sidebar.date_input("Date range", (min_d, max_d))
    if isinstance(date_range, tuple):
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    else:
        start = pd.Timestamp(date_range)
        end = start + pd.Timedelta(days=1)
else:
    start = None
    end = None

agents = sorted(df["_agent"].unique())
sel_agents = st.sidebar.multiselect("Agent tokens (blank=all)", agents, default=[])
scope = st.sidebar.selectbox("Scope", ["All URLs", "Content-only (exclude static)"], index=1)
top_n = st.sidebar.slider("Top N", 5, 50, 25)

df_f = df.copy()
if start is not None:
    df_f = df_f[(df_f["timestamp"] >= start) & (df_f["timestamp"] < end)]
if sel_agents:
    df_f = df_f[df_f["_agent"].isin(sel_agents)]
if scope == "Content-only (exclude static)":
    df_f = df_f[~df_f["_is_static"]]

# ----------------------------
# Improved KPI panel
# ----------------------------
# helper functions for KPI panel
def safe_date_bounds_local(df_local):
    if "timestamp" not in df_local.columns:
        return None, None
    valid = df_local["timestamp"].dropna()
    if valid.empty:
        return None, None
    return valid.min(), valid.max()

def previous_period_bounds(a, b):
    dur = b - a
    prev_b = a
    prev_a = a - dur
    return prev_a, prev_b

def aggregate_series_count(df_local, freq="D"):
    if df_local.empty or "timestamp" not in df_local.columns:
        return pd.Series([], dtype=float)
    s = df_local.groupby(pd.Grouper(key="timestamp", freq=freq)).size()
    # fill missing
    try:
        s = s.asfreq(freq, fill_value=0).sort_index()
    except Exception:
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq=freq), fill_value=0) if not s.empty else s
    return s

def pct_change_val(cur, prev):
    if prev in (None, 0, np.nan) or pd.isna(prev):
        return None
    try:
        return (cur - prev) / prev * 100.0
    except Exception:
        return None

# compute base metrics
total_hits = len(df_f)
bot_hits = int(df_f["_agent"].isin(UA_TOKENS).sum()) if "_agent" in df_f.columns else 0
static_hits = int(df_f["_is_static"].sum()) if "_is_static" in df_f.columns else 0
unique_urls = int(df_f["_path_only"].nunique()) if "_path_only" in df_f.columns else 0
unique_agents = int(df_f["_agent"].nunique()) if "_agent" in df_f.columns else 0

bot_pct = (bot_hits / total_hits * 100.0) if total_hits else 0.0
static_pct = (static_hits / total_hits * 100.0) if total_hits else 0.0

# compute trends & deltas vs previous same-length period if timestamps exist in original df
min_ts, max_ts = safe_date_bounds_local(df_f)
delta_info = {}
if min_ts and max_ts and min_ts < max_ts:
    prev_start, prev_end = previous_period_bounds(min_ts, max_ts)
    span_days = (max_ts - min_ts).days + 1
    freq = "H" if span_days <= 2 else "D"

    # total series (current filtered df aggregated)
    ts_total = aggregate_series_count(df_f, freq=freq)
    # for previous period counts, filter original df (not already filtered by agent selection)
    prev_mask = (df["timestamp"] >= prev_start) & (df["timestamp"] < prev_end) if "timestamp" in df.columns else pd.Series([False]*len(df))
    cur_mask  = (df["timestamp"] >= min_ts) & (df["timestamp"] <= max_ts) if "timestamp" in df.columns else pd.Series([False]*len(df))
    prev_df = df[prev_mask] if "timestamp" in df.columns else pd.DataFrame()
    cur_df = df[cur_mask] if "timestamp" in df.columns else pd.DataFrame()

    prev_total = len(prev_df) if not prev_df.empty else None
    cur_total = len(cur_df) if not cur_df.empty else None
    delta_info["total_hits"] = (total_hits, pct_change_val(cur_total, prev_total), ts_total)

    # bot hits
    if "_agent" in df.columns:
        ts_bot = aggregate_series_count(df_f[df_f["_agent"].isin(UA_TOKENS)], freq=freq)
        prev_bot_count = len(prev_df[prev_df["_agent"].isin(UA_TOKENS)]) if not prev_df.empty and "_agent" in prev_df.columns else None
        cur_bot_count = len(cur_df[cur_df["_agent"].isin(UA_TOKENS)]) if not cur_df.empty and "_agent" in cur_df.columns else None
        delta_info["bot_hits"] = (bot_hits, pct_change_val(cur_bot_count, prev_bot_count), ts_bot)
    else:
        delta_info["bot_hits"] = (bot_hits, None, None)

    # static hits
    if "_is_static" in df.columns:
        ts_static = aggregate_series_count(df_f[df_f["_is_static"]], freq=freq)
        prev_static_count = len(prev_df[prev_df["_is_static"]]) if not prev_df.empty and "_is_static" in prev_df.columns else None
        cur_static_count = len(cur_df[cur_df["_is_static"]]) if not cur_df.empty and "_is_static" in cur_df.columns else None
        delta_info["static_hits"] = (static_hits, pct_change_val(cur_static_count, prev_static_count), ts_static)
    else:
        delta_info["static_hits"] = (static_hits, None, None)

    # unique urls trend (daily new)
    if "_path_only" in df_f.columns and "timestamp" in df_f.columns:
        df_sorted = df_f.sort_values("timestamp")
        df_sorted["_is_new"] = ~df_sorted["_path_only"].duplicated()
        ts_new = df_sorted.groupby(pd.Grouper(key="timestamp", freq=freq))["_is_new"].sum().asfreq(freq, fill_value=0)
        delta_info["unique_urls"] = (unique_urls, None, ts_new)
    else:
        delta_info["unique_urls"] = (unique_urls, None, None)

    # unique agents trend
    if "_agent" in df_f.columns:
        ts_agents = df_f.groupby(pd.Grouper(key="timestamp", freq=freq))["_agent"].nunique().asfreq(freq, fill_value=0)
        delta_info["unique_agents"] = (unique_agents, None, ts_agents)
    else:
        delta_info["unique_agents"] = (unique_agents, None, None)
else:
    delta_info["total_hits"] = (total_hits, None, None)
    delta_info["bot_hits"] = (bot_hits, None, None)
    delta_info["static_hits"] = (static_hits, None, None)
    delta_info["unique_urls"] = (unique_urls, None, None)
    delta_info["unique_agents"] = (unique_agents, None, None)

# render KPIs
kpi_cols = st.columns(5)
kpi_defs = [
    ("Total hits", "total_hits", f"{total_hits:,}", ""),
    ("Bot hits", "bot_hits", f"{bot_hits:,}", f"{bot_pct:.1f}% of total"),
    ("Static hits", "static_hits", f"{static_hits:,}", f"{static_pct:.1f}% of total"),
    ("Unique URLs", "unique_urls", f"{unique_urls:,}", ""),
    ("Unique agents", "unique_agents", f"{unique_agents:,}", "")
]

for col_obj, (label, key, val_str, subtitle) in zip(kpi_cols, kpi_defs):
    cur_val, pct_delta, ts = delta_info.get(key, (None, None, None))
    with col_obj:
        delta_display = f"{pct_delta:+.1f}%" if pct_delta is not None else None
        if delta_display:
            st.metric(label, val_str, delta=delta_display)
        else:
            st.metric(label, val_str)
        if subtitle:
            st.caption(subtitle)
        if ts is not None and not ts.empty:
            # limit points for performance
            max_pts = 60
            ts_plot = ts
            if len(ts) > max_pts:
                # resample to reduce points
                try:
                    step = int(len(ts) / max_pts) or 1
                    ts_plot = ts.resample(ts.index.freq * step).sum()
                except Exception:
                    ts_plot = ts.iloc[-max_pts:]
            try:
                fig = tiny_sparkline(ts_plot)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.line_chart(ts_plot)
        else:
            st.write(" ")

st.markdown("---")

# ----------------------------
# Rest of SEO metrics (examples kept compact)
# ----------------------------
# Top Crawled Paths
st.subheader("Top Crawled Paths")
top_paths = (df_f.groupby("_path_only")
             .agg(hits=("_path_only","count"),
                  bot_hits=("_agent", lambda s: s.isin(UA_TOKENS).sum()),
                  unique_agents=("_agent","nunique"),
                  first_seen=("timestamp","min"),
                  last_seen=("timestamp","max"))
             .reset_index().sort_values("hits", ascending=False).head(top_n))
top_paths["first_seen"] = pd.to_datetime(top_paths["first_seen"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
top_paths["last_seen"]  = pd.to_datetime(top_paths["last_seen"],  errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
st.dataframe(top_paths, use_container_width=True)
fig_paths = px.bar(top_paths.sort_values("hits"), x="hits", y="_path_only", orientation="h", title="Top crawled paths")
st.plotly_chart(fig_paths, use_container_width=True)

# Bot exposure to errors
st.subheader("Bot exposure to 4xx/5xx")
err = df_f[df_f["_status"].astype(str).str.startswith(("4","5"), na=False)]
err_summary = err.groupby(["_agent","_status"]).size().reset_index(name="count").sort_values("count", ascending=False)
st.dataframe(err_summary.head(200), use_container_width=True)
if not err_summary.empty:
    fig_err = px.bar(err_summary.head(50), x="count", y="_agent", color="_status", orientation="h", title="Errors by agent")
    st.plotly_chart(fig_err, use_container_width=True)

# Orphaned URLs
st.subheader("Orphaned URLs (bot-only)")
human_seen = set(df_f[~df_f["_agent"].isin(UA_TOKENS) & (df_f["_agent"]!="bot")]["_path_only"])
bot_seen = set(df_f[df_f["_agent"].isin(UA_TOKENS) | (df_f["_agent"]=="bot")]["_path_only"])
orphaned = sorted(list(bot_seen - human_seen))
st.write(f"Orphaned count: {len(orphaned)} — showing top {top_n}")
st.dataframe(pd.DataFrame({"path": orphaned[:top_n]}), use_container_width=True)

# Duplicate content risk
st.subheader("Duplicate-content risk (same path, many distinct query strings)")
dup_qs = (df_f.groupby("_path_only")
          .agg(hits=("_path_only","count"), distinct_qs=("_qs","nunique"), distinct_agents=("_agent","nunique"))
          .reset_index().sort_values(["distinct_qs","hits"], ascending=False))
st.dataframe(dup_qs.head(top_n), use_container_width=True)

# Distinct URL discovery trend
st.subheader("Distinct URL discovery trend (new URLs)")
if df_f["timestamp"].notna().any():
    df_f_sorted = df_f.sort_values("timestamp")
    df_f_sorted["_is_new"] = ~df_f_sorted["_path_only"].duplicated()
    new_urls = df_f_sorted.groupby(pd.Grouper(key="timestamp", freq="D"))["_is_new"].sum().reset_index().rename(columns={"_is_new":"new_urls"})
    st.dataframe(new_urls.tail(60), use_container_width=True)
    fig_new = px.line(new_urls, x="timestamp", y="new_urls", title="New distinct URLs per day")
    st.plotly_chart(fig_new, use_container_width=True)
else:
    st.info("Timestamps not available; discovery trend skipped.")

# Export
st.markdown("---")
st.header("Export & sample")
st.dataframe(df_f[["_path_only","_status","_agent","_ip","timestamp"]].head(200), use_container_width=True)
csv_data = df_f.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", csv_data, file_name="filtered_logs.csv")

st.success("Analysis complete.")

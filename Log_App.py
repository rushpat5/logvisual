# app.py
# SEO Log Analyzer — improved KPI visuals for client delivery
# Single-file Streamlit app. Requirements: streamlit, pandas, numpy, plotly

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
# Utilities
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
    # columns should be lower-cased form of df.columns
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

# Hardened timestamp parsing
def parse_timestamp_column(df, col):
    if not col or col not in df.columns:
        df["timestamp"] = pd.NaT
        return df
    s = df[col].astype(str)
    df["_tmp_ts"] = pd.NaT
    with np.errstate(invalid='ignore'):
        num = pd.to_numeric(s, errors="coerce")
    # epoch heuristics
    if num.notna().any():
        if (num > 1e12).any():
            df["_tmp_ts"] = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
        elif (num > 1e9).any():
            df["_tmp_ts"] = pd.to_datetime(num, unit="s", errors="coerce", utc=True)
    # parse remaining as strings
    rem = df["_tmp_ts"].isna()
    if rem.any():
        try:
            parsed = pd.to_datetime(s[rem], errors="coerce", utc=True, infer_datetime_format=True)
        except Exception:
            parsed = pd.to_datetime(s[rem], errors="coerce", utc=True)
        df.loc[rem, "_tmp_ts"] = parsed
    # fallback formats if mostly NaT
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
    # coerce and remove tz
    df["_tmp_ts"] = pd.to_datetime(df["_tmp_ts"], errors="coerce", utc=True)
    try:
        df["timestamp"] = df["_tmp_ts"].dt.tz_convert(None)
    except Exception:
        df["timestamp"] = pd.to_datetime(df["_tmp_ts"], errors="coerce").dt.tz_localize(None)
    df.drop(columns=["_tmp_ts"], inplace=True, errors="ignore")
    return df

# UA token extraction
def extract_agent_token(ua):
    if not isinstance(ua, str) or ua.strip()=="":
        return "unknown"
    s = ua.lower()
    for t in UA_TOKENS:
        if t in s:
            return t
    for b in BROWSER_TOKENS:
        if b in s:
            if b == "opr": return "opera"
            if b in ("msie","trident"): return "ie"
            return b
    if re.search(r"(bot|spider|crawler|scraper|wget|curl|python-requests|headless|phantomjs)", s):
        return "bot"
    tok = s.split()[0].split("/")[0]
    if 0 < len(tok) < 40:
        return tok
    return "other"

# Sessionization
def sessionize_sorted(df_sorted, ip_col, agent_col, timeout_minutes=30):
    last = {}
    sids = []
    counter = 0
    for idx, row in df_sorted.iterrows():
        ts = row["timestamp"]
        if pd.isna(ts):
            sids.append(f"nosess-{idx}")
            continue
        key = (row.get(ip_col,""), row.get(agent_col,""))
        if key not in last:
            counter += 1
            sid = f"s{counter}"
            last[key] = ts
            sids.append(sid)
        else:
            diff = (ts - last[key]).total_seconds()/60.0
            if diff > timeout_minutes:
                counter += 1
                sid = f"s{counter}"
                last[key] = ts
                sids.append(sid)
            else:
                last[key] = ts
                sids.append(sid)
    return sids

# Visual helpers
def tiny_sparkline(series, color="#4DA6FF"):
    # series: pd.Series indexed by datetime
    if series.empty:
        fig = go.Figure()
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=64)
        return fig
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(width=2, color=color), hoverinfo="skip"))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=64, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False, visible=False)
    return fig

# ----------------------------
# Uploader (no mapping UI)
# ----------------------------
st.sidebar.info("Upload CSV/TSV log file. App auto-detects columns.")
uploaded = st.sidebar.file_uploader("Upload CSV/TSV", type=["csv","tsv","txt"])
sitemap_file = st.sidebar.file_uploader("Optional: sitemap.xml", type=["xml","txt"])
if not uploaded:
    st.info("Upload a CSV/TSV file to begin.")
    st.stop()

df = robust_read(uploaded)

# ----------------------------
# Auto-detect columns and prepare
# ----------------------------
cols_lower = [c.lower() for c in df.columns]
guesses = guess_columns(cols_lower)

def pick_column(df, guess_key):
    target = guesses.get(guess_key)
    if not target:
        return None
    # case-insensitive match
    for c in df.columns:
        if c.lower() == target:
            return c
    return None

# timestamp parsing
ts_col = pick_column(df, "timestamp")
if ts_col:
    df = parse_timestamp_column(df, ts_col)
else:
    df["timestamp"] = pd.NaT

# path selection
path_col = pick_column(df, "path")
if path_col:
    df["_raw_path"] = df[path_col].astype(str)
else:
    # fallback heuristics: column with many "/" values
    candidate = None
    for c in df.columns:
        sample = df[c].astype(str).head(50).tolist()
        if any("/" in (x or "") for x in sample):
            candidate = c
            break
    df["_raw_path"] = df[candidate].astype(str) if candidate else df.iloc[:,0].astype(str)

def canonicalize(path):
    if not isinstance(path, str):
        return ""
    p = path.strip()
    p = re.sub(r"^https?://[^/]+", "", p)
    return p if p else "/"

df["_canon"] = df["_raw_path"].apply(canonicalize)
def split_path_qs(p):
    if "?" in p:
        a,b = p.split("?",1); return a,b
    return p,""
df["_path_only"], df["_qs"] = zip(*df["_canon"].apply(split_path_qs))
df["_qs_count"] = pd.Series([0 if q=="" else len([x for x in re.split("[&;]", q) if x]) for q in df["_qs"]])

# status, ip, ua, bytes, section
status_col = pick_column(df, "status"); df["_status"] = df[status_col].astype(str) if status_col else df.get("Status","").astype(str)
ip_col = pick_column(df, "ip"); df["_ip"] = df[ip_col].astype(str) if ip_col else df.get("ClientIP","").astype(str) if "ClientIP" in df.columns else ""
ua_col = pick_column(df, "ua"); df["_ua"] = df[ua_col].astype(str) if ua_col else df.get("User-Agent","").astype(str) if "User-Agent" in df.columns else ""
bytes_col = pick_column(df, "bytes"); df["_bytes"] = pd.to_numeric(df[bytes_col], errors="coerce") if bytes_col else pd.to_numeric(df.get("Bytes", pd.Series([np.nan]*len(df))), errors="coerce")
section_col = pick_column(df, "section"); df["_section"] = df[section_col].astype(str) if section_col else df.get("Section","").astype(str) if "Section" in df.columns else ""

# static detect
def detect_static_file(path):
    if not isinstance(path, str) or path.strip()=="":
        return False
    p = path.split("?")[0].lower()
    exts = [".css",".js",".svg",".png",".jpg",".jpeg",".gif",".ico",".woff",".woff2",".ttf",".map",".eot",".webp"]
    return any(p.endswith(e) for e in exts)

df["_is_static"] = df.get("IsStatic", pd.NA)
if df["_is_static"].isna().all():
    df["_is_static"] = df["_path_only"].apply(detect_static_file)

# derived fields
df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
df["hour"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour

# agent token & sessions
df["_agent"] = df["_ua"].apply(extract_agent_token)

# sessionize: do on sorted copy and map results back
df_sorted = df.sort_values("timestamp").reset_index()
sids = sessionize_sorted(df_sorted, "_ip", "_agent", timeout_minutes=30)
df.loc[df_sorted["index"], "_session"] = sids

# optional sitemap
sitemap_paths = set()
if sitemap_file:
    try:
        txt = sitemap_file.getvalue().decode("utf-8", errors="ignore")
        locs = re.findall(r"<loc>(.*?)</loc>", txt, flags=re.IGNORECASE)
        sitemap_paths = {re.sub(r"^https?://[^/]+", "", u).split("?")[0] for u in locs}
    except Exception:
        sitemap_paths = set()

# ----------------------------
# Filters
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
sel_agents = st.sidebar.multiselect("Agent tokens (blank = all)", agents, default=[])
scope = st.sidebar.selectbox("Scope", ["All URLs", "Content-only (exclude static)"], index=1)
top_n = st.sidebar.slider("Top N (lists)", 5, 50, 25)

df_f = df.copy()
if start is not None:
    df_f = df_f[(df_f["timestamp"] >= start) & (df_f["timestamp"] < end)]
if sel_agents:
    df_f = df_f[df_f["_agent"].isin(sel_agents)]
if scope == "Content-only (exclude static)":
    df_f = df_f[~df_f["_is_static"]]

# ----------------------------
# KPI logic (clean & client-facing)
# ----------------------------
def safe_bounds(df_local):
    if "timestamp" not in df_local.columns:
        return None, None
    s = df_local["timestamp"].dropna()
    if s.empty:
        return None, None
    return s.min(), s.max()

def prev_period(a, b):
    dur = b - a
    return a - dur, a

def aggregate_counts(df_local, freq="D"):
    if df_local.empty or "timestamp" not in df_local.columns:
        return pd.Series(dtype=float)
    s = df_local.groupby(pd.Grouper(key="timestamp", freq=freq)).size()
    # fill missing range
    try:
        s = s.asfreq(freq, fill_value=0).sort_index()
    except Exception:
        if not s.empty:
            full = pd.date_range(s.index.min(), s.index.max(), freq=freq)
            s = s.reindex(full, fill_value=0)
    return s

def compute_pct_delta(cur, prev):
    if prev in (None, 0, np.nan) or pd.isna(prev):
        return None
    try:
        return (cur - prev) / prev * 100.0
    except Exception:
        return None

# Base numbers
total_hits = len(df_f)
bot_hits = int(df_f["_agent"].isin(UA_TOKENS).sum()) if "_agent" in df_f.columns else 0
static_hits = int(df_f["_is_static"].sum()) if "_is_static" in df_f.columns else 0
unique_urls = int(df_f["_path_only"].nunique()) if "_path_only" in df_f.columns else 0
unique_agents = int(df_f["_agent"].nunique()) if "_agent" in df_f.columns else 0

bot_ratio = (bot_hits / total_hits * 100.0) if total_hits else 0.0
static_ratio = (static_hits / total_hits * 100.0) if total_hits else 0.0

min_ts, max_ts = safe_bounds(df_f)
delta_map = {}

if min_ts and max_ts and min_ts < max_ts:
    prev_start, prev_end = prev_period(min_ts, max_ts)
    span_days = (max_ts - min_ts).days + 1
    freq = "H" if span_days <= 2 else "D"

    # total: current vs previous (whole dataset baseline)
    ts_total = aggregate_counts(df_f, freq=freq)
    prev_mask = (df["timestamp"] >= prev_start) & (df["timestamp"] < prev_end) if "timestamp" in df.columns else pd.Series([False]*len(df))
    cur_mask  = (df["timestamp"] >= min_ts) & (df["timestamp"] < max_ts) if "timestamp" in df.columns else pd.Series([False]*len(df))
    prev_df = df[prev_mask] if "timestamp" in df.columns else pd.DataFrame()
    cur_df = df[cur_mask] if "timestamp" in df.columns else pd.DataFrame()
    prev_total = len(prev_df) if not prev_df.empty else None
    cur_total  = len(cur_df) if not cur_df.empty else None
    delta_map["total_hits"] = (total_hits, compute_pct_delta(cur_total, prev_total), ts_total)

    # bot hits
    ts_bot = aggregate_counts(df_f[df_f["_agent"].isin(UA_TOKENS)], freq=freq) if "_agent" in df_f.columns else pd.Series(dtype=float)
    prev_bot_count = len(prev_df[prev_df["_agent"].isin(UA_TOKENS)]) if not prev_df.empty and "_agent" in prev_df.columns else None
    cur_bot_count  = len(cur_df[cur_df["_agent"].isin(UA_TOKENS)]) if not cur_df.empty and "_agent" in cur_df.columns else None
    delta_map["bot_hits"] = (bot_hits, compute_pct_delta(cur_bot_count, prev_bot_count), ts_bot)

    # static hits
    ts_static = aggregate_counts(df_f[df_f["_is_static"]], freq=freq) if "_is_static" in df_f.columns else pd.Series(dtype=float)
    prev_static_count = len(prev_df[prev_df["_is_static"]]) if not prev_df.empty and "_is_static" in prev_df.columns else None
    cur_static_count  = len(cur_df[cur_df["_is_static"]]) if not cur_df.empty and "_is_static" in cur_df.columns else None
    delta_map["static_hits"] = (static_hits, compute_pct_delta(cur_static_count, prev_static_count), ts_static)

    # unique urls trend
    if "_path_only" in df_f.columns:
        df_sorted2 = df_f.sort_values("timestamp")
        df_sorted2["_is_new"] = ~df_sorted2["_path_only"].duplicated()
        ts_new = df_sorted2.groupby(pd.Grouper(key="timestamp", freq=freq))["_is_new"].sum().asfreq(freq, fill_value=0)
        delta_map["unique_urls"] = (unique_urls, None, ts_new)
    else:
        delta_map["unique_urls"] = (unique_urls, None, pd.Series(dtype=float))

    # unique agents trend
    if "_agent" in df_f.columns:
        ts_agents = df_f.groupby(pd.Grouper(key="timestamp", freq=freq))["_agent"].nunique().asfreq(freq, fill_value=0)
        delta_map["unique_agents"] = (unique_agents, None, ts_agents)
    else:
        delta_map["unique_agents"] = (unique_agents, None, pd.Series(dtype=float))
else:
    delta_map["total_hits"] = (total_hits, None, pd.Series(dtype=float))
    delta_map["bot_hits"] = (bot_hits, None, pd.Series(dtype=float))
    delta_map["static_hits"] = (static_hits, None, pd.Series(dtype=float))
    delta_map["unique_urls"] = (unique_urls, None, pd.Series(dtype=float))
    delta_map["unique_agents"] = (unique_agents, None, pd.Series(dtype=float))

# Render clean KPI cards
def render_kpi(col, title, key, value_str, context=None, positive_is_good=False):
    # key used to read delta_map
    val, pct_delta, ts = delta_map.get(key, (None, None, None))
    # format delta display and color logic
    delta_label = None
    if pct_delta is None:
        delta_label = None
    else:
        # show sign and percent
        delta_label = f"{pct_delta:+.1f}%"
    # show metric
    if delta_label:
        # use st.metric delta (it colors automatically if streamlit supports it)
        col.metric(title, value_str, delta=delta_label)
    else:
        col.metric(title, value_str)
    if context:
        col.caption(context)
    # sparkline figure
    if isinstance(ts, pd.Series) and not ts.empty:
        # downsample for performance
        max_pts = 60
        ts_plot = ts
        if len(ts_plot) > max_pts:
            try:
                step = int(len(ts_plot)/max_pts) or 1
                # aggregate by step using rolling/resample may be tricky; sample last max_pts
                ts_plot = ts_plot.iloc[-max_pts:]
            except Exception:
                ts_plot = ts_plot.iloc[-max_pts:]
        fig = tiny_sparkline(ts_plot)
        col.plotly_chart(fig, use_container_width=True)
    else:
        # placeholder spacing for visual alignment
        col.write(" ")

kcols = st.columns(5)
render_kpi(kcols[0], "Total hits", "total_hits", f"{total_hits:,}", context=None)
render_kpi(kcols[1], "Bot hits", "bot_hits", f"{bot_hits:,}", context=f"{bot_ratio:.1f}% of total")
render_kpi(kcols[2], "Static hits", "static_hits", f"{static_hits:,}", context=f"{static_ratio:.1f}% of total")
render_kpi(kcols[3], "Unique URLs", "unique_urls", f"{unique_urls:,}")
render_kpi(kcols[4], "Unique agents", "unique_agents", f"{unique_agents:,}")

st.markdown("---")

# ----------------------------
# Other SEO metric sections — kept succinct but presentable
# ----------------------------
st.subheader("Top Crawled Paths")
top_paths = (df_f.groupby("_path_only")
             .agg(hits=("_path_only","count"),
                  bot_hits=("_agent", lambda s: s.isin(UA_TOKENS).sum()),
                  unique_agents=("_agent","nunique"),
                  first_seen=("timestamp","min"),
                  last_seen=("timestamp","max"))
             .reset_index()
             .sort_values("hits", ascending=False)
             .head(top_n))
top_paths["first_seen"] = pd.to_datetime(top_paths["first_seen"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
top_paths["last_seen"]  = pd.to_datetime(top_paths["last_seen"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
st.dataframe(top_paths, use_container_width=True)
if not top_paths.empty:
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
st.subheader("Orphaned URLs (seen only by bots)")
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
st.subheader("Distinct URL discovery trend")
if df_f["timestamp"].notna().any():
    df_f_sorted = df_f.sort_values("timestamp")
    df_f_sorted["_is_new"] = ~df_f_sorted["_path_only"].duplicated()
    new_urls = df_f_sorted.groupby(pd.Grouper(key="timestamp", freq="D"))["_is_new"].sum().reset_index().rename(columns={"_is_new":"new_urls"})
    st.dataframe(new_urls.tail(60), use_container_width=True)
    fig_new = px.line(new_urls, x="timestamp", y="new_urls", title="New distinct URLs per day")
    st.plotly_chart(fig_new, use_container_width=True)
else:
    st.info("Timestamps unavailable; discovery trend skipped.")

# Exports
st.markdown("---")
st.header("Export & sample")
st.dataframe(df_f[["_path_only","_status","_agent","_ip","timestamp"]].head(200), use_container_width=True)
csv_data = df_f.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", csv_data, file_name="filtered_logs.csv")

st.success("Analysis complete.")

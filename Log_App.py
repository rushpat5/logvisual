# app.py
# SEO-focused Log Analyzer (Streamlit)
# Single-file app:
# - No left-side column-mapping UI (only uploader)
# - Curated ~60 user-agent tokens
# - Implements 19 SEO metrics (enabled)
# - Auto-detects columns with heuristics
# - Clean Plotly visuals and data tables
# Requirements: streamlit, pandas, numpy, plotly, python-dateutil, pyarrow (optional for parquet export)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import re
from datetime import timedelta

st.set_page_config(page_title="SEO Log Analyzer — Bot & Crawl Insights", layout="wide", initial_sidebar_state="expanded")
st.title("SEO Log Analyzer — Bot & Crawl Insights")

# ---------------------------
# Curated UA tokens (~60) — Option 2
# ---------------------------
UA_TOKENS = [
    "gptbot","chatgpt-user","oai-searchbot","openai","perplexity","perplexitybot",
    "claude","claudebot","anthropic","mistral","bytespider","ccbot","serpapi",
    "copilot","bingbot","googlebot","yandex","duckduckbot","baiduspider","slurp",
    "ahrefsbot","semrushbot","bingpreview","sogou","applebot","facebookexternalhit",
    "linkedinbot","duckduckgo","bingpreview","majestic","ia_archiver","facebot",
    "iesnare","rogerbot","seznambot","rambler","sistrix","mj12bot","dotbot","surge",
    "screaming frog","screamingfrog","python-requests","curl","wget","httpclient",
    "phantomjs","headless","scrapy","crawler","spider","bot","robot","sitebot",
    "bingbot","baiduspider","yeti","yisou","yandexbot","semrushbot","ahrefsbot-proxy",
    "uptimebot","uptime-kuma"
]
# ensure uniqueness, lowercase
UA_TOKENS = sorted({t.lower() for t in UA_TOKENS})

# ---------------------------
# Helpers: column guessing & parsing
# ---------------------------
def guess_columns(columns):
    """Heuristic mapping for common log exports. Returns dict of guessed names or None."""
    low = {c.lower(): c for c in columns}
    def find_any(keywords):
        for kw in keywords:
            for c in columns:
                if kw in c.lower():
                    return c
        return None
    out = {}
    out["timestamp"] = find_any(["time_parsed","time","timestamp","date","datetime","ts","time_parsed"])
    out["path"] = find_any(["pathclean","path","uri","request","url"])
    out["raw_path"] = find_any(["path","uri","request","url"])
    out["status"] = find_any(["status","statuscode","response","statusclass","status_code"])
    out["client_ip"] = find_any(["clientip","ip","remote_addr","remote"])
    out["user_agent"] = find_any(["user-agent","user_agent","useragent","ua","agent"])
    out["bytes"] = find_any(["bytes","size","response_size","content_length"])
    out["referer"] = find_any(["referer","referrer","refer"])
    out["query_params"] = find_any(["queryparams","query","qs","query_string","params"])
    out["section"] = find_any(["section","site_section","area"])
    out["session"] = find_any(["sessionid","session","sid"])
    out["response_time"] = find_any(["response_time","latency","time_taken","request_time"])
    return out

def robust_read(file):
    """Attempt to read CSV/TSV robustly."""
    raw = file.getvalue()
    text_sample = raw.decode("utf-8", errors="replace").splitlines()
    first = text_sample[0] if text_sample else ""
    delim = "\t" if "\t" in first else ","
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="c", low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="python", on_bad_lines="skip", low_memory=False)
        except Exception:
            # fallback to one-column load
            df = pd.read_fwf(io.BytesIO(raw))
    return df

def parse_timestamp_column(df, col):
    """Robust timestamp parsing: epoch ms/s, ISO, common formats. Produces tz-naive UTC."""
    s = df[col]
    # attempt numeric epoch
    try:
        num = pd.to_numeric(s, errors="coerce")
    except Exception:
        num = pd.Series([np.nan]*len(s))
    df["_ts_tmp"] = pd.NaT
    if num.notna().any():
        if (num > 1e12).any():
            df["_ts_tmp"] = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
        elif (num > 1e9).any():
            df["_ts_tmp"] = pd.to_datetime(num, unit="s", errors="coerce", utc=True)
    # parse strings for remaining
    rem = df["_ts_tmp"].isna()
    if rem.any():
        try:
            parsed = pd.to_datetime(s[rem], errors="coerce", utc=True, infer_datetime_format=True)
        except Exception:
            parsed = pd.to_datetime(s[rem], errors="coerce", utc=True)
        df.loc[rem, "_ts_tmp"] = parsed
    # final fallback parse common formats
    if df["_ts_tmp"].isna().all():
        for fmt in ("%d/%b/%Y:%H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M"):
            try:
                df["_ts_tmp"] = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
                if df["_ts_tmp"].notna().any():
                    break
            except Exception:
                pass
    # convert to tz-naive (UTC)
    df["timestamp"] = df["_ts_tmp"].dt.tz_convert(None)
    df.drop(columns=["_ts_tmp"], inplace=True, errors="ignore")
    return df

# ---------------------------
# UA token extraction
# ---------------------------
BROWSER_TOKENS = ["chrome","firefox","safari","edge","msie","trident","opera","opr","mozilla"]
def extract_agent_token(ua):
    if not isinstance(ua, str) or ua.strip()=="":
        return "unknown"
    s = ua.lower()
    # check curated tokens first
    for t in UA_TOKENS:
        if t in s:
            return t
    # browser heuristics
    for b in BROWSER_TOKENS:
        if b in s:
            if b in ("msie","trident"):
                return "ie"
            if b=="opr":
                return "opera"
            return b
    # generic bot heuristics
    if re.search(r"(bot|spider|crawler|scraper|wget|curl|python-requests|headless|phantomjs)", s):
        return "bot"
    # fallback: first token before / or space
    tok = s.split()[0].split("/")[0]
    if len(tok)>0 and len(tok)<40:
        return tok
    return "other"

# ---------------------------
# Analysis functions (SEO-focused)
# ---------------------------
def canonicalize_path(raw_path):
    """Return path without domain; keep path+query optional. Handles typical 'https://domain/path?qs' inputs."""
    if not isinstance(raw_path, str):
        return ""
    p = raw_path.strip()
    # remove scheme+domain
    if p.startswith("http://") or p.startswith("https://"):
        # remove protocol and domain
        try:
            p = re.sub(r"^https?://[^/]+", "", p)
        except Exception:
            pass
    # ensure begins with /
    if p=="":
        return "/"
    return p

def split_path_and_query(path):
    if not isinstance(path, str):
        return (path or "", "")
    if "?" in path:
        a,b = path.split("?",1)
        return (a, b)
    return (path, "")

def count_query_params(qs):
    if not qs:
        return 0
    # parse key=value pairs heuristically
    pairs = [p for p in re.split("[&;]", qs) if p.strip()!=""]
    return len(pairs)

# sessionize by client_ip + agent_token (30m inactivity)
def sessionize(df, ip_col="client_ip", agent_col="agent_token", timeout_minutes=30):
    # require timestamp sorted
    df = df.sort_values("timestamp").reset_index(drop=True)
    session_ids = []
    last = {}
    sid_counter = 0
    for idx, row in df.iterrows():
        key = (row.get(ip_col,""), row.get(agent_col,""))
        ts = row["timestamp"]
        if pd.isna(ts):
            sid = f"nosess-{idx}"
            session_ids.append(sid)
            continue
        if key not in last:
            sid_counter += 1
            sid = f"s{sid_counter}"
            last[key] = (ts, sid)
            session_ids.append(sid)
            continue
        last_ts, last_sid = last[key]
        diff = (ts - last_ts).total_seconds()/60.0
        if diff > timeout_minutes:
            sid_counter += 1
            sid = f"s{sid_counter}"
            last[key] = (ts, sid)
            session_ids.append(sid)
        else:
            # continue same session
            last[key] = (ts, last_sid)
            session_ids.append(last_sid)
    return session_ids

# ---------------------------
# UI: uploader only (no mapping fields)
# ---------------------------
st.sidebar.info("Upload server log CSV/TSV. App will auto-detect columns. No column mapping UI shown per settings.")
uploaded = st.sidebar.file_uploader("Upload CSV/TSV", type=["csv","tsv","txt"], help="Upload CSV or TSV exported from your logs. Auto-detection used.")
sitemap_file = st.sidebar.file_uploader("Optional: sitemap.xml (plain text)", type=["xml","txt"], help="Optional sitemap to compare coverage", key="sitemap")

if not uploaded:
    st.info("Upload a CSV/TSV to begin analysis.")
    st.stop()

df = robust_read(uploaded)

# ---------------------------
# Auto-detect columns and prepare
# ---------------------------
guesses = guess_columns(df.columns.tolist())
# if timestamp not found, try any likely candidate
if not guesses.get("timestamp"):
    # fallback: common header variations
    for candidate in ["Time_parsed","Time","time_parsed","time","timestamp","Date"]:
        if candidate in df.columns:
            guesses["timestamp"] = candidate
            break

# require timestamp presence to enable time-based metrics; still conduct others when missing
if guesses.get("timestamp"):
    df = parse_timestamp_column(df, guesses["timestamp"])
else:
    df["timestamp"] = pd.NaT

# canonicalize path column: prefer PathClean, Path, raw path columns
path_col = guesses.get("path") or guesses.get("raw_path")
if path_col:
    df["_raw_path"] = df[path_col].astype(str)
else:
    # try a column named 'Path' or 'Request'
    fallback = None
    for c in df.columns:
        if re.search(r"path|request|uri|url", c, re.IGNORECASE):
            fallback = c
            break
    df["_raw_path"] = df[fallback].astype(str) if fallback else df.iloc[:,0].astype(str)

# split into canonical path and query params
df["_canon_path"] = df["_raw_path"].apply(lambda p: canonicalize_path(p))
df["_path_only"], df["_qs"] = zip(*df["_canon_path"].apply(split_path_and_query))
df["_qs_count"] = df["_qs"].apply(count_query_params)

# status
status_col = guesses.get("status")
df["_status"] = df[status_col].astype(str) if status_col and status_col in df.columns else df.get("Status", "").astype(str)

# client ip
ip_col_guess = guesses.get("client_ip")
df["_client_ip"] = df[ip_col_guess].astype(str) if ip_col_guess and ip_col_guess in df.columns else df.get("ClientIP", "").astype(str) if "ClientIP" in df.columns else ""

# user-agent
ua_col = guesses.get("user_agent")
df["_user_agent"] = df[ua_col].astype(str) if ua_col and ua_col in df.columns else df.get("User-Agent", "").astype(str) if "User-Agent" in df.columns else ""

# bytes / response size
bytes_col = guesses.get("bytes")
df["_bytes"] = pd.to_numeric(df[bytes_col], errors="coerce") if bytes_col and bytes_col in df.columns else pd.to_numeric(df.get("Bytes", pd.Series([np.nan]*len(df))), errors="coerce")

# section
section_col = guesses.get("section")
df["_section"] = df[section_col].astype(str) if section_col and section_col in df.columns else df.get("Section", "").astype(str) if "Section" in df.columns else ""

# session id if provided
session_col = guesses.get("session")
df["_session_id_orig"] = df[session_col].astype(str) if session_col and session_col in df.columns else df.get("SessionID","").astype(str) if "SessionID" in df.columns else ""

# response time if provided
rt_col = guesses.get("response_time")
df["_resp_time"] = pd.to_numeric(df[rt_col], errors="coerce") if rt_col and rt_col in df.columns else None

# agent token
df["_agent_token"] = df["_user_agent"].apply(extract_agent_token)

# is_static heuristic: prefer IsStatic column if present, else compute from extension
isstatic_col = None
for c in df.columns:
    if c.lower() in ("isstatic","is_static","file","is_static_flag"):
        isstatic_col = c
        break
if isstatic_col:
    df["_is_static"] = df[isstatic_col].astype(str).str.upper().isin(["TRUE","1","YES"])
else:
    def detect_static_from_path(p):
        if not isinstance(p, str) or p.strip()=="":
            return False
        p2 = p.split("?")[0].lower()
        exts = [".css",".js",".svg",".png",".jpg",".jpeg",".gif",".ico",".woff",".woff2",".ttf",".map",".eot",".webp"]
        return any(p2.endswith(e) for e in exts)
    df["_is_static"] = df["_path_only"].apply(detect_static_from_path)

# basic derived
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour

# create sessionized ids (using client_ip + agent_token)
df = df.sort_values("timestamp").reset_index(drop=True)
df["_sessionizer_key"] = list(zip(df["_client_ip"].fillna(""), df["_agent_token"].fillna("")))
df["_session_id"] = sessionize(df, ip_col="_client_ip", agent_col="_agent_token", timeout_minutes=30)
# ensure types
df["_qs_count"] = df["_qs_count"].fillna(0).astype(int)
df["_bytes"] = pd.to_numeric(df["_bytes"], errors="coerce")

# ---------------------------
# Optional sitemap parse (extract URLs) for metric 13
# ---------------------------
sitemap_urls = set()
if sitemap_file:
    try:
        txt = sitemap_file.getvalue().decode("utf-8", errors="ignore")
        # rough extract <loc> tags
        sitemap_urls = set(re.findall(r"<loc>(.*?)</loc>", txt, flags=re.IGNORECASE))
        sitemap_paths = {canonicalize_path(u).split("?")[0] for u in sitemap_urls}
    except Exception:
        sitemap_urls = set()
        sitemap_paths = set()
else:
    sitemap_paths = set()

# ---------------------------
# Filters (minimal): date range, select agents optional
# ---------------------------
st.sidebar.header("Filters")
min_date = df["date"].min() if df["date"].notna().any() else None
max_date = df["date"].max() if df["date"].notna().any() else None
if min_date and max_date:
    date_range = st.sidebar.date_input("Date range", (min_date, max_date))
else:
    date_range = None

unique_agents = sorted(df["_agent_token"].value_counts().index.tolist())
selected_agents = st.sidebar.multiselect("Agent tokens (filter; leave blank = all)", options=unique_agents, default=[])

content_scope = st.sidebar.selectbox("Scope", ["All URLs", "Content-only (exclude static assets)"], index=1)
top_n = st.sidebar.slider("Top N (lists)", 5, 100, 25)

# apply filters
df_f = df.copy()
if date_range:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    df_f = df_f[(df_f["timestamp"]>=start) & (df_f["timestamp"]<end)]
if selected_agents:
    df_f = df_f[df_f["_agent_token"].isin(selected_agents)]
if content_scope=="Content-only (exclude static assets)":
    df_f = df_f[~df_f["_is_static"]]

# ---------------------------
# Metrics (19 items)
# All enabled per user choice
# Output layout: top-level KPIs, then detailed sections
# ---------------------------
st.header("Key SEO KPIs")

col1, col2, col3, col4, col5 = st.columns(5)
total_hits = len(df_f)
bot_hits = df_f["_agent_token"].isin(UA_TOKENS).sum()
static_hits = df_f["_is_static"].sum()
unique_urls = df_f["_path_only"].nunique()
unique_agents_count = df_f["_agent_token"].nunique()

col1.metric("Total hits", f"{total_hits:,}")
col2.metric("Bot-token hits", f"{bot_hits:,}", f"{bot_hits/total_hits*100:.1f}%" if total_hits else "0%")
col3.metric("Static asset hits", f"{static_hits:,}", f"{static_hits/total_hits*100:.1f}%" if total_hits else "0%")
col4.metric("Unique URLs", f"{unique_urls:,}")
col5.metric("Unique agent tokens", f"{unique_agents_count:,}")

st.markdown("---")

# 1) Top Crawled Paths
st.subheader("1 — Top Crawled Paths")
top_paths = (df_f.groupby("_path_only")
             .agg(hits=("_path_only","count"),
                  bot_hits=("_agent_token", lambda s: s.isin(UA_TOKENS).sum()),
                  unique_agents=("_agent_token","nunique"),
                  first_seen=("timestamp","min"),
                  last_seen=("timestamp","max"))
             .reset_index().sort_values("hits", ascending=False).head(top_n))
top_paths["first_seen"] = top_paths["first_seen"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
top_paths["last_seen"] = top_paths["last_seen"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
st.dataframe(top_paths, use_container_width=True)

fig = px.bar(top_paths.sort_values("hits"), x="hits", y="_path_only", orientation="h", title="Top crawled paths")
st.plotly_chart(fig, use_container_width=True)

# 2) Crawled Sections
st.subheader("2 — Crawled Sections")
if df_f["_section"].notna().any() and df_f["_section"].str.strip().replace("","NaN").notna().any():
    sec = df_f.groupby("_section").agg(hits=("_section","count")).reset_index().sort_values("hits", ascending=False)
    st.dataframe(sec.head(top_n))
    fig = px.pie(sec.head(15), names="_section", values="hits", title="Top sections by hits")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No section data available in log to compute Crawled Sections.")

# 3) Crawl Frequency by User-Agent token
st.subheader("3 — Crawl Frequency by Agent Token (Top)")
freq_agt = df_f.groupby(["_agent_token"]).agg(hits=("_agent_token","count")).reset_index().sort_values("hits", ascending=False)
st.dataframe(freq_agt.head(50), use_container_width=True)
fig = px.bar(freq_agt.head(25), x="hits", y="_agent_token", orientation="h", title="Top agent tokens by hits")
st.plotly_chart(fig, use_container_width=True)

# 4) Crawl Budget Waste (static hits, tracking params)
st.subheader("4 — Crawl Budget Waste")
# static hits ratio and hits with tracking-like params (e.g., utm_, gclid, sa=U, ved, usg)
tracking_pattern = re.compile(r"(utm_|gclid=|fbclid=|_ga=|_gl=|_fbp=|sa=|ved=|usg=)", re.IGNORECASE)
df_f["_has_tracking_params"] = df_f["_qs"].astype(str).apply(lambda q: bool(tracking_pattern.search(q)))
waste_stats = {
    "static_hits": int(df_f["_is_static"].sum()),
    "tracking_param_hits": int(df_f["_has_tracking_params"].sum()),
    "distinct_tracked_urls": int(df_f[df_f["_has_tracking_params"]]["_path_only"].nunique())
}
st.write(waste_stats)
fig = px.bar(x=["static_hits","tracking_param_hits"], y=[waste_stats["static_hits"], waste_stats["tracking_param_hits"]],
             labels={"x":"type","y":"count"}, title="Crawl budget waste (static vs tracking-param hits)")
st.plotly_chart(fig, use_container_width=True)

# 5) Crawl Depth (unique path segments per agent)
st.subheader("5 — Crawl Depth Approx (unique pages crawled per agent token)")
crawl_depth = (df_f.groupby("_agent_token")["_path_only"].nunique().reset_index().rename(columns={"_path_only":"unique_urls"}).sort_values("unique_urls", ascending=False))
st.dataframe(crawl_depth.head(100), use_container_width=True)

# 6) Query-Parameter Abuse List (many params)
st.subheader("6 — Query Parameter Abuse (high param counts)")
param_abuse = df_f[df_f["_qs_count"]>=3].groupby("_path_only").agg(hits=("_path_only","count"), avg_params=("_qs_count","mean")).reset_index().sort_values("hits", ascending=False)
st.dataframe(param_abuse.head(top_n), use_container_width=True)

# 7) Bot Exposure to Error Codes (4xx/5xx)
st.subheader("7 — Bot exposure to errors (4xx / 5xx)")
err_df = df_f[df_f["_status"].str.startswith(("4","5"), na=False)]
err_by_agent = err_df.groupby(["_agent_token","_status"]).size().reset_index(name="count").sort_values("count", ascending=False)
st.dataframe(err_by_agent.head(200), use_container_width=True)
fig = px.bar(err_by_agent.head(50), x="count", y="_agent_token", color="_status", orientation="h", title="Errors by agent token")
st.plotly_chart(fig, use_container_width=True)

# 8) Orphaned URLs Indicator (bot-only URLs)
st.subheader("8 — Orphaned URLs (URLs seen only by bots, not humans)")
# define human as agent token not in UA_TOKENS and not generic 'bot'
human_mask = ~df_f["_agent_token"].isin(UA_TOKENS) & (df_f["_agent_token"]!="bot")
human_seen = set(df_f[human_mask]["_path_only"].unique())
bot_seen = set(df_f[df_f["_agent_token"].isin(UA_TOKENS) | (df_f["_agent_token"]=="bot")]["_path_only"].unique())
orphaned = sorted(list(bot_seen - human_seen))
st.write(f"Orphaned URL count: {len(orphaned)} — showing top {top_n}")
st.dataframe(pd.DataFrame({"path":orphaned[:top_n]}), use_container_width=True)

# 9) Duplicate Content Hits (same pathname different query)
st.subheader("9 — Duplicate-content risk: same path, many distinct query strings")
dup_qs = (df_f.groupby("_path_only")
          .agg(hits=("_path_only","count"),
               distinct_qs=("_qs","nunique"),
               distinct_agents=("_agent_token","nunique"))
          .reset_index().sort_values(["distinct_qs","hits"], ascending=False))
st.dataframe(dup_qs.head(top_n), use_container_width=True)

# 10) Slow Response Pages (if response time exists)
st.subheader("10 — Slow Response Pages")
if "_resp_time" in df_f.columns and df_f["_resp_time"].notna().any():
    slow = df_f.groupby("_path_only")["_resp_time"].agg(["mean","median","max","count"]).reset_index().sort_values("mean", ascending=False)
    st.dataframe(slow.head(top_n), use_container_width=True)
else:
    st.info("No response-time column detected. Skipping slow response metric.")

# 11) Response Bytes / Load Impact
st.subheader("11 — Response Bytes (size) impact")
bytes_stats = df_f.groupby("_path_only")["_bytes"].agg(["sum","mean","count"]).reset_index().sort_values("sum", ascending=False)
st.dataframe(bytes_stats.head(top_n), use_container_width=True)
fig = px.bar(bytes_stats.head(25), x="sum", y="_path_only", orientation="h", title="Top URLs by total bytes served")
st.plotly_chart(fig, use_container_width=True)

# 12) Sessionized Bot Journeys (samples)
st.subheader("12 — Sessionized Bot Journeys (sample)")
# show sample sessions for top bot token
top_bot = df_f[df_f["_agent_token"].isin(UA_TOKENS)]["_agent_token"].value_counts().idxmax() if not df_f[df_f["_agent_token"].isin(UA_TOKENS)].empty else None
if top_bot:
    sample_sessions = df_f[df_f["_agent_token"]==top_bot].groupby("_session_id").filter(lambda g: len(g)>=3).groupby("_session_id").head(1)["_session_id"].unique()[:10]
    sessions_list = []
    for sid in sample_sessions:
        srows = df_f[df_f["_session_id"]==sid].sort_values("timestamp")[["_session_id","timestamp","_path_only","_status","_agent_token"]]
        sessions_list.append(srows)
    for s in sessions_list:
        st.dataframe(s, use_container_width=True)
else:
    st.info("No bot sessions available for sampling.")

# 13) Sitemap Coverage vs Bot Hits (optional)
st.subheader("13 — Sitemap coverage vs hits (optional)")
if sitemap_paths:
    # compute coverage: which sitemap paths got hits
    hit_paths = set(df_f["_path_only"].unique())
    sitemap_in_hits = sorted(list(sitemap_paths & hit_paths))
    sitemap_not_hit = sorted(list(sitemap_paths - hit_paths))
    st.write(f"Sitemap URLs total: {len(sitemap_paths)}. In logs: {len(sitemap_in_hits)}. Missing from logs: {len(sitemap_not_hit)}")
    st.dataframe(pd.DataFrame({"sitemap_hit_sample": sitemap_in_hits[:top_n]}))
else:
    st.info("No sitemap uploaded. Skip sitemap coverage metric.")

# 14) Parameter Pattern Clustering (basic)
st.subheader("14 — Parameter pattern clustering (count of unique param keys top)")
def param_keys(qs):
    if not qs:
        return ""
    pairs = [p for p in re.split("[&;]", qs) if p]
    keys = [p.split("=")[0] if "=" in p else p for p in pairs]
    keys = [k for k in keys if k]
    keys_sorted = ",".join(sorted(set(keys)))
    return keys_sorted
df_f["_param_keys"] = df_f["_qs"].astype(str).apply(param_keys)
param_pattern = df_f.groupby("_param_keys").agg(hits=("_param_keys","count")).reset_index().sort_values("hits", ascending=False)
st.dataframe(param_pattern.head(50), use_container_width=True)

# 15) Device/Platform Parsing (mobile/desktop heuristics)
st.subheader("15 — Device / Platform (mobile vs desktop estimates)")
def detect_mobile(ua):
    if not isinstance(ua, str):
        return False
    u = ua.lower()
    return bool(re.search(r"mobile|android|iphone|ipad", u))
df_f["_is_mobile_est"] = df_f["_user_agent"].apply(detect_mobile)
mobile_share = df_f["_is_mobile_est"].mean()*100 if len(df_f)>0 else 0
st.metric("Estimated mobile share", f"{mobile_share:.1f}%")
mob = df_f.groupby("_is_mobile_est").size().reset_index(name="count")
fig = px.pie(mob, names="_is_mobile_est", values="count", title="Estimated Mobile vs Desktop (True=mobile)")
st.plotly_chart(fig, use_container_width=True)

# 16) Section Abuse (sections crawled excessively)
st.subheader("16 — Section abuse (top sections by hits)")
if df_f["_section"].notna().any() and df_f["_section"].str.strip().replace("","NaN").notna().any():
    sec_abuse = df_f.groupby("_section").agg(hits=("_section","count"), unique_urls=("_path_only","nunique")).reset_index().sort_values("hits", ascending=False)
    st.dataframe(sec_abuse.head(top_n), use_container_width=True)
else:
    st.info("Section metadata not available.")

# 17) Heavy Hitters (IPs or agents requesting thousands+ pages)
st.subheader("17 — Heavy hitters (IPs and agent tokens)")
top_ips = df_f.groupby("_client_ip").size().reset_index(name="hits").sort_values("hits", ascending=False).head(25)
st.dataframe(top_ips, use_container_width=True)
top_agents = df_f.groupby("_agent_token").size().reset_index(name="hits").sort_values("hits", ascending=False).head(50)
st.dataframe(top_agents, use_container_width=True)

# 18) Status-Class Distribution (2xx/3xx/4xx/5xx)
st.subheader("18 — Status-class distribution")
def status_class(s):
    if not isinstance(s, str) or s=="":
        return "other"
    s = s.strip()
    if s.startswith("2"):
        return "2xx"
    if s.startswith("3"):
        return "3xx"
    if s.startswith("4"):
        return "4xx"
    if s.startswith("5"):
        return "5xx"
    return "other"
df_f["_status_class"] = df_f["_status"].apply(status_class)
status_summary = df_f.groupby("_status_class").size().reset_index(name="count")
st.dataframe(status_summary, use_container_width=True)
fig = px.pie(status_summary, names="_status_class", values="count", title="Status class distribution")
st.plotly_chart(fig, use_container_width=True)

# 19) Distinct URL Discovery Trend (new URLs over time)
st.subheader("19 — Distinct URL discovery trend")
if df_f["timestamp"].notna().any():
    df_f = df_f.sort_values("timestamp")
    df_f["_is_new_url"] = ~df_f["_path_only"].duplicated()
    new_urls = df_f.groupby(pd.Grouper(key="timestamp", freq="D"))["_is_new_url"].sum().reset_index().rename(columns={"_is_new_url":"new_urls"})
    st.dataframe(new_urls.tail(50), use_container_width=True)
    fig = px.line(new_urls, x="timestamp", y="new_urls", title="New distinct URLs discovered per day")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Timestamps not available; cannot compute discovery trend.")

st.markdown("---")

# Exports and utilities
st.header("Exports & Data")
exp_col1, exp_col2, exp_col3 = st.columns([2,1,1])
with exp_col1:
    st.subheader("Filtered dataset sample")
    st.dataframe(df_f[["_path_only","_status","_agent_token","_client_ip","timestamp"]].head(200), use_container_width=True)
with exp_col2:
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, file_name="filtered_logs.csv")
with exp_col3:
    try:
        import pyarrow as pa  # optional
        buf = io.BytesIO()
        df_f.to_parquet(buf, index=False)
        st.download_button("Download Parquet", buf.getvalue(), file_name="filtered_logs.parquet")
    except Exception:
        st.info("Install pyarrow for Parquet export (optional).")

st.success("Analysis complete.")

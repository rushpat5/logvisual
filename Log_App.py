# app.py
# SEO Log Analyzer — corrected timestamp parsing
# Single-file Streamlit app. No left-side column-mapping UI.
# Curated UA tokens (~60). SEO-focused metrics enabled.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import re
from datetime import timedelta

st.set_page_config(page_title="SEO Log Analyzer", layout="wide")
st.title("SEO Log Analyzer — Bot & Crawl Insights (fixed timestamp parsing)")

# ---------------------------
# Curated UA tokens (~60)
# ---------------------------
UA_TOKENS = sorted(set([
    "gptbot","chatgpt-user","oai-searchbot","openai","perplexity","perplexitybot",
    "claude","claudebot","anthropic","mistral","bytespider","ccbot","serpapi",
    "copilot","bingbot","googlebot","yandex","duckduckbot","baiduspider","slurp",
    "ahrefsbot","semrushbot","bingpreview","sogou","applebot","facebookexternalhit",
    "linkedinbot","duckduckgo","majestic","ia_archiver","facebot","iesnare","rogerbot",
    "seznambot","rambler","sistrix","mj12bot","dotbot","surge","screaming frog",
    "screamingfrog","python-requests","curl","wget","httpclient","phantomjs","headless",
    "scrapy","crawler","spider","bot","robot","sitebot","yeti","yisou","yandexbot",
    "ahrefsbot-proxy","uptimebot","uptime-kuma"
]))

BROWSER_TOKENS = ["chrome","firefox","safari","edge","msie","trident","opera","opr","mozilla"]

# ---------------------------
# Helpers: robust read, guess columns
# ---------------------------
def robust_read(uploaded_file):
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = ""
    first_line = text.splitlines()[0] if text else ""
    delim = "\t" if "\t" in first_line else ","
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="c", low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="python", on_bad_lines="skip", low_memory=False)
        except Exception:
            df = pd.read_fwf(io.BytesIO(raw))
    return df

def guess_columns(columns):
    def find_any(keywords):
        for kw in keywords:
            for c in columns:
                if kw in c.lower():
                    return c
        return None
    out = {}
    cols = [c for c in columns]
    out["timestamp"] = find_any(["time_parsed","time","timestamp","date","datetime","ts"])
    out["path"] = find_any(["pathclean","path","uri","request","url"])
    out["status"] = find_any(["status","statuscode","response","statusclass","status_code"])
    out["client_ip"] = find_any(["clientip","ip","remote_addr","remote"])
    out["user_agent"] = find_any(["user-agent","user_agent","useragent","ua","agent"])
    out["bytes"] = find_any(["bytes","size","response_size","content_length"])
    out["section"] = find_any(["section","site_section","area"])
    out["session"] = find_any(["sessionid","session","sid"])
    return out

# ---------------------------
# Robust timestamp parsing (FIXED)
# ---------------------------
def parse_timestamp_column(df, col):
    """
    Robust timestamp parsing:
    - Attempt numeric epoch detection (ms / s)
    - Parse strings (infer)
    - Ensure final timestamp column is tz-naive (UTC) datetimelike
    """
    s = df[col].astype(str) if col in df.columns else pd.Series([None]*len(df))
    # Prepare container
    df["_ts_tmp"] = pd.NaT

    # numeric detection
    with np.errstate(invalid='ignore'):
        num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        # ms?
        if (num > 1e12).any():
            try:
                df["_ts_tmp"] = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
            except Exception:
                df["_ts_tmp"] = pd.to_datetime(num, errors="coerce", utc=True)
        elif (num > 1e9).any():
            try:
                df["_ts_tmp"] = pd.to_datetime(num, unit="s", errors="coerce", utc=True)
            except Exception:
                df["_ts_tmp"] = pd.to_datetime(num, errors="coerce", utc=True)

    # parse remaining strings
    remaining = df["_ts_tmp"].isna()
    if remaining.any():
        try:
            parsed = pd.to_datetime(s[remaining], errors="coerce", utc=True, infer_datetime_format=True)
        except Exception:
            parsed = pd.to_datetime(s[remaining], errors="coerce", utc=True)
        df.loc[remaining, "_ts_tmp"] = parsed

    # fallback with common formats if still mostly NaT
    if df["_ts_tmp"].isna().mean() > 0.9:
        for fmt in ("%d/%b/%Y:%H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S"):
            try:
                parsed = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
                if parsed.notna().any():
                    df["_ts_tmp"] = parsed
                    break
            except Exception:
                pass

    # Finally, ensure dtype is datetime64 with timezone awareness, then drop tz
    # Coerce regardless to ensure datetimelike dtype
    df["_ts_tmp"] = pd.to_datetime(df["_ts_tmp"], errors="coerce", utc=True)
    # Convert to tz-naive (UTC)
    try:
        df["timestamp"] = df["_ts_tmp"].dt.tz_convert(None)
    except Exception:
        # If .dt accessor fails (shouldn't after to_datetime), do fallback: remove tzinfo manually
        tmp = df["_ts_tmp"]
        df["timestamp"] = pd.to_datetime(tmp, errors="coerce", utc=True).dt.tz_convert(None)
    df.drop(columns=["_ts_tmp"], inplace=True, errors="ignore")
    return df

# ---------------------------
# Agent token extraction
# ---------------------------
def extract_agent_token(ua):
    if not isinstance(ua, str) or ua.strip()=="":
        return "unknown"
    s = ua.lower()
    for t in UA_TOKENS:
        if t in s:
            return t
    for b in BROWSER_TOKENS:
        if b in s:
            if b in ("msie","trident"):
                return "ie"
            if b == "opr":
                return "opera"
            return b
    if re.search(r"(bot|spider|crawler|scraper|wget|curl|python-requests|headless|phantomjs)", s):
        return "bot"
    tok = s.split()[0].split("/")[0]
    if 0 < len(tok) < 40:
        return tok
    return "other"

# ---------------------------
# Sessionize
# ---------------------------
def sessionize(df, ip_col="_client_ip", agent_col="_agent_token", timeout_minutes=30):
    df = df.sort_values("timestamp").reset_index(drop=True)
    last_seen = {}
    sids = []
    counter = 0
    for idx, row in df.iterrows():
        key = (row.get(ip_col, ""), row.get(agent_col, ""))
        ts = row["timestamp"]
        if pd.isna(ts):
            sids.append(f"nosess-{idx}")
            continue
        if key not in last_seen:
            counter += 1
            sid = f"s{counter}"
            last_seen[key] = (ts, sid)
            sids.append(sid)
        else:
            last_ts, sid = last_seen[key]
            diff_min = (ts - last_ts).total_seconds() / 60.0
            if diff_min > timeout_minutes:
                counter += 1
                sid = f"s{counter}"
                last_seen[key] = (ts, sid)
                sids.append(sid)
            else:
                last_seen[key] = (ts, sid)
                sids.append(sid)
    return sids

# ---------------------------
# Uploader (no mapping UI)
# ---------------------------
st.sidebar.info("Upload your CSV/TSV. The app auto-detects columns (no mapping UI).")
uploaded = st.sidebar.file_uploader("Upload CSV/TSV", type=["csv","tsv","txt"])
sitemap_file = st.sidebar.file_uploader("Optional: sitemap.xml", type=["xml","txt"])
if not uploaded:
    st.info("Upload a CSV/TSV file to start.")
    st.stop()

df = robust_read(uploaded)

# ---------------------------
# Auto-detect and prepare columns
# ---------------------------
guesses = guess_columns([c.lower() for c in df.columns])
# preference: search in original columns case-insensitively
def pick(original_df, guess_key):
    col = guesses.get(guess_key)
    if col and col in original_df.columns:
        return col
    # try case-insensitive match
    if col:
        for c in original_df.columns:
            if c.lower() == col.lower():
                return c
    return None

# find best timestamp candidate
ts_candidate = pick(df, "timestamp")
if not ts_candidate:
    # fallback try known names
    for candidate in ["Time_parsed","Time","time_parsed","time","timestamp","Date","Time_parsed"]:
        if candidate in df.columns:
            ts_candidate = candidate
            break

if ts_candidate:
    df = parse_timestamp_column(df, ts_candidate)
else:
    df["timestamp"] = pd.NaT

# path
path_candidate = pick(df, "path")
if path_candidate:
    df["_raw_path"] = df[path_candidate].astype(str)
else:
    # fallback: first column that looks like URL or contains '/'
    fallback = None
    for c in df.columns:
        sample = df[c].astype(str).head(50).tolist()
        if any("/" in (str(x) or "") for x in sample):
            fallback = c
            break
    df["_raw_path"] = df[fallback].astype(str) if fallback else df.iloc[:,0].astype(str)

# canonicalize path (remove domain)
def canonicalize_path(p):
    if not isinstance(p, str):
        return ""
    p = p.strip()
    p = re.sub(r"^https?://[^/]+", "", p)
    return p if p else "/"

df["_canon"] = df["_raw_path"].apply(canonicalize_path)
def split_path_query(p):
    if "?" in p:
        a,b = p.split("?",1); return a,b
    return p,""
df["_path_only"], df["_qs"] = zip(*df["_canon"].apply(split_path_query))
df["_qs_count"] = df["_qs"].astype(str).apply(lambda q: 0 if q=="" else len([i for i in re.split("[&;]", q) if i]))

# status
status_candidate = pick(df, "status")
df["_status"] = df[status_candidate].astype(str) if status_candidate else df.get("Status", "").astype(str)

# client ip
ip_candidate = pick(df, "client_ip")
df["_client_ip"] = df[ip_candidate].astype(str) if ip_candidate else df.get("ClientIP", "").astype(str) if "ClientIP" in df.columns else ""

# user agent
ua_candidate = pick(df, "user_agent")
df["_user_agent"] = df[ua_candidate].astype(str) if ua_candidate else df.get("User-Agent", "").astype(str) if "User-Agent" in df.columns else ""

# bytes
bytes_candidate = pick(df, "bytes")
df["_bytes"] = pd.to_numeric(df[bytes_candidate], errors="coerce") if bytes_candidate else pd.to_numeric(df.get("Bytes", pd.Series([np.nan]*len(df))), errors="coerce")

# section
section_candidate = pick(df, "section")
df["_section"] = df[section_candidate].astype(str) if section_candidate else df.get("Section", "").astype(str) if "Section" in df.columns else ""

# agent token and static detection
def detect_static_from_path(path):
    if not isinstance(path, str) or path.strip()=="":
        return False
    p = path.split("?")[0].lower()
    exts = [".css",".js",".svg",".png",".jpg",".jpeg",".gif",".ico",".woff",".woff2",".ttf",".map",".eot",".webp"]
    return any(p.endswith(e) for e in exts)

df["_agent_token"] = df["_user_agent"].apply(extract_agent_token)
df["_is_static"] = df.get("IsStatic", None)
if df["_is_static"].isnull().all():
    df["_is_static"] = df["_path_only"].apply(detect_static_from_path)

# derived fields
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour

# sessionize
df = df.sort_values("timestamp").reset_index(drop=True)
df["_session_id"] = sessionize(df, ip_col="_client_ip", agent_col="_agent_token", timeout_minutes=30)

# optional sitemap
sitemap_paths = set()
if sitemap_file:
    try:
        txt = sitemap_file.getvalue().decode("utf-8", errors="ignore")
        locs = re.findall(r"<loc>(.*?)</loc>", txt, flags=re.IGNORECASE)
        sitemap_paths = {re.sub(r"^https?://[^/]+", "", u).split("?")[0] for u in locs}
    except Exception:
        sitemap_paths = set()

# Filters
st.sidebar.header("Filters")
min_date = df["date"].min() if df["date"].notna().any() else None
max_date = df["date"].max() if df["date"].notna().any() else None
if min_date and max_date:
    date_range = st.sidebar.date_input("Date range", (min_date, max_date))
else:
    date_range = None
selected_agents = st.sidebar.multiselect("Agent tokens (leave blank = all)", options=sorted(df["_agent_token"].unique()), default=[])
content_scope = st.sidebar.selectbox("Scope", ["All URLs", "Content-only (exclude static assets)"], index=1)
top_n = st.sidebar.slider("Top N lists", 5, 50, 25)

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

# ---------- KPIs ----------
st.header("Key SEO KPIs")
c1,c2,c3,c4,c5 = st.columns(5)
total_hits = len(df_f)
bot_hits = df_f["_agent_token"].isin(UA_TOKENS).sum()
static_hits = df_f["_is_static"].sum()
unique_urls = df_f["_path_only"].nunique()
unique_agents = df_f["_agent_token"].nunique()
c1.metric("Total hits", f"{total_hits:,}")
c2.metric("Bot-token hits", f"{bot_hits:,}", f"{bot_hits/total_hits*100:.1f}%" if total_hits else "")
c3.metric("Static hits", f"{static_hits:,}")
c4.metric("Unique URLs", f"{unique_urls:,}")
c5.metric("Unique agents", f"{unique_agents:,}")

st.markdown("---")

# ---------- Example metric: Top Crawled Paths ----------
st.subheader("Top Crawled Paths")
top_paths = (df_f.groupby("_path_only")
    .agg(hits=("_path_only","count"),
         bot_hits=("_agent_token", lambda s: s.isin(UA_TOKENS).sum()),
         unique_agents=("_agent_token","nunique"),
         first_seen=("timestamp","min"),
         last_seen=("timestamp","max"))
    .reset_index().sort_values("hits", ascending=False).head(top_n))
# format datetimes safely
top_paths["first_seen"] = pd.to_datetime(top_paths["first_seen"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
top_paths["last_seen"] = pd.to_datetime(top_paths["last_seen"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
st.dataframe(top_paths, use_container_width=True)
fig = px.bar(top_paths.sort_values("hits"), x="hits", y="_path_only", orientation="h", title="Top crawled paths")
st.plotly_chart(fig, use_container_width=True)

# ---------- Bot exposure to errors ----------
st.subheader("Bot exposure to errors (4xx/5xx)")
err = df_f[df_f["_status"].astype(str).str.startswith(("4","5"), na=False)]
err_summary = err.groupby(["_agent_token","_status"]).size().reset_index(name="count").sort_values("count", ascending=False)
st.dataframe(err_summary.head(200), use_container_width=True)
if not err_summary.empty:
    fig_err = px.bar(err_summary.head(50), x="count", y="_agent_token", color="_status", orientation="h", title="Errors by agent token")
    st.plotly_chart(fig_err, use_container_width=True)

# ---------- Orphaned URLs ----------
st.subheader("Orphaned URLs (seen only by bots)")
human_mask = ~df_f["_agent_token"].isin(UA_TOKENS) & (df_f["_agent_token"]!="bot")
human_seen = set(df_f[human_mask]["_path_only"].unique())
bot_seen = set(df_f[df_f["_agent_token"].isin(UA_TOKENS) | (df_f["_agent_token"]=="bot")]["_path_only"].unique())
orphaned = sorted(list(bot_seen - human_seen))
st.write(f"Orphaned URL count: {len(orphaned)} — showing top {top_n}")
st.dataframe(pd.DataFrame({"path":orphaned[:top_n]}), use_container_width=True)

# ---------- Duplicate content risk ----------
st.subheader("Duplicate content risk (same path, many distinct queries)")
dup_qs = (df_f.groupby("_path_only")
          .agg(hits=("_path_only","count"), distinct_qs=("_qs","nunique"), distinct_agents=("_agent_token","nunique"))
          .reset_index().sort_values(["distinct_qs","hits"], ascending=False))
st.dataframe(dup_qs.head(top_n), use_container_width=True)

# ---------- Distinct URL discovery trend ----------
st.subheader("Distinct URL discovery trend (new URLs over time)")
if df_f["timestamp"].notna().any():
    df_f = df_f.sort_values("timestamp")
    df_f["_is_new"] = ~df_f["_path_only"].duplicated()
    new_urls = df_f.groupby(pd.Grouper(key="timestamp", freq="D"))["_is_new"].sum().reset_index().rename(columns={"_is_new":"new_urls"})
    st.dataframe(new_urls.tail(60), use_container_width=True)
    fig_new = px.line(new_urls, x="timestamp", y="new_urls", title="New distinct URLs discovered per day")
    st.plotly_chart(fig_new, use_container_width=True)
else:
    st.info("Timestamps not available; discovery trend skipped.")

# ---------- Export ----------
st.markdown("---")
st.header("Export")
sample = df_f[["_path_only","_status","_agent_token","_client_ip","timestamp"]].head(200)
st.dataframe(sample, use_container_width=True)
csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", csv, file_name="filtered_logs.csv")
st.success("Analysis complete.")

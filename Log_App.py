# app.py
# SEO Log Analyzer — hardened timestamp parsing + safe date handling
# Full rewrite with your requirements applied.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import re
from datetime import timedelta

st.set_page_config(page_title="SEO Log Analyzer", layout="wide")
st.title("SEO Log Analyzer — Bot & Crawl Insights")

# -----------------------------------------------------------
# Curated UA tokens (~60)
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# Robust reader
# -----------------------------------------------------------
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
            return pd.read_csv(io.BytesIO(raw), sep=delim, engine="python",
                               on_bad_lines="skip", low_memory=False)
        except Exception:
            return pd.read_fwf(io.BytesIO(raw))

# -----------------------------------------------------------
# Auto-detect columns
# -----------------------------------------------------------
def guess_columns(columns):
    def find_any(keys):
        for kw in keys:
            for c in columns:
                if kw in c:
                    return c
        return None

    out = {}
    out["timestamp"] = find_any(["time_parsed","time","timestamp","date","datetime","ts"])
    out["path"]       = find_any(["pathclean","path","uri","request","url"])
    out["status"]     = find_any(["status","statuscode","status_code","statusclass"])
    out["ip"]         = find_any(["clientip","ip","remote_addr"])
    out["ua"]         = find_any(["user-agent","user_agent","useragent","ua"])
    out["bytes"]      = find_any(["bytes","size","response_size","content_length"])
    out["section"]    = find_any(["section"])
    return out

# -----------------------------------------------------------
# Hardened timestamp parsing
# -----------------------------------------------------------
def parse_timestamp_column(df, col):
    if col not in df.columns:
        df["timestamp"] = pd.NaT
        return df

    s = df[col].astype(str)
    df["_tmp_ts"] = pd.NaT

    # numeric detection
    with np.errstate(invalid='ignore'):
        num = pd.to_numeric(s, errors="coerce")

    if num.notna().any():
        # epoch ms
        if (num > 1e12).any():
            df["_tmp_ts"] = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
        # epoch s
        elif (num > 1e9).any():
            df["_tmp_ts"] = pd.to_datetime(num, unit="s", errors="coerce", utc=True)

    # strings for remaining
    rem = df["_tmp_ts"].isna()
    if rem.any():
        try:
            parsed = pd.to_datetime(s[rem], errors="coerce", utc=True, infer_datetime_format=True)
        except Exception:
            parsed = pd.to_datetime(s[rem], errors="coerce", utc=True)
        df.loc[rem, "_tmp_ts"] = parsed

    # fallback formats
    if df["_tmp_ts"].isna().mean() > 0.9:
        fmts = [
            "%d/%b/%Y:%H:%M:%S %z", "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S"
        ]
        for fmt in fmts:
            try:
                parsed = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
                if parsed.notna().any():
                    df["_tmp_ts"] = parsed
                    break
            except:
                pass

    # Final coercion
    df["_tmp_ts"] = pd.to_datetime(df["_tmp_ts"], errors="coerce", utc=True)

    # Finalize
    try:
        df["timestamp"] = df["_tmp_ts"].dt.tz_convert(None)
    except Exception:
        # remove timezone manually
        df["timestamp"] = pd.to_datetime(df["_tmp_ts"], errors="coerce").dt.tz_localize(None)

    df.drop(columns=["_tmp_ts"], inplace=True, errors="ignore")
    return df

# -----------------------------------------------------------
# Agent token extraction
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# Sessionization
# -----------------------------------------------------------
def sessionize(df, ip_col, agent_col, timeout=30):
    df = df.sort_values("timestamp").reset_index(drop=True)
    last_seen = {}
    sids = []
    counter = 0

    for idx, row in df.iterrows():
        ts = row["timestamp"]
        if pd.isna(ts):
            sids.append(f"nosess-{idx}")
            continue
        key = (row[ip_col], row[agent_col])
        if key not in last_seen:
            counter += 1
            sid = f"s{counter}"
            last_seen[key] = ts
            sids.append(sid)
        else:
            diff = (ts - last_seen[key]).total_seconds() / 60
            if diff > timeout:
                counter += 1
                sid = f"s{counter}"
                last_seen[key] = ts
                sids.append(sid)
            else:
                last_seen[key] = ts
                sids.append(f"s{counter}")
    return sids

# -----------------------------------------------------------
# Load file
# -----------------------------------------------------------
st.sidebar.info("Upload your log file. No column mapping UI.")
uploaded = st.sidebar.file_uploader("Upload CSV/TSV", type=["csv","tsv","txt"])
if not uploaded:
    st.stop()

df = robust_read(uploaded)

# -----------------------------------------------------------
# Auto-detect and prepare
# -----------------------------------------------------------
cols_lower = [c.lower() for c in df.columns]
guesses = guess_columns(cols_lower)

def pick(df, key):
    g = guesses.get(key)
    if not g:
        return None
    # exact
    if g in df.columns:
        return g
    # case-insensitive
    for c in df.columns:
        if c.lower() == g:
            return c
    return None

# Timestamp
ts_col = pick(df, "timestamp")
df = parse_timestamp_column(df, ts_col) if ts_col else df.assign(timestamp=pd.NaT)

# Path
path_col = pick(df, "path")
if path_col:
    df["_raw_path"] = df[path_col].astype(str)
else:
    # fallback to first col containing '/'
    candidate = None
    for c in df.columns:
        sample = df[c].astype(str).head(50)
        if any("/" in x for x in sample):
            candidate = c
            break
    df["_raw_path"] = df[candidate].astype(str) if candidate else df.iloc[:,0].astype(str)

def canonical(p):
    if not isinstance(p, str): return ""
    p = p.strip()
    p = re.sub(r"^https?://[^/]+", "", p)
    return p if p else "/"

df["_canon"] = df["_raw_path"].apply(canonical)

# path + query split
def split_pq(p):
    if "?" in p:
        a,b = p.split("?",1); return a,b
    return p,""

df["_path_only"], df["_qs"] = zip(*df["_canon"].apply(split_pq))
df["_qs_count"] = df["_qs"].astype(str).apply(lambda q: 0 if q=="" else len([x for x in re.split("[&;]", q) if x]))

# Status
status_col = pick(df, "status")
df["_status"] = df[status_col].astype(str) if status_col else ""

# IP
ip_col = pick(df, "ip")
df["_ip"] = df[ip_col].astype(str) if ip_col else ""

# UA
ua_col = pick(df, "ua")
df["_ua"] = df[ua_col].astype(str) if ua_col else ""

# Bytes
bytes_col = pick(df, "bytes")
df["_bytes"] = pd.to_numeric(df[bytes_col], errors="coerce") if bytes_col else np.nan

# Section
section_col = pick(df, "section")
df["_section"] = df[section_col].astype(str) if section_col else ""

# Static detection
def detect_static(p):
    if not isinstance(p, str): return False
    p = p.split("?")[0].lower()
    exts = [".css",".js",".svg",".png",".jpg",".jpeg",".gif",".ico",".woff",".woff2",".ttf",".map",".eot",".webp"]
    return any(p.endswith(e) for e in exts)

df["_is_static"] = df["_path_only"].apply(detect_static)

# Derived
valid_ts = df["timestamp"].notna()
df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
df["hour"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour

# Sessionize
df["_agent"] = df["_ua"].apply(extract_agent_token)
df["_session"] = sessionize(df, "_ip", "_agent", timeout=30)

# -----------------------------------------------------------
# Filters (safe date handling)
# -----------------------------------------------------------
st.sidebar.header("Filters")

if valid_ts.any():
    min_d = df.loc[valid_ts, "date"].min()
    max_d = df.loc[valid_ts, "date"].max()
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
scope = st.sidebar.selectbox("Scope", ["All URLs","Content-only"], index=1)
top_n = st.sidebar.slider("Top N", 5, 50, 25)

# Apply filters
df_f = df.copy()
if valid_ts.any() and start is not None:
    df_f = df_f[(df_f["timestamp"]>=start) & (df_f["timestamp"]<end)]
if sel_agents:
    df_f = df_f[df_f["_agent"].isin(sel_agents)]
if scope=="Content-only":
    df_f = df_f[~df_f["_is_static"]]

# -----------------------------------------------------------
# KPIs
# -----------------------------------------------------------
st.header("Key SEO KPIs")

total_hits = len(df_f)
bot_hits = df_f["_agent"].isin(UA_TOKENS).sum()
static_hits = df_f["_is_static"].sum()
unique_urls = df_f["_path_only"].nunique()
unique_agents = df_f["_agent"].nunique()

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total hits", f"{total_hits:,}")
c2.metric("Bot-token hits", f"{bot_hits:,}")
c3.metric("Static hits", f"{static_hits:,}")
c4.metric("Unique URLs", f"{unique_urls:,}")
c5.metric("Unique agents", f"{unique_agents:,}")

st.markdown("---")

# -----------------------------------------------------------
# Top Crawled Paths
# -----------------------------------------------------------
st.subheader("Top Crawled Paths")

tp = (df_f.groupby("_path_only")
      .agg(hits=("_path_only","count"),
           bot_hits=("_agent", lambda s: s.isin(UA_TOKENS).sum()),
           agents=("_agent","nunique"),
           first_seen=("timestamp","min"),
           last_seen=("timestamp","max"))
      .reset_index()
      .sort_values("hits", ascending=False)
      .head(top_n))

tp["first_seen"] = pd.to_datetime(tp["first_seen"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
tp["last_seen"]  = pd.to_datetime(tp["last_seen"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")

st.dataframe(tp, use_container_width=True)

fig = px.bar(tp.sort_values("hits"), x="hits", y="_path_only", orientation="h",
             title="Top Crawled Paths")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# Bot Exposure to Errors
# -----------------------------------------------------------
st.subheader("Bot Exposure to 4xx/5xx")

err = df_f[df_f["_status"].str.startswith(("4","5"), na=False)]
err_sum = (err.groupby(["_agent","_status"])
           .size().reset_index(name="count")
           .sort_values("count", ascending=False))

st.dataframe(err_sum.head(200), use_container_width=True)

if not err_sum.empty:
    fig_err = px.bar(err_sum.head(50), x="count", y="_agent", color="_status",
                     orientation="h", title="Error hits by agent")
    st.plotly_chart(fig_err, use_container_width=True)

# -----------------------------------------------------------
# Orphaned URLs
# -----------------------------------------------------------
st.subheader("Orphaned URLs")

human_seen = set(df_f[~df_f["_agent"].isin(UA_TOKENS) & (df_f["_agent"]!="bot")]["_path_only"])
bot_seen = set(df_f[df_f["_agent"].isin(UA_TOKENS) | (df_f["_agent"]=="bot")]["_path_only"])

orph = sorted(list(bot_seen - human_seen))
st.write(f"Count: {len(orph)} — showing {top_n}")
st.dataframe(pd.DataFrame({"path":orph[:top_n]}), use_container_width=True)

# -----------------------------------------------------------
# Duplicate Content Risk
# -----------------------------------------------------------
st.subheader("Duplicate Content Risk")

dup = (df_f.groupby("_path_only")
       .agg(hits=("_path_only","count"),
            distinct_qs=("_qs","nunique"),
            agents=("_agent","nunique"))
       .reset_index()
       .sort_values(["distinct_qs","hits"], ascending=False))

st.dataframe(dup.head(top_n), use_container_width=True)

# -----------------------------------------------------------
# Distinct URL Discovery Trend
# -----------------------------------------------------------
st.subheader("Distinct URL Discovery Trend")

if valid_ts.any():
    df_f_sorted = df_f.sort_values("timestamp")
    df_f_sorted["_is_new"] = ~df_f_sorted["_path_only"].duplicated()
    daily_new = (df_f_sorted.groupby(pd.Grouper(key="timestamp", freq="D"))["_is_new"]
                 .sum()
                 .reset_index()
                 .rename(columns={"_is_new":"new_urls"}))

    st.dataframe(daily_new.tail(60), use_container_width=True)

    fig_new = px.line(daily_new, x="timestamp", y="new_urls",
                      title="New URLs per day")
    st.plotly_chart(fig_new, use_container_width=True)
else:
    st.info("Timestamps invalid — discovery trend skipped.")

# -----------------------------------------------------------
# Export
# -----------------------------------------------------------
st.markdown("---")
st.header("Export")

sample = df_f[["_path_only","_status","_agent","_ip","timestamp"]].head(200)
st.dataframe(sample, use_container_width=True)

csv_data = df_f.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", csv_data, "filtered.csv")

st.success("Analysis complete.")

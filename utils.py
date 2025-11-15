# utils.py
import re
import pandas as pd
import numpy as np
from datetime import datetime

# Signatures lists (extendable)
BOT_SIGNATURES = [
    "gptbot", "gpt-4o", "chatgpt-user", "openai", "oai-searchbot", "perplexity", "perplexitybot",
    "claude", "claudebot", "anthropic", "mistral", "bytespider", "ccbot", "serpapi",
    "copilot", "bingbot", "googlebot", "yandex", "duckduckbot", "baiduspider", "slurp",
    "ahrefsbot", "semrushbot", "bingpreview", "sogou", "curl", "wget", "python-requests"
]

AI_SEARCH_SIGNALS = {
    "training": ["gptbot", "bytespider", "ccbot"],
    "search": ["oai-searchbot", "perplexitybot", "claude-searchbot", "bingpreview"],
    "user": ["chatgpt-user", "perplexity-user", "claude-user", "mistral-user"]
}

# --- Column guessing ---
def guess_columns(columns):
    names = {c.lower(): c for c in columns}
    out = {}
    def find_any(keywords):
        for kw in keywords:
            for c in columns:
                if kw in c.lower():
                    return c
        return None
    out["timestamp"] = find_any(["time","timestamp","date","datetime","ts","logtime"])
    out["user_agent"] = find_any(["useragent","user_agent","ua","agent"])
    out["path"] = find_any(["url","uri","path","request","page"])
    out["status"] = find_any(["status","http_status","code","response_code"])
    out["ip"] = find_any(["ip","clientip","remote_addr","remote"])
    out["bytes"] = find_any(["bytes","size","response_size","content_length"])
    out["section"] = find_any(["section","area","site_section"])
    return out

# --- Timestamp parsing ---
def parse_timestamp_column(df, col):
    # Attempts multiple robust parsing strategies and writes df["timestamp"]
    s = df[col]
    # numeric epoch detection
    try:
        numeric_like = pd.to_numeric(s, errors="coerce")
        if numeric_like.notna().any():
            # heuristics: > 1e12 -> ms, > 1e9 -> s
            if (numeric_like > 1e12).any():
                df["timestamp"] = pd.to_datetime(numeric_like, unit="ms", errors="coerce", utc=True)
            elif (numeric_like > 1e9).any():
                df["timestamp"] = pd.to_datetime(numeric_like, unit="s", errors="coerce", utc=True)
            else:
                # fallback to string parse
                df["timestamp"] = pd.to_datetime(s, errors="coerce", utc=True, infer_datetime_format=True)
        else:
            df["timestamp"] = pd.to_datetime(s, errors="coerce", utc=True, infer_datetime_format=True)
    except Exception:
        df["timestamp"] = pd.to_datetime(s, errors="coerce", utc=True, infer_datetime_format=True)

    # fallback common CLF format if all NaT
    if df["timestamp"].isna().all():
        for fmt in ("%d/%b/%Y:%H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                df["timestamp"] = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
                if df["timestamp"].notna().any():
                    break
            except Exception:
                pass
    # final fallback - infer
    if df["timestamp"].isna().all():
        df["timestamp"] = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

    return df

def normalize_timestamp_series(series):
    # Ensure timezone-naive (UTC) datetimes for uniform grouping; keep NaT
    s = pd.to_datetime(series, errors="coerce", utc=True)
    # convert to naive (drop tzinfo)
    return s.dt.tz_convert(None)

# --- User-Agent classification ---
def classify_user_agent(ua):
    if not isinstance(ua, str) or ua.strip() == "":
        return "unknown"
    ua_l = ua.lower()
    # ai-search/user priority
    for kind, sigs in AI_SEARCH_SIGNALS.items():
        for s in sigs:
            if s.lower() in ua_l:
                return f"ai_search_{kind}"
    # known bot signatures
    for s in BOT_SIGNATURES:
        if s in ua_l:
            # return the canonical signature
            return s
    # generic heuristics
    if re.search(r"(bot|spider|crawler|scraper|wget|curl|python-requests)", ua_l):
        return "bot_other"
    # else assume human/browser
    # add detection for common browser strings to improve label
    if re.search(r"(mozilla|chrome|safari|edge|firefox)", ua_l):
        return "human"
    return "unknown"

# --- static detection ---
def detect_static_path(path, extensions=None, regex=None):
    if not isinstance(path, str) or path.strip() == "":
        return False
    p = path.split("?",1)[0].lower()
    if extensions:
        for ext in extensions:
            e = ext.strip().lower()
            if not e:
                continue
            if not e.startswith("."):
                e = "." + e
            if p.endswith(e):
                return True
    if regex:
        try:
            if re.search(regex, p):
                return True
        except Exception:
            pass
    return False

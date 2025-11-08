# utils.py
import re
import pandas as pd
from datetime import datetime

BOT_SIGNATURES = [
    "GPTBot", "ChatGPT", "ChatGPT-User", "OAI-SearchBot", "PerplexityBot",
    "Perplexity", "Claude", "ClaudeBot", "Claude-Searchbot", "CCBot",
    "Bytespider", "Bingbot", "Googlebot", "Mistral", "Bard", "bingpreview",
    "DuckDuckBot", "Yandex", "Sogou"
]

AI_SEARCH_SIGNALS = {
    "training": ["GPTBot", "Bytespider", "CCBot"],
    "search": ["OAI-SearchBot", "PerplexityBot", "Claude-Searchbot", "BingPreview"],
    "user": ["ChatGPT-User", "Perplexity-User", "Claude-User", "Mistral-User", "ChatGPT"]
}

def guess_columns(columns):
    # heuristic mapping for common log exports
    names = {c.lower(): c for c in columns}
    out = {}
    for keyword in ["time", "timestamp", "date", "datetime", "ts"]:
        for c in columns:
            if keyword in c.lower():
                out["timestamp"] = c
                break
        if out.get("timestamp"):
            break
    for keyword in ["useragent", "user_agent", "agent", "ua"]:
        for c in columns:
            if keyword in c.lower():
                out["user_agent"] = c
                break
        if out.get("user_agent"):
            break
    for keyword in ["url", "uri", "path", "request", "page"]:
        for c in columns:
            if keyword in c.lower():
                out["path"] = c
                break
        if out.get("path"):
            break
    for keyword in ["status", "http_status", "code"]:
        for c in columns:
            if keyword in c.lower():
                out["status"] = c
                break
        if out.get("status"):
            break
    for keyword in ["ip", "clientip", "remote_addr"]:
        for c in columns:
            if keyword in c.lower():
                out["ip"] = c
                break
        if out.get("ip"):
            break
    for keyword in ["bytes", "size", "response_size", "content_length"]:
        for c in columns:
            if keyword in c.lower():
                out["bytes"] = c
                break
        if out.get("bytes"):
            break
    for keyword in ["resp", "response", "latency", "time_taken"]:
        for c in columns:
            if keyword in c.lower():
                out["response_time"] = c
                break
        if out.get("response_time"):
            break
    return out

def parse_timestamp_column(df, col):
    # robust parsing attempts: ISO, epoch (ms or s), common formats
    s = df[col].copy()
    # epoch detection
    try:
        # if numeric-ish and large values: treat as epoch ms or s
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s) or s.dropna().astype(str).str.match(r'^\d{9,}$').all():
            # convert column to numeric
            s_num = pd.to_numeric(s, errors="coerce")
            # heuristics: > 1e12 => ms, > 1e9 => s
            if (s_num > 1e12).any():
                df["timestamp"] = pd.to_datetime(s_num, unit="ms", errors="coerce")
            else:
                df["timestamp"] = pd.to_datetime(s_num, unit="s", errors="coerce")
            # If conversion produced many NaNs, try parsing as string next
            if df["timestamp"].isna().mean() > 0.5:
                df["timestamp"] = pd.to_datetime(df[col], errors="coerce", utc=None)
        else:
            df["timestamp"] = pd.to_datetime(df[col], errors="coerce", utc=None)
    except Exception:
        df["timestamp"] = pd.to_datetime(df[col], errors="coerce", utc=None)
    # fallback: if still NaT, try common formats
    if df["timestamp"].isna().all():
        tried = False
        for fmt in ("%d/%b/%Y:%H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                df["timestamp"] = pd.to_datetime(df[col], format=fmt, errors="coerce")
                if df["timestamp"].notna().any():
                    tried = True
                    break
            except Exception:
                pass
        if not tried:
            # last resort parse
            df["timestamp"] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
    return df

def classify_user_agent(ua):
    if not isinstance(ua, str) or ua.strip() == "":
        return "unknown"
    ua_l = ua.lower()
    # priority ai-specific patterns
    for k, sigs in AI_SEARCH_SIGNALS.items():
        for s in sigs:
            if s.lower() in ua_l:
                return f"ai_search_{k}"
    # known bot signatures
    for s in BOT_SIGNATURES:
        if s.lower() in ua_l:
            return s
    # generic bot detection heuristics
    if re.search(r"(bot|spider|crawler|crawl|scraper|wget|curl|python-requests)", ua_l):
        return "bot_other"
    # else assume human / normal browser
    return "human"

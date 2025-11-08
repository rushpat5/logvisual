# Log_App.py
# Streamlit Website Log Analyzer — MyVi Structured Log Edition

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Website Log Analyzer — MyVi Logs", layout="wide", initial_sidebar_state="expanded")
st.title("Website Log Analyzer — AI Search & Bot Insights")

# Sidebar upload
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV log file", type=["csv", "txt"])

@st.cache_data
def load_csv(file):
    return pd.read_csv(file, low_memory=False)

if not uploaded_file:
    st.info("Upload your CSV file (e.g., detailed_hits.csv) to start analysis.")
    st.stop()

df = load_csv(uploaded_file)
st.sidebar.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

# --- Parse timestamp ---
if "Time" not in df.columns:
    st.error("Expected a 'Time' column in your CSV (ISO 8601 format).")
    st.stop()

df["timestamp"] = pd.to_datetime(df["Time"], errors="coerce", utc=True).dt.tz_convert(None)
df = df[df["timestamp"].notna()].copy()

# --- Normalize expected fields ---
rename_map = {
    "User-Agent": "user_agent",
    "Path": "path",
    "PathClean": "path_clean",
    "Status": "status",
    "Bot Type": "bot_type",
    "IsStatic": "is_static",
    "IsMobile": "is_mobile",
    "Section": "section",
}
for k, v in rename_map.items():
    if k in df.columns:
        df.rename(columns={k: v}, inplace=True)

# --- Basic prep ---
df["bot_type"] = df["bot_type"].fillna("Unknown")
df["is_static"] = df["is_static"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
df["is_bot"] = df["bot_type"].ne("Unknown")

# --- Sidebar filters ---
st.sidebar.header("Filters")

min_date, max_date = df["timestamp"].min(), df["timestamp"].max()
default_range = (min_date.date(), max_date.date())
date_range = st.sidebar.date_input("Date range", value=default_range)
start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]

filter_bots_content = st.sidebar.checkbox("Show only bots hitting content pages (exclude static assets)", value=False)
if filter_bots_content:
    df = df[(df["is_bot"]) & (~df["is_static"])]

show_bots = st.sidebar.checkbox("Include bots in charts", value=True)
top_n = st.sidebar.slider("Top N Sections", 5, 50, 10)

# --- KPIs ---
k1, k2, k3, k4 = st.columns(4)
total_hits = len(df)
bot_hits = len(df[df["is_bot"]])
unique_sections = df["section"].nunique() if "section" in df.columns else 0
unique_pages = df["path_clean"].nunique() if "path_clean" in df.columns else 0

k1.metric("Total Hits", f"{total_hits:,}")
k2.metric("Bot Hits", f"{bot_hits:,}", f"{(bot_hits / total_hits * 100):.1f}%" if total_hits else "0%")
k3.metric("Unique Sections", f"{unique_sections:,}")
k4.metric("Unique Pages", f"{unique_pages:,}")

st.markdown("---")

# --- Time-series ---
st.subheader("Traffic Over Time (Hourly)")
ts = df.copy()
ts["hour"] = ts["timestamp"].dt.floor("H")
if show_bots:
    ts_group = ts.groupby(["hour", "is_bot"]).size().reset_index(name="count")
    ts_group["type"] = ts_group["is_bot"].map({True: "Bot", False: "Human"})
else:
    ts_group = ts[~ts["is_bot"]].groupby(["hour"]).size().reset_index(name="count")
    ts_group["type"] = "Human"

fig_ts = px.line(ts_group, x="hour", y="count", color="type", markers=False, title="Hourly Traffic (Bots vs Humans)")
st.plotly_chart(fig_ts, use_container_width=True)

# --- Section distribution ---
if "section" in df.columns:
    st.subheader("Traffic by Site Section")
    section_df = df.groupby(["section", "is_bot"]).size().reset_index(name="count")
    section_df["type"] = section_df["is_bot"].map({True: "Bot", False: "Human"})
    section_df = section_df.sort_values("count", ascending=False)
    top_sections = section_df.groupby("section")["count"].sum().nlargest(top_n).index
    fig_section = px.bar(
        section_df[section_df["section"].isin(top_sections)],
        x="section",
        y="count",
        color="type",
        barmode="group",
        title=f"Top {top_n} Sections by Traffic",
    )
    st.plotly_chart(fig_section, use_container_width=True)

# --- Bot type breakdown ---
st.subheader("Bot Type Breakdown")
bot_summary = df[df["is_bot"]].groupby("bot_type").size().reset_index(name="count").sort_values("count", ascending=False)
if not bot_summary.empty:
    fig_bot = px.pie(bot_summary, names="bot_type", values="count", title="Distribution of Bot Types")
    st.plotly_chart(fig_bot, use_container_width=True)
else:
    st.info("No bot activity detected for this selection.")

# --- Static vs Dynamic pages ---
st.subheader("Static vs Dynamic Requests")
static_counts = df.groupby(["is_static", "is_bot"]).size().reset_index(name="count")
static_counts["content_type"] = static_counts["is_static"].map({True: "Static", False: "Content"})
static_counts["actor"] = static_counts["is_bot"].map({True: "Bot", False: "Human"})
fig_static = px.bar(static_counts, x="content_type", y="count", color="actor", barmode="group", title="Static vs Content Page Hits")
st.plotly_chart(fig_static, use_container_width=True)

# --- Device type ---
if "is_mobile" in df.columns:
    st.subheader("Device Type Breakdown")
    device_df = df.groupby(["is_mobile", "is_bot"]).size().reset_index(name="count")
    device_df["device"] = device_df["is_mobile"].map({True: "Mobile", False: "Desktop"})
    device_df["actor"] = device_df["is_bot"].map({True: "Bot", False: "Human"})
    fig_device = px.bar(device_df, x="device", y="count", color="actor", barmode="group", title="Mobile vs Desktop Activity")
    st.plotly_chart(fig_device, use_container_width=True)

# --- Status codes ---
st.subheader("HTTP Status Code Distribution")
status_df = df.groupby(["status", "is_bot"]).size().reset_index(name="count")
status_df["actor"] = status_df["is_bot"].map({True: "Bot", False: "Human"})
fig_status = px.bar(status_df, x="status", y="count", color="actor", barmode="group", title="HTTP Status Codes by Actor")
st.plotly_chart(fig_status, use_container_width=True)

# --- Data explorer ---
st.subheader("Data Explorer (first 1000 rows)")
st.dataframe(df.head(1000), use_container_width=True)

st.download_button("Download Filtered CSV", df.to_csv(index=False).encode("utf-8"), file_name="filtered_hits.csv")

st.markdown("---")
st.sidebar.markdown("---")
st.sidebar.caption("MyVi Log Analyzer — Context-aware for structured exports.")
st.success("Analysis complete.")

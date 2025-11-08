import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(page_title="Bot Traffic Intelligence", layout="wide", page_icon="🤖")

# -------------------------------
# Bot Signatures
# -------------------------------
BOT_SIGNATURES = {
    "Googlebot": [r"googlebot", r"googleother"],
    "Bingbot": [r"bingbot", r"msnbot"],
    "Applebot": [r"applebot"],
    "GPTBot": [r"gptbot"],
    "ChatGPT-User": [r"chatgpt[-_ ]?user"],
    "ClaudeBot": [r"claudebot"],
    "PerplexityBot": [r"perplexity(bot|ai)"],
    "Perplexity-User": [r"perplexity[-_ ]?user"],
    "MozBot": [r"moz"],
    "AhrefsBot": [r"ahrefs"],
    "SemrushBot": [r"semrush"],
    "CommonCrawl": [r"commoncrawl"],
}

def identify_bot(ua: str) -> str:
    ua = str(ua).lower()
    for bot, patterns in BOT_SIGNATURES.items():
        for p in patterns:
            if re.search(p, ua):
                return bot
    return "Unclassified Bot"

# -------------------------------
# Layout Header
# -------------------------------
st.markdown(
    "<h1 style='text-align:center; color:#FFFFFF;'>🤖 Unified Bot Intelligence Dashboard</h1>",
    unsafe_allow_html=True,
)
st.caption("AI and traditional crawler analytics with executive-level visualization.")

uploaded = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded:
    df = pd.read_excel(uploaded)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Validate
    required = {"user-agent", "url", "status", "date"}
    if not required.issubset(df.columns):
        st.error(f"Missing required columns: {required - set(df.columns)}")
        st.stop()

    # Enrich
    df["bot_normalized"] = df["user-agent"].apply(identify_bot)
    df["status"] = df["status"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Sidebar
    st.sidebar.header("Filters")
    min_d, max_d = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input("Date Range", [min_d, max_d])
    df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]

    # Metrics
    total_hits = len(df)
    unique_bots = df["bot_normalized"].nunique()
    unique_urls = df["url"].nunique()
    avg_hits_bot = int(df.groupby("bot_normalized").size().mean())
    ai_hits = df[df["bot_normalized"].str.contains("gpt|claude|perplexity", case=False)].shape[0]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Hits", f"{total_hits:,}")
    k2.metric("Unique Bots", unique_bots)
    k3.metric("Unique URLs", unique_urls)
    k4.metric("Avg Hits/Bot", f"{avg_hits_bot:,}")
    k5.metric("AI-Driven Hits", f"{ai_hits:,}")

    st.markdown("---")

    # -------------------------------
    # Top Bots
    # -------------------------------
    bot_hits = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
    fig1 = px.bar(
        bot_hits.head(15),
        y="bot_normalized",
        x="hits",
        orientation="h",
        color="hits",
        color_continuous_scale="Plasma",
        title="Top 15 Bots by Total Hits",
    )
    fig1.update_layout(
        height=600,
        xaxis_title="Hits",
        yaxis_title=None,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FFFFFF", size=13),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # -------------------------------
    # HTTP Status Visualization (Fixed)
    # -------------------------------
    st.markdown("### 🧩 HTTP Status Efficiency (Categorical, % Share)")
    status_summary = df.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
    status_summary["percent"] = (
        status_summary["count"]
        / status_summary.groupby("bot_normalized")["count"].transform("sum")
        * 100
    )
    status_summary["status"] = status_summary["status"].astype(str)
    order = ["200", "204", "301", "302", "304", "400", "403", "404", "500"]

    color_map = {
        "200": "#00C851",
        "301": "#ffbb33",
        "302": "#ffbb33",
        "304": "#33b5e5",
        "400": "#ff4444",
        "403": "#ff4444",
        "404": "#CC0000",
        "500": "#9933CC",
    }

    fig2 = go.Figure()
    for bot in status_summary["bot_normalized"].unique():
        bot_data = status_summary[status_summary["bot_normalized"] == bot]
        fig2.add_trace(
            go.Bar(
                x=bot_data["status"],
                y=bot_data["percent"],
                name=bot,
                marker_color=[color_map.get(x, "#888") for x in bot_data["status"]],
                hovertemplate="%{x}: %{y:.1f}%<extra>" + bot + "</extra>",
            )
        )
    fig2.update_layout(
        barmode="group",
        xaxis=dict(title="HTTP Status Code", categoryorder="array", categoryarray=order),
        yaxis=dict(title="Traffic Share (%)"),
        height=700,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FFFFFF", size=13),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------
    # Trend Line
    # -------------------------------
    st.markdown("### 📈 Crawl Activity Trend")
    trend = df.groupby(["date", "bot_normalized"]).size().reset_index(name="hits")
    fig3 = px.line(
        trend,
        x="date",
        y="hits",
        color="bot_normalized",
        markers=True,
        title="Daily Crawl Volume by Bot",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig3.update_layout(
        height=600,
        xaxis_title="Date",
        yaxis_title="Hits",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FFFFFF"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # -------------------------------
    # URL Heatmap
    # -------------------------------
    st.markdown("### 🌐 Most Crawled URLs (Heatmap)")
    top_urls = df.groupby(["bot_normalized", "url"]).size().reset_index(name="hits")
    top_urls = top_urls.groupby("bot_normalized").apply(lambda x: x.nlargest(5, "hits")).reset_index(drop=True)
    fig4 = px.treemap(
        top_urls,
        path=["bot_normalized", "url"],
        values="hits",
        color="hits",
        color_continuous_scale="Viridis",
        title="Top 5 URLs per Bot (Treemap View)",
    )
    fig4.update_layout(
        height=900,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FFFFFF"),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # -------------------------------
    # Deep Dive Table
    # -------------------------------
    st.markdown("### 🎯 Detailed Bot View")
    bot_choice = st.selectbox("Select Bot for Detailed Breakdown", sorted(df["bot_normalized"].unique()))
    df_bot = df[df["bot_normalized"] == bot_choice]
    table = df_bot.groupby(["url", "status"]).size().reset_index(name="hits").sort_values("hits", ascending=False)
    st.dataframe(table, use_container_width=True, height=400)
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Breakdown", csv, f"{bot_choice}_report.csv", "text/csv")

else:
    st.info("Upload an Excel file to begin.")

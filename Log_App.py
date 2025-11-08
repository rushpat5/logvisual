import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(page_title="Bot Traffic Intelligence Dashboard", layout="wide", page_icon="🤖")

# --- BOT SIGNATURES ---
BOT_SIGNATURES = {
    # Search
    "Googlebot": [r"googlebot", r"googleother", r"mediapartners-google"],
    "Bingbot": [r"bingbot", r"msnbot", r"adidxbot"],
    "Applebot": [r"applebot"],
    "YandexBot": [r"yandex(bot|images|video|news)"],
    "BaiduSpider": [r"baiduspider"],
    "DuckDuckBot": [r"duckduck(bot|go)"],
    "YahooBot": [r"slurp"],
    "Qwantify": [r"qwantify"],

    # AI Crawlers
    "GPTBot": [r"gptbot", r"openai[-_ ]?(collector|crawler|bot)"],
    "OAI-SearchBot": [r"oai[-_ ]?searchbot"],
    "ChatGPT-User": [r"chatgpt[-_ ]?user"],
    "ClaudeBot": [r"claudebot", r"anthropic"],
    "Claude-SearchBot": [r"claude[-_ ]?searchbot"],
    "Claude-User": [r"claude[-_ ]?user"],
    "PerplexityBot": [r"perplexity(bot|ai)"],
    "Perplexity-User": [r"perplexity[-_ ]?user"],
    "MistralAI-User": [r"mistral(ai|user)"],
    "CCBot": [r"ccbot"],
    "Bytespider": [r"bytespider"],
    "YouBot": [r"you[-_ ]?search[-_ ]?bot"],

    # SEO Tools
    "AhrefsBot": [r"ahrefs"],
    "SemrushBot": [r"semrush"],
    "MozBot": [r"moz"],
    "MajesticBot": [r"mj12bot"],
    "ScreamingFrog": [r"screamingfrog"],
    "Sitebulb": [r"sitebulb"],
    "DeepCrawl": [r"deepcrawl"],
    "JetOctopus": [r"jetoctopus"],

    # Social
    "FacebookBot": [r"facebookexternalhit", r"facebot"],
    "TwitterBot": [r"twitterbot", r"xbot"],
    "LinkedInBot": [r"linkedinbot"],
    "PinterestBot": [r"pinterest"],
    "Slackbot": [r"slackbot"],
    "DiscordBot": [r"discordbot"],

    # Monitoring
    "GTMetrix": [r"gtmetrix"],
    "Pingdom": [r"pingdom"],
    "UptimeRobot": [r"uptimerobot"],
    "StatusCake": [r"statuscake"],
    "CloudflareBot": [r"cloudflare"],
    "CommonCrawl": [r"commoncrawl"],
    "DotBot": [r"dotbot"]
}

AI_GROUPS = {
    "Training": ["GPTBot", "ClaudeBot", "CCBot", "Bytespider"],
    "Search": ["OAI-SearchBot", "PerplexityBot", "Claude-SearchBot", "YouBot"],
    "User": ["ChatGPT-User", "Claude-User", "Perplexity-User", "MistralAI-User"]
}

def identify_bot(ua: str) -> str:
    ua = str(ua).lower()
    for bot, patterns in BOT_SIGNATURES.items():
        if any(re.search(p, ua) for p in patterns):
            return bot
    return "Unclassified Bot"

def classify_ai_group(bot: str) -> str:
    for group, bots in AI_GROUPS.items():
        if bot in bots:
            return group
    return "Non-AI"

# --- UI HEADER ---
st.markdown("<h1 style='text-align: center;'>🤖 Unified Bot Traffic Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.caption("Analyze activity across AI crawlers, search engines, and SEO tools for deeper visibility insights.")

uploaded = st.file_uploader("📂 Upload Excel (with User-Agent, URL, Status, Date columns)", type=["xlsx", "xls"])

if uploaded:
    df = pd.read_excel(uploaded)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required_cols = {"user-agent", "url", "status", "date"}
    if not required_cols.issubset(df.columns):
        st.error(f"Missing columns: {required_cols - set(df.columns)}")
        st.stop()

    df["bot_normalized"] = df["user-agent"].apply(identify_bot)
    df["ai_group"] = df["bot_normalized"].apply(classify_ai_group)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["status"] = df["status"].astype(str)

    # --- FILTERS ---
    st.sidebar.header("🔍 Filters")
    min_date, max_date = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]

    # --- KPIs ---
    total_hits = len(df)
    unique_bots = df["bot_normalized"].nunique()
    unique_urls = df["url"].nunique()
    ai_hits = df[df["ai_group"] != "Non-AI"].shape[0]
    search_hits = df[df["bot_normalized"].isin(["Googlebot", "Bingbot", "Applebot"])].shape[0]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Bot Hits", f"{total_hits:,}")
    k2.metric("Unique Bots", unique_bots)
    k3.metric("Unique URLs", unique_urls)
    k4.metric("AI Bot Hits", f"{ai_hits:,}")
    k5.metric("Search Bot Hits", f"{search_hits:,}")

    st.markdown("---")

    # --- BOT HIT DISTRIBUTION ---
    st.subheader("📊 Bot Activity Overview")
    hits = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
    fig = px.bar(
        hits.head(20),
        x="hits",
        y="bot_normalized",
        orientation="h",
        color="hits",
        color_continuous_scale="viridis",
        title="Top 20 Bots by Total Hits"
    )
    fig.update_layout(yaxis_title=None, xaxis_title="Total Hits", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # --- AI GROUP DISTRIBUTION ---
    st.subheader("🤖 AI Bot Segmentation")
    ai_summary = df.groupby("ai_group").size().reset_index(name="hits")
    fig2 = px.pie(
        ai_summary,
        names="ai_group",
        values="hits",
        title="AI vs Non-AI Bot Traffic Composition",
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- STATUS EFFICIENCY ---
    st.subheader("📈 HTTP Status Efficiency")
    status_summary = df.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
    status_order = ["200", "204", "301", "302", "304", "400", "403", "404", "500"]
    fig3 = px.bar(
        status_summary,
        x="status",
        y="count",
        color="bot_normalized",
        category_orders={"status": status_order},
        barmode="group",
        title="HTTP Status Distribution by Bot"
    )
    fig3.update_layout(height=700, xaxis_title="HTTP Status Code", yaxis_title="Hit Count")
    st.plotly_chart(fig3, use_container_width=True)

    # --- TIME TREND ---
    st.subheader("📅 Bot Crawl Trend Over Time")
    trend = df.groupby(["date", "ai_group"]).size().reset_index(name="hits")
    fig4 = px.line(
        trend,
        x="date",
        y="hits",
        color="ai_group",
        markers=True,
        title="Daily Bot Activity (AI vs Non-AI)"
    )
    fig4.update_layout(height=600, xaxis_title="Date", yaxis_title="Hits")
    st.plotly_chart(fig4, use_container_width=True)

    # --- TOP URLS ---
    st.subheader("🌐 Most Crawled URLs by Bot")
    top_urls = df.groupby(["bot_normalized", "url"]).size().reset_index(name="hits").sort_values("hits", ascending=False)
    top_urls = top_urls.groupby("bot_normalized").head(5)
    fig5 = px.bar(
        top_urls,
        x="hits",
        y="url",
        color="bot_normalized",
        orientation="h",
        title="Top 5 URLs per Bot"
    )
    fig5.update_layout(height=900, yaxis_title=None, xaxis_title="Hits")
    st.plotly_chart(fig5, use_container_width=True)

    # --- DETAILED BOT VIEW ---
    st.subheader("🎯 Deep Dive: Individual Bot")
    selected_bot = st.selectbox("Select Bot", sorted(df["bot_normalized"].unique()))
    df_bot = df[df["bot_normalized"] == selected_bot]
    detail = df_bot.groupby(["url", "status"]).size().reset_index(name="hits").sort_values("hits", ascending=False)
    st.write(f"**{selected_bot}** — {len(df_bot):,} hits across {df_bot['url'].nunique()} unique URLs")

    fig6 = px.treemap(detail, path=["status", "url"], values="hits", title=f"{selected_bot}: URL & Status Breakdown")
    fig6.update_traces(textinfo="label+value")
    st.plotly_chart(fig6, use_container_width=True)

    csv = detail.to_csv(index=False).encode("utf-8")
    st.download_button(f"⬇ Download {selected_bot} Breakdown", csv, f"{selected_bot}_details.csv", "text/csv")

else:
    st.info("Upload your Excel file to start analyzing bot traffic patterns.")

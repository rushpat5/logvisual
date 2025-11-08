import streamlit as st
import pandas as pd
import plotly.express as px
import re
from datetime import datetime

st.set_page_config(page_title="Bot Traffic Intelligence Dashboard", page_icon="🤖", layout="wide")

# --- BOT SIGNATURES (Core + AI + Search + SEO) ---
BOT_SIGNATURES = {
    # Search Bots
    "Googlebot": [r"googlebot", r"googleother", r"mediapartners-google"],
    "Bingbot": [r"bingbot", r"msnbot", r"adidxbot"],
    "Applebot": [r"applebot"],
    "YandexBot": [r"yandex(bot|images|video|news)"],
    "BaiduSpider": [r"baiduspider"],
    "DuckDuckBot": [r"duckduck(bot|go)"],
    "Yahoo! Slurp": [r"slurp"],
    "SogouSpider": [r"sogou"],
    "Exabot": [r"exabot"],
    "Qwantify": [r"qwantify"],
    "SeznamBot": [r"seznambot"],

    # AI / LLM Crawlers
    "GPTBot": [r"gptbot", r"openai[-_ ]?(collector|crawler|bot)"],
    "OAI-SearchBot": [r"oai[-_ ]?searchbot"],
    "ChatGPT-User": [r"chatgpt[-_ ]?user"],
    "ClaudeBot": [r"claudebot", r"anthropic[-_ ]?ai"],
    "Claude-SearchBot": [r"claude[-_ ]?searchbot"],
    "Claude-User": [r"claude[-_ ]?user"],
    "PerplexityBot": [r"perplexity(bot|ai)"],
    "Perplexity-User": [r"perplexity[-_ ]?user"],
    "MistralAI-User": [r"mistral(ai|user)"],
    "CCBot": [r"ccbot"],
    "Bytespider": [r"bytespider"],
    "YouBot": [r"you[-_ ]?search[-_ ]?bot"],
    "CohereAI": [r"cohere[-_ ]?ai"],

    # SEO & Marketing Bots
    "AhrefsBot": [r"ahrefs"],
    "SemrushBot": [r"semrush"],
    "MajesticBot": [r"mj12bot"],
    "MozBot": [r"moz"],
    "ScreamingFrog": [r"screamingfrog"],
    "DeepCrawlBot": [r"deepcrawl"],
    "Sitebulb": [r"sitebulb"],
    "JetOctopus": [r"jetoctopus"],
    "OncrawlBot": [r"oncrawl"],
    "LumarBot": [r"lumar"],

    # Social / Media
    "FacebookBot": [r"facebookexternalhit", r"facebot"],
    "TwitterBot": [r"twitterbot", r"xbot"],
    "LinkedInBot": [r"linkedinbot"],
    "PinterestBot": [r"pinterest"],
    "RedditBot": [r"reddit"],
    "Slackbot": [r"slackbot"],
    "DiscordBot": [r"discordbot"],
    "InstagramBot": [r"instagram"],
    "TikTokBot": [r"tiktok"],

    # Monitoring / Utility
    "GTMetrix": [r"gtmetrix"],
    "Pingdom": [r"pingdom"],
    "UptimeRobot": [r"uptimerobot"],
    "StatusCake": [r"statuscake"],
    "GooglePageSpeed": [r"pagespeed"],
    "CloudflareHealthCheck": [r"cloudflare[-_ ]?(healthcheck|diagnostics)"],
    "PetalBot": [r"petalbot"],
    "AmazonBot": [r"amazon(bot|adsbot)"],
    "CommonCrawl": [r"commoncrawl"],
    "DotBot": [r"dotbot"],
    "MegaIndexBot": [r"megaindex"],
    "ArchiveBot": [r"archive(-| )?org"],
    "GenericCrawler": [r"crawler", r"spider", r"fetcher", r"scanner"]
}

# --- AI BOT PURPOSE SEGMENTATION ---
AI_GROUPS = {
    "Training": ["GPTBot", "ClaudeBot", "CCBot", "Bytespider"],
    "Search": ["OAI-SearchBot", "PerplexityBot", "Claude-SearchBot", "YouBot"],
    "User": ["ChatGPT-User", "Perplexity-User", "Claude-User", "MistralAI-User"]
}

def identify_bot(user_agent: str) -> str:
    ua = str(user_agent).lower()
    for bot_name, patterns in BOT_SIGNATURES.items():
        for p in patterns:
            if re.search(p, ua):
                return bot_name
    return "Unclassified Bot"

def classify_ai_group(bot_name: str) -> str:
    for group, bots in AI_GROUPS.items():
        if bot_name in bots:
            return group
    return "Non-AI"

# --- INTERFACE ---
st.title("🤖 Unified Bot Traffic Intelligence Dashboard")
st.caption("AI, Search, SEO, and Utility bot analytics for full crawl visibility")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required_cols = ["user-agent", "url", "status", "date"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns: {required_cols}")
        st.stop()

    df["bot_normalized"] = df["user-agent"].apply(identify_bot)
    df["ai_group"] = df["bot_normalized"].apply(classify_ai_group)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- DATE FILTER ---
    min_date, max_date = df["date"].min(), df["date"].max()
    st.subheader("📅 Date Range Filter")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = c2.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

    # --- METRICS ---
    total_hits = len(df)
    unique_bots = df["bot_normalized"].nunique()
    unique_urls = df["url"].nunique()
    status_codes = df["status"].nunique()
    ai_hits = df[df["ai_group"] != "Non-AI"].shape[0]
    search_hits = df[df["bot_normalized"].isin(["Googlebot", "Bingbot", "Applebot", "YandexBot", "BaiduSpider"])].shape[0]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Hits", f"{total_hits:,}")
    c2.metric("Unique Bots", unique_bots)
    c3.metric("Unique URLs Hit", unique_urls)
    c4.metric("Status Codes", status_codes)
    c5.metric("AI Bot Hits", f"{ai_hits:,}")
    c6.metric("Search Bot Hits", f"{search_hits:,}")

    # --- VISUALIZATIONS ---
    st.markdown("### 📊 Hits by Bot Family")
    bot_hits = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
    fig1 = px.bar(bot_hits.head(20), x="hits", y="bot_normalized", orientation="h",
                  title="Top 20 Bots by Activity", color="bot_normalized")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 🤖 AI Group Composition")
    ai_summary = df.groupby("ai_group").size().reset_index(name="hits")
    fig2 = px.pie(ai_summary, names="ai_group", values="hits", title="AI vs Non-AI Bot Distribution", hole=0.4)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### ⏱ Crawl Trend Over Time")
    trend = df.groupby(["date", "ai_group"]).size().reset_index(name="hits")
    fig3 = px.line(trend, x="date", y="hits", color="ai_group",
                   title="AI Bot Crawl Trend (Daily Activity)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 🌐 Top URLs Crawled")
    url_hits = df.groupby(["bot_normalized", "url"]).size().reset_index(name="hits").sort_values("hits", ascending=False)
    top_urls = url_hits.groupby("bot_normalized").head(5)
    fig4 = px.bar(top_urls, x="hits", y="url", color="bot_normalized",
                  title="Top 5 URLs per Bot", orientation="h", height=900)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 🔢 HTTP Status Efficiency")
    status_summary = df.groupby(["bot_normalized", "status"]).size().reset_index(name="count")
    fig5 = px.bar(status_summary, x="status", y="count", color="bot_normalized",
                  title="HTTP Status Distribution by Bot", barmode="group")
    st.plotly_chart(fig5, use_container_width=True)

    # --- DEEP DIVE SECTION ---
    st.markdown("### 🎯 Deep Dive into Specific Bot")
    bot_choice = st.selectbox("Select a bot for detailed view", sorted(df["bot_normalized"].unique()))
    df_bot = df[df["bot_normalized"] == bot_choice]
    df_bot_summary = df_bot.groupby(["url", "status"]).size().reset_index(name="hits").sort_values("hits", ascending=False)
    st.write(f"**{bot_choice}** — {len(df_bot):,} hits, {df_bot['url'].nunique()} unique URLs")
    st.dataframe(df_bot_summary, use_container_width=True)

    fig6 = px.treemap(df_bot_summary, path=["url", "status"], values="hits",
                      title=f"{bot_choice} — URL & Status Breakdown", color="hits")
    st.plotly_chart(fig6, use_container_width=True)

    csv_export = df_bot_summary.to_csv(index=False).encode("utf-8")
    st.download_button(f"⬇️ Download {bot_choice} Data", csv_export,
                       f"{bot_choice.lower()}_report.csv", "text/csv")
else:
    st.info("Upload an Excel file containing User-Agent, URL, Status, and Date columns to begin analysis.")

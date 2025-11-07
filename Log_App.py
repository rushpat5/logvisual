import streamlit as st
import pandas as pd
import plotly.express as px
import re

st.set_page_config(page_title="Bot Intelligence Dashboard", page_icon="🤖", layout="wide")

# --- Bot Signatures (Top 50+ Most Common) ---
BOT_SIGNATURES = {
    # Search Engines
    "Googlebot": [r"googlebot", r"googleother", r"mediapartners-google"],
    "Bingbot": [r"bingbot", r"msnbot", r"adidxbot"],
    "Applebot": [r"applebot"],
    "YandexBot": [r"yandex(bot|images|video|news)"],
    "BaiduSpider": [r"baiduspider"],
    "DuckDuckBot": [r"duckduck(bot|go)"],
    "Yahoo! Slurp": [r"slurp"],
    "SogouSpider": [r"sogou"],
    "Exabot": [r"exabot"],
    "SeznamBot": [r"seznambot"],
    "Qwantify": [r"qwantify"],
    "Ezooms": [r"ezooms"],

    # AI / LLM Crawlers
    "GPTBot": [r"gptbot", r"openai[-_ ]?(collector|crawler|bot)"],
    "ChatGPT-User": [r"chatgpt[-_ ]?user"],
    "ClaudeBot": [r"claudebot", r"anthropic[-_ ]?ai"],
    "PerplexityBot": [r"perplexity(bot|ai)"],
    "YouBot": [r"you[-_ ]?search[-_ ]?bot"],
    "MistralAI": [r"mistral(ai)?"],
    "CohereAI": [r"cohere[-_ ]?ai"],
    "AI2Bot": [r"ai2bot"],
    "CCBot": [r"ccbot"],

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
    "CognitiveSEO": [r"cognitiveseo"],
    "RyteBot": [r"ryte"],

    # Social / Media
    "FacebookBot": [r"facebookexternalhit", r"facebot", r"meta[-_ ]?(external|bot)"],
    "TwitterBot": [r"twitterbot", r"xbot"],
    "LinkedInBot": [r"linkedinbot"],
    "PinterestBot": [r"pinterest"],
    "RedditBot": [r"reddit"],
    "Slackbot": [r"slackbot"],
    "DiscordBot": [r"discordbot"],
    "InstagramBot": [r"instagram"],
    "TikTokBot": [r"tiktok"],

    # Monitoring / Performance
    "GTMetrix": [r"gtmetrix"],
    "Pingdom": [r"pingdom"],
    "UptimeRobot": [r"uptimerobot"],
    "StatusCake": [r"statuscake"],
    "GooglePageSpeed": [r"pagespeed"],
    "CloudflareHealthCheck": [r"cloudflare[-_ ]?(healthcheck|diagnostics)"],

    # Miscellaneous
    "PetalBot": [r"petalbot"],
    "AmazonBot": [r"amazon(bot|adsbot)"],
    "CommonCrawl": [r"commoncrawl"],
    "DotBot": [r"dotbot"],
    "MegaIndexBot": [r"megaindex"],
    "ArchiveBot": [r"archive(-| )?org"],
    "Crawler": [r"crawler", r"spider", r"fetcher", r"scanner"]
}


# --- Bot Normalizer ---
def identify_bot(user_agent: str) -> str:
    ua = str(user_agent).lower()
    for bot_name, patterns in BOT_SIGNATURES.items():
        for p in patterns:
            if re.search(p, ua):
                return bot_name
    return "Unclassified Bot"


# --- Streamlit UI ---
st.title("🤖 Advanced Bot Traffic Intelligence Dashboard")
st.caption("Upload a consolidated Excel file with User-Agent data and URL hits")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if not all(col in df.columns for col in ["user-agent", "url", "status", "date"]):
        st.error("Missing required columns: user-agent, url, status, date")
        st.stop()

    df["bot_normalized"] = df["user-agent"].apply(identify_bot)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- Summary Metrics ---
    total_hits = len(df)
    unique_bots = df["bot_normalized"].nunique()
    unique_urls = df["url"].nunique()
    status_codes = df["status"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Hits", f"{total_hits:,}")
    c2.metric("Unique Bots", unique_bots)
    c3.metric("Unique URLs Hit", unique_urls)
    c4.metric("Status Codes", status_codes)

    # --- Aggregations ---
    bot_hits = df.groupby("bot_normalized").size().reset_index(name="hits").sort_values("hits", ascending=False)
    url_hits = df.groupby(["bot_normalized", "url"]).size().reset_index(name="hits").sort_values("hits", ascending=False)
    status_summary = df.groupby(["bot_normalized", "status"]).size().reset_index(name="count")

    # --- Visuals ---
    st.markdown("### 🔹 Total Hits per Bot")
    fig1 = px.bar(bot_hits, x="hits", y="bot_normalized", orientation="h",
                  title="Total Hits per Bot", color="bot_normalized", height=700)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 🔹 Traffic Share by Bot")
    fig2 = px.pie(bot_hits, names="bot_normalized", values="hits",
                  title="Traffic Share by Bot (Top 50+ Families)", hole=0.4)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🔹 Top URLs Hit per Bot")
    top_urls = url_hits.groupby("bot_normalized").apply(lambda x: x.nlargest(5, "hits")).reset_index(drop=True)
    fig3 = px.bar(top_urls, x="hits", y="url", color="bot_normalized",
                  title="Top URLs per Bot (Top 5 Each)", orientation="h", height=900)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 🔹 Status Code Distribution")
    fig4 = px.bar(status_summary, x="status", y="count", color="bot_normalized",
                  title="HTTP Status Distribution Across Bots", barmode="group", height=700)
    st.plotly_chart(fig4, use_container_width=True)

    # --- Deep Dive Section ---
    st.markdown("### 🎯 Deep Dive: Select a Bot")
    bot_choice = st.selectbox("Select a bot for detailed URL and status breakdown", sorted(df["bot_normalized"].unique()))
    df_bot = df[df["bot_normalized"] == bot_choice]
    st.write(f"**{bot_choice}** — {len(df_bot):,} hits, {df_bot['url'].nunique()} unique URLs")

    df_bot_summary = df_bot.groupby(["url", "status"]).size().reset_index(name="hits").sort_values("hits", ascending=False)
    st.dataframe(df_bot_summary, use_container_width=True)

    fig5 = px.treemap(df_bot_summary, path=["url", "status"], values="hits",
                      title=f"Treemap — {bot_choice} Activity by URL and Status", color="hits")
    st.plotly_chart(fig5, use_container_width=True)

    csv_export = df_bot_summary.to_csv(index=False).encode("utf-8")
    st.download_button(f"⬇️ Download {bot_choice} Breakdown", csv_export,
                       f"{bot_choice.lower()}_breakdown.csv", "text/csv")
else:
    st.info("Upload an Excel file with User-Agent data to analyze.")

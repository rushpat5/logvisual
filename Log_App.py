import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Advanced Bot Traffic Dashboard", page_icon="🤖", layout="wide")

# ----------------- Custom Styling -----------------
st.markdown("""
    <style>
        body {background-color: #0f1117; color: #e8e8e8;}
        .block-container {padding-top:2rem;}
        h1, h2, h3, h4, h5, h6 {color: #e8e8e8;}
        .metric-card {
            background: #1c1f2b;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            color: white;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Advanced Bot Traffic Intelligence Dashboard")

# ----------------- Upload File -----------------
uploaded_file = st.file_uploader("Upload consolidated Excel (from log analyzer export)", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [c.strip().capitalize() for c in df.columns]

    required_cols = ["Bot type", "User-agent", "Url", "Status", "Date"]
    if not all(col in df.columns for col in required_cols):
        st.error("Invalid file structure. Required columns: Bot Type, User-Agent, URL, Status, Date.")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df["Status"] = df["Status"].astype(str)
    df["Url"] = df["Url"].astype(str).replace("", "-")
    df["User-agent"] = df["User-agent"].astype(str)
    df["Hour"] = df["Date"].dt.hour

    # ----------------- Filters -----------------
    st.sidebar.header("🔍 Filters")
    min_date, max_date = df["Date"].min(), df["Date"].max()
    date_range = st.sidebar.date_input("Filter by Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    df = df[(df["Date"] >= pd.Timestamp(date_range[0])) & (df["Date"] <= pd.Timestamp(date_range[1]))]

    # ----------------- Key Metrics -----------------
    total_hits = len(df)
    unique_bots = df["User-agent"].nunique()
    unique_urls = df["Url"].nunique()
    unique_status = df["Status"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h3>Total Hits</h3><h2>{total_hits:,}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h3>Unique Bots</h3><h2>{unique_bots:,}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h3>Unique URLs Hit</h3><h2>{unique_urls:,}</h2></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h3>Status Codes Seen</h3><h2>{unique_status:,}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ----------------- Aggregations -----------------
    bot_counts = df.groupby("User-agent").size().reset_index(name="Hits").sort_values("Hits", ascending=False)
    bot_counts["% Share"] = round((bot_counts["Hits"] / bot_counts["Hits"].sum()) * 100, 2)

    url_counts = df.groupby(["User-agent", "Url"]).size().reset_index(name="Hits").sort_values("Hits", ascending=False)
    status_counts = df.groupby(["User-agent", "Status"]).size().reset_index(name="Count")

    # ----------------- Visuals -----------------
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(bot_counts, y="User-agent", x="Hits", orientation="h",
                      title="Total Hits per Bot", color="Hits", color_continuous_scale="viridis")
        fig1.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.pie(bot_counts, names="User-agent", values="Hits",
                      title="Traffic Share by Bot", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    df_daily = df.groupby(["Date", "Bot type"]).size().reset_index(name="Hits")
    fig3 = px.line(df_daily, x="Date", y="Hits", color="Bot type", title="📈 Daily Activity Trend by Bot Type",
                   markers=True, color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig3, use_container_width=True)

    # ----------------- Heatmap -----------------
    st.subheader("🌐 Hourly Activity Heatmap")
    df_heatmap = df.groupby(["Hour", "Bot type"]).size().reset_index(name="Hits")
    heatmap_data = df_heatmap.pivot(index="Bot type", columns="Hour", values="Hits").fillna(0)
    fig4 = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale="Cividis"))
    fig4.update_layout(title="Bot Activity by Hour of Day", xaxis_title="Hour", yaxis_title="Bot Type")
    st.plotly_chart(fig4, use_container_width=True)

    # ----------------- Deep Dive -----------------
    st.markdown("---")
    st.subheader("🎯 Deep Dive per Bot")

    bot_selected = st.selectbox("Select a Bot for Detailed Breakdown:", sorted(df["User-agent"].unique()))
    df_bot = df[df["User-agent"] == bot_selected]

    hit_count = df_bot.shape[0]
    urls_hit = df_bot["Url"].nunique()
    codes = ", ".join(df_bot["Status"].unique())

    st.markdown(f"**Bot:** `{bot_selected}`")
    st.markdown(f"- **Total Hits:** {hit_count:,}")
    st.markdown(f"- **Unique URLs Hit:** {urls_hit}")
    st.markdown(f"- **Status Codes Observed:** {codes}")

    df_bot_url = df_bot.groupby(["Url", "Status"]).size().reset_index(name="Hits").sort_values("Hits", ascending=False)
    st.dataframe(df_bot_url, use_container_width=True)

    fig5 = px.treemap(df_bot_url, path=["Url", "Status"], values="Hits", title="URL Breakdown for Selected Bot",
                      color="Hits", color_continuous_scale="Blues")
    st.plotly_chart(fig5, use_container_width=True)

    # ----------------- Status Code Overview -----------------
    st.subheader("📊 Status Code Distribution Across Bots")
    fig6 = px.bar(status_counts, x="Status", y="Count", color="User-agent",
                  title="HTTP Status Code Distribution", color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig6, use_container_width=True)

    # ----------------- Export Option -----------------
    csv_export = df_bot_url.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Selected Bot Breakdown CSV", csv_export, "bot_url_breakdown.csv", "text/csv")

    st.success("✅ Dashboard Ready for Client Presentation.")
else:
    st.info("Upload your consolidated Excel file to generate insights.")
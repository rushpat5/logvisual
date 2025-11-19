# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import re
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SEO Log Analyzer Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for "Pro" look
st.markdown("""
<style>
    /* Global styles */
    .main { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
    
    /* KPI Cards */
    .kpi-card {
        background-color: #262730;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-title { font-size: 14px; color: #b0b0b0; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-size: 28px; font-weight: bold; color: #ffffff; margin: 5px 0; }
    .kpi-delta { font-size: 14px; }
    .kpi-delta.pos { color: #00cc96; }
    .kpi-delta.neg { color: #ef553b; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    
    /* Charts */
    .js-plotly-plot .plotly .modebar { orientation: v; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & LOGIC MAPS
# -----------------------------------------------------------------------------

# Enhanced Bot Categories
BOT_MAP = {
    "Google": ["googlebot", "mediapartners-google", "adsbot-google"],
    "Bing/MS": ["bingbot", "bingpreview", "msnbot"],
    "AI & LLMs": ["gptbot", "chatgpt-user", "oai-searchbot", "claude", "anthropic", "perplexity", "ccbot", "bytespider", "mistral"],
    "SEO Tools": ["ahrefsbot", "semrushbot", "dotbot", "mj12bot", "rogerbot", "screamingfrog", "sitebulb"],
    "Social": ["facebookexternalhit", "twitterbot", "linkedinbot", "pinterest", "slackbot"],
    "Other Bots": ["yandex", "baiduspider", "duckduckbot", "sogou", "exabot", "slurp", "curl", "wget", "python-requests", "spider", "crawl"]
}

def categorize_agent(ua_string):
    """Returns (Category, SpecificName)"""
    if not isinstance(ua_string, str) or ua_string == "":
        return ("Unknown", "Unknown")
    
    ua = ua_string.lower()
    
    # 1. Check specific bot map
    for category, tokens in BOT_MAP.items():
        for token in tokens:
            if token in ua:
                return (category, token)
    
    # 2. Check browsers
    if "mozilla" in ua and any(x in ua for x in ["chrome", "safari", "firefox", "edge"]):
        return ("Browser/Human", "Browser")
        
    # 3. Generic Fallback
    if "bot" in ua or "crawl" in ua:
        return ("Other Bots", "Generic Bot")
        
    return ("Unknown", "Unknown")

# -----------------------------------------------------------------------------
# 3. DATA PROCESSING ENGINE
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_process_data(file_content):
    """
    Robust CSV/Log reader with automatic column mapping and feature engineering.
    """
    # 1. Read File
    try:
        decoded = file_content.getvalue().decode("utf-8", errors="replace")
        first_line = decoded.splitlines()[0] if decoded else ""
        sep = "\t" if "\t" in first_line else ","
        
        # Try C engine, fallback to Python engine
        try:
            df = pd.read_csv(io.BytesIO(file_content.getvalue()), sep=sep, engine="c", low_memory=False)
        except:
            df = pd.read_csv(io.BytesIO(file_content.getvalue()), sep=sep, engine="python", on_bad_lines="skip")
    except Exception as e:
        return None, f"Error reading file: {e}"

    # 2. Column Mapping (Smart Guessing)
    col_map = {}
    cols_lower = [c.lower() for c in df.columns]
    
    def find_col(keywords):
        for k in keywords:
            for i, col in enumerate(cols_lower):
                if k in col:
                    return df.columns[i]
        return None

    col_map['ts'] = find_col(["time", "date", "timestamp", "ts"])
    col_map['url'] = find_col(["request", "url", "path", "uri"])
    col_map['status'] = find_col(["status", "code", "sc-status"])
    col_map['ua'] = find_col(["user-agent", "user_agent", "ua", "agent"])
    col_map['method'] = find_col(["method", "verb"])
    col_map['size'] = find_col(["bytes", "size", "sc-bytes"])
    # Optional: Response time
    col_map['time_taken'] = find_col(["time-taken", "duration", "ms", "response_time"])

    if not col_map['ts'] or not col_map['url']:
        return None, "Could not auto-detect 'Timestamp' or 'URL' columns. Please ensure headers exist."

    # 3. Normalize Data
    # Timestamp
    df['timestamp'] = pd.to_datetime(df[col_map['ts']], errors='coerce', utc=True)
    if df['timestamp'].isna().mean() > 0.9: # If conversion failed, try manual formats
         for fmt in ["%d/%b/%Y:%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
            try:
                df['timestamp'] = pd.to_datetime(df[col_map['ts']], format=fmt, errors='coerce', utc=True)
                if df['timestamp'].notna().any(): break
            except: pass
    
    # Clean URL
    df['full_path'] = df[col_map['url']].astype(str)
    # Remove method if included in path (common in raw logs like 'GET /foo HTTP/1.1')
    df['full_path'] = df['full_path'].apply(lambda x: x.split(' ')[1] if len(x.split(' ')) > 1 and x.startswith(('GET', 'POST')) else x)
    
    # Feature Engineering
    df['path_clean'] = df['full_path'].apply(lambda x: x.split('?')[0].split('#')[0])
    df['query_string'] = df['full_path'].apply(lambda x: x.split('?')[1] if '?' in x else "")
    df['has_params'] = df['query_string'].str.len() > 0
    df['status'] = pd.to_numeric(df[col_map['status']], errors='coerce').fillna(0).astype(int)
    df['status_cls'] = (df['status'] // 100).astype(str) + "xx"
    
    # UA Classification
    ua_col = col_map.get('ua')
    if ua_col:
        df['user_agent'] = df[ua_col].astype(str)
        df[['bot_category', 'bot_name']] = df['user_agent'].apply(lambda x: pd.Series(categorize_agent(x)))
    else:
        df['bot_category'] = "Unknown"
        df['bot_name'] = "Unknown"

    # Static vs Content
    static_exts = ('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff', '.woff2', '.ttf', '.eot', '.map', '.webp')
    df['is_static'] = df['path_clean'].str.lower().str.endswith(static_exts)
    
    # Sections (Top directory)
    df['section'] = df['path_clean'].apply(lambda x: x.strip('/').split('/')[0] if len(x.strip('/').split('/')) > 0 else 'Root')
    
    # Dates
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_name'] = df['timestamp'].dt.day_name()

    return df.sort_values('timestamp'), None

# -----------------------------------------------------------------------------
# 4. HELPER VISUALIZATIONS
# -----------------------------------------------------------------------------

def kpi_card(col, title, value, delta=None, delta_txt=""):
    """Renders a custom HTML KPI card"""
    delta_html = ""
    if delta is not None:
        color_class = "pos" if delta >= 0 else "neg"
        delta_html = f'<div class="kpi-delta {color_class}">{delta_txt}</div>'
    
    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)

def plot_sparkline(df_ts):
    """Simple sparkline for dashboard"""
    if df_ts.empty: return None
    fig = px.area(df_ts, x='date', y='count', template="plotly_dark")
    fig.update_traces(line_color='#ff4b4b', fill_color='rgba(255, 75, 75, 0.2)')
    fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0), height=80, 
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# -----------------------------------------------------------------------------
# 5. MAIN APP LAYOUT
# -----------------------------------------------------------------------------

st.title("🚀 SEO Log Analyzer Pro")
st.markdown("Transform raw server logs into **Actionable SEO Intelligence**.")

# SIDEBAR
with st.sidebar:
    st.header("1. Upload & Config")
    uploaded_file = st.file_uploader("Upload Log File (CSV/TSV)", type=['csv', 'tsv', 'txt'])
    
    if uploaded_file:
        df_raw, error = load_and_process_data(uploaded_file)
        
        if error:
            st.error(error)
            st.stop()
            
        st.success(f"Loaded {len(df_raw):,} rows.")
        
        st.divider()
        st.header("2. Filters")
        
        # Date Filter
        min_date, max_date = df_raw['date'].min(), df_raw['date'].max()
        date_range = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        # Bot Filter
        all_cats = sorted(df_raw['bot_category'].unique())
        sel_cats = st.multiselect("Bot Categories", all_cats, default=[c for c in all_cats if c != "Browser/Human"])
        
        # Status Filter
        all_status = sorted(df_raw['status_cls'].unique())
        sel_status = st.multiselect("Status Class", all_status, default=all_status)
        
        # Exclude Static
        exclude_static = st.checkbox("Exclude Static Files (.css, .js, etc.)", value=True)
        
        # Apply Filters
        mask = (df_raw['date'] >= date_range[0]) & (df_raw['date'] <= date_range[1])
        mask &= df_raw['bot_category'].isin(sel_cats)
        mask &= df_raw['status_cls'].isin(sel_status)
        if exclude_static:
            mask &= (~df_raw['is_static'])
            
        df = df_raw[mask].copy()
        
        st.divider()
        st.caption(f"Analysis View: {len(df):,} hits")
    else:
        st.info("Upload a file to begin.")
        st.stop()

# TABS LAYOUT
tab_overview, tab_budget, tab_status, tab_ai, tab_explorer = st.tabs([
    "📊 Executive Dashboard", 
    "💰 Crawl Budget & Efficiency", 
    "🩺 Status & Errors", 
    "🤖 AI & Bot Intelligence", 
    "🔎 Data Explorer"
])

# --- TAB 1: EXECUTIVE DASHBOARD ---
with tab_overview:
    # Top Level KPIs
    c1, c2, c3, c4 = st.columns(4)
    
    total_hits = len(df)
    unique_urls = df['path_clean'].nunique()
    error_hits = len(df[df['status'] >= 400])
    error_rate = (error_hits / total_hits * 100) if total_hits > 0 else 0
    
    # Calculate simple delta (vs first half of period) - rudimentary but effective visual
    midpoint = df['timestamp'].min() + (df['timestamp'].max() - df['timestamp'].min()) / 2
    prev_period = df[df['timestamp'] < midpoint]
    curr_period = df[df['timestamp'] >= midpoint]
    
    hits_delta = len(curr_period) - len(prev_period)
    
    kpi_card(c1, "Total Bot Hits", f"{total_hits:,}", hits_delta, "vs prev period")
    kpi_card(c2, "Unique URLs Crawled", f"{unique_urls:,}")
    kpi_card(c3, "Error Rate (4xx/5xx)", f"{error_rate:.2f}%", -error_rate, "lower is better")
    kpi_card(c4, "Active Bot Agents", f"{df['bot_name'].nunique()}", None)

    # Visuals
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("Crawl Volume Over Time")
        # Resample for trend
        ts_data = df.set_index('timestamp').resample('H').size().reset_index(name='hits')
        fig_trend = px.area(ts_data, x='timestamp', y='hits', title="Hourly Crawl Volume", template="plotly_dark", color_discrete_sequence=['#4DA6FF'])
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_side:
        st.subheader("Bot Category Share")
        pie_data = df['bot_category'].value_counts().reset_index()
        fig_pie = px.donut(pie_data, names='bot_category', values='count', hole=0.4, template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Actionable Insights Generator
    st.subheader("⚡ Automated SEO Insights")
    insights = []
    if error_rate > 5:
        insights.append(f"🔴 **Critical:** Error rate is high ({error_rate:.1f}%). Check the 'Status & Errors' tab immediately.")
    if df['has_params'].mean() > 0.3:
        insights.append(f"🟠 **Warning:** 30%+ of crawled URLs have query parameters. Ensure these aren't wasting crawl budget.")
    if "Google" in df['bot_category'].values:
        g_hits = len(df[df['bot_category']=="Google"])
        if g_hits < 100:
            insights.append("🟡 **Observation:** Googlebot activity is very low. Check if the site is new or blocked.")
    if "AI & LLMs" in df['bot_category'].unique():
        ai_hits = len(df[df['bot_category']=="AI & LLMs"])
        insights.append(f"🔵 **Info:** AI Bots (GPT/Claude) made {ai_hits:,} requests. Check the 'AI Intelligence' tab.")

    if insights:
        for i in insights: st.markdown(i)
    else:
        st.success("✅ No critical anomalies detected in this sample.")

# --- TAB 2: CRAWL BUDGET ---
with tab_budget:
    st.markdown("### Are bots wasting time on low-value pages?")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Top Sections Crawled")
        section_counts = df['section'].value_counts().head(10)
        fig_sec = px.bar(section_counts, orientation='h', title="Hits by Site Section", template="plotly_dark")
        st.plotly_chart(fig_sec, use_container_width=True)
        
    with c2:
        st.subheader("Waste Analysis: Parameters")
        param_df = df[df['has_params']]
        if not param_df.empty:
            # Group by parameter keys (naive split)
            param_hits = param_df.groupby('path_clean').size().reset_index(name='hits').sort_values('hits', ascending=False).head(10)
            fig_param = px.bar(param_hits, x='hits', y='path_clean', title="Top Parametrized URLs (Potential Trap)", template="plotly_dark", color_discrete_sequence=['#FF9900'])
            st.plotly_chart(fig_param, use_container_width=True)
        else:
            st.info("No parametrized URLs found in filtered data.")

    st.markdown("---")
    st.subheader("Top 20 URLs consuming Crawl Budget")
    top_urls = df.groupby(['path_clean', 'status', 'bot_category']).size().reset_index(name='hits').sort_values('hits', ascending=False).head(20)
    st.dataframe(top_urls, use_container_width=True)

# --- TAB 3: STATUS & ERRORS ---
with tab_status:
    col_err_1, col_err_2 = st.columns([3, 1])
    
    with col_err_1:
        st.subheader("Status Code Breakdown")
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']
        status_counts['color'] = status_counts['status'].apply(lambda x: 'red' if x >= 400 else ('green' if x < 300 else 'orange'))
        
        fig_status = px.bar(status_counts, x='status', y='count', color='color', title="Response Codes Distribution", template="plotly_dark")
        st.plotly_chart(fig_status, use_container_width=True)

    with col_err_2:
        st.subheader("Quick Stats")
        st.metric("5xx Server Errors", len(df[df['status'] >= 500]))
        st.metric("404 Not Found", len(df[df['status'] == 404]))
        st.metric("3xx Redirects", len(df[(df['status'] >= 300) & (df['status'] < 400)]))

    # Error Heatmap
    st.subheader("🔥 Error Heatmap (When do errors happen?)")
    err_df = df[df['status'] >= 400].copy()
    if not err_df.empty:
        heatmap_data = err_df.groupby(['day_name', 'hour']).size().reset_index(name='count')
        # Ensure sorting
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data['day_name'] = pd.Categorical(heatmap_data['day_name'], categories=days_order, ordered=True)
        
        fig_heat = px.density_heatmap(heatmap_data, x='hour', y='day_name', z='count', nbinsx=24, 
                                      title="Error Intensity by Day & Hour", template="plotly_dark", color_continuous_scale="Viridis")
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.subheader("Top Broken Links (404s)")
        top_404 = df[df['status'] == 404]['path_clean'].value_counts().head(10).reset_index(name='hits')
        st.dataframe(top_404, use_container_width=True)
    else:
        st.success("Great! No 4xx/5xx errors detected in the filtered data.")

# --- TAB 4: AI & BOT INTELLIGENCE ---
with tab_ai:
    st.markdown("### Who is crawling you? (Google vs AI vs Tools)")
    
    # Pivot table for Bot Category over time
    pivot_bots = df.groupby([pd.Grouper(key='timestamp', freq='H'), 'bot_category']).size().reset_index(name='hits')
    fig_bots = px.line(pivot_bots, x='timestamp', y='hits', color='bot_category', title="Bot Activity Trends", template="plotly_dark")
    st.plotly_chart(fig_bots, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("AI Scraper Analysis")
        ai_df = df[df['bot_category'] == "AI & LLMs"]
        if not ai_df.empty:
            fig_ai = px.bar(ai_df['bot_name'].value_counts().reset_index(), x='bot_name', y='count', 
                            title="Top AI Bots (Training & Search)", template="plotly_dark", color_discrete_sequence=['#00cc96'])
            st.plotly_chart(fig_ai, use_container_width=True)
            st.info("💡 **Tip:** Use `robots.txt` to block `GPTBot` if you don't want your content used for LLM training.")
        else:
            st.write("No AI bot activity detected.")
            
    with c2:
        st.subheader("SEO Tool Activity")
        tool_df = df[df['bot_category'] == "SEO Tools"]
        if not tool_df.empty:
            st.dataframe(tool_df['bot_name'].value_counts(), use_container_width=True)
        else:
            st.write("No SEO tool activity detected.")

# --- TAB 5: DATA EXPLORER ---
with tab_explorer:
    st.subheader("Raw Data Inspector")
    st.dataframe(df[['timestamp', 'status', 'bot_name', 'method', 'full_path']].sort_values('timestamp', ascending=False), use_container_width=True)
    
    st.download_button(
        label="Download Filtered Data (CSV)",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_seo_logs.csv',
        mime='text/csv',
    )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("SEO Log Analyzer Pro | Built with Streamlit & Plotly")

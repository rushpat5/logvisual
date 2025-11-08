# Website Log Analyzer (Streamlit)

A Streamlit app to analyze server logs with a focus on AI-bot traffic (GPTBot, OAI-SearchBot, ChatGPT-User, Claude-*, Bytespider, etc.). Upload your CSV (e.g., `detailed_hits.csv`) and the app will auto-detect columns, allow remapping, and produce interactive visualizations.

## Files
- `app.py` — main Streamlit application
- `utils.py` — helpers for guessing columns, parsing timestamps, classifying user agents
- `requirements.txt` — Python dependencies

## Quick start (local)
1. Create a Python 3.9+ virtualenv.
2. Install requirements:
   ```bash
   pip install -r requirements.txt

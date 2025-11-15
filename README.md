# Website Log Analyzer (Streamlit) — Revamped

A Streamlit app to analyze server logs with a focus on AI and bot traffic (GPTBot, OAI-SearchBot, ChatGPT-User, Claude-*, Bytespider, etc.). Upload your CSV/TSV log export and the app will auto-suggest columns, allow remapping, and provide interactive visualizations and heuristics.

## Features
- Robust column guessing and manual remapping
- Flexible timestamp parsing (epoch ms/s, ISO, CLF)
- AI vs traditional bot classification and grouping
- Static asset detection (configurable)
- Time-series charts, hourly heatmap, status code analysis
- Heuristics: burst detection, crawler-like agents, error exposure
- Export filtered data (CSV / JSON / Parquet)
- Scales to large files (chunked read option)
- CI workflow (tests + lint)

## Files
- `app.py` — main Streamlit application
- `utils.py` — helper functions for parsing and classification
- `requirements.txt` — dependencies
- `.github/workflows/ci.yml` — CI pipeline (tests + lint)
- `tests/` — unit tests (examples)

## Usage
1. Create a virtualenv and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

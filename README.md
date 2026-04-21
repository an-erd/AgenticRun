# 🏃 AgenticRun

**Garmin running history — analysis, trends, and coaching recommendations**

A local Python + Streamlit analytics app that turns 10 years of Garmin data 
into explainable performance insight. Deterministic analytics first, 
LLM as explainer second.

---

## What it does

- Imports Garmin FIT, CSV, and ZIP archives at scale
- Classifies sessions by intensity domain (threshold, VO2, easy/recovery)
- Compares each run against prior sessions in the same training family
- Derives deterministic signals: trend, execution quality, fatigue, fitness
- Generates structured coaching recommendations with a dominant rule ID
- Uses OpenAI to summarize findings in natural language — grounded, 
  not invented

## Architecture principle

Import → Classify → Analyse → Compare → Recommend → LLM Explain

The LLM receives structured deterministic findings as context.
It summarizes and phrases — it does not decide.

## Tech stack

- Python
- Streamlit
- SQLite
- OpenAI API
- Garmin FIT file parsing
- Built with Cursor

## Screenshots

[Screenshot Dashboard]
[Screenshot Interval Analysis]

## Setup

```bash
git clone https://github.com/an-erd/AgenticRun.git
cd AgenticRun
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
streamlit run app.py
```

## Key numbers (real data)

- 1,216 running activities
- 10 years of history (2016–2026)
- 14,700 FIT files processed in bulk import
- 0 errors

## License

MIT
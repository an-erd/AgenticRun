# AgenticRun Prototype

A pragmatic Python prototype for an **agentic running analysis system**.

It is intentionally built as a **modular workflow with specialized agents** instead of a single opaque "super agent".

## What it does

- imports Garmin CSV exports
- normalizes session-level metrics
- stores runs in SQLite
- classifies the session type
- compares the run to similar historical sessions
- generates a conservative training recommendation
- optionally asks an LLM to turn the structured output into a readable summary

## Architecture

The prototype uses these modules:

- `ImportAgent` – reads Garmin CSV files and extracts one normalized run per file
- `SessionAnalysisAgent` – classifies the run and calculates basic signals
- `TrendAgent` – compares the run against historical runs in the database
- `RecommendationAgent` – derives the next-step recommendation
- `OutputAgent` – writes CSV/JSON output and optional Markdown summary

There is **no heavy framework dependency** and no mandatory orchestration framework.
The pipeline is coordinated by a simple Python entrypoint and a shared `RunState` object.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py ingest --input ./data
```

This will:
- create `agenticrun.db`
- parse all CSV files in `./data`
- write outputs to `./out`

## Optional LLM summaries

Set your API key in `.env`:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Then run:

```bash
python main.py ingest --input ./data --llm
```

If no API key is present, the system still works and falls back to deterministic summaries.

## Folder structure

```text
AgenticRunPrototype/
├── main.py
├── requirements.txt
├── .env.example
├── data/
├── out/
└── agenticrun/
    ├── agents/
    ├── core/
    ├── services/
    └── utils/
```

## Current scope

This is a **Prototype / MVP 1.5**:
- works with the sample Garmin CSV variants you uploaded
- uses conservative rules
- does not do medical evaluation
- does not write back to Garmin or adjust external plans automatically

## Suggested next steps

1. refine session classification
2. add workout-intent recognition from interval patterns
3. add weekly load logic and fatigue guardrails
4. improve LLM prompt templates
5. add a simple local UI with Streamlit or FastAPI

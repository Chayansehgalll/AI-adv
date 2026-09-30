# Groq Agent Systems

This project contains two separate tool-using systems built with Groq, without LangGraph:

- **Single model:** one LLM chooses between Tavily web search and a safe arithmetic calculator. It makes at most four tool calls.
- **Multi-agent:** a manager LLM delegates to a web-search helper or a maths helper. The manager makes at most four delegations, and helper responses are appended to `manager_notes.jsonl`.

Both programs report prompt, completion, and total token usage. Each has the same default test query about NVIDIA and Microsoft revenue. Run them separately with the same question to compare token usage.

## Setup

Use Python 3.13 or newer. From this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Groq and Tavily API keys to `.env`. The default model is `openai/gpt-oss-120b`; set `GROQ_MODEL` in `.env` to use another Groq model that supports tool calling.

## Run

```powershell
python single_agent.py
python multi_agent.py
```

Press Enter to use the built-in test query, or type a different question. The multi-agent helper responses are saved in `manager_notes.txt`. `main.py` remains the original starter file.

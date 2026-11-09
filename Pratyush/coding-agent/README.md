# Java coding agent

The agent asks for a bug description, then uses Gemini function calls to search
Java files, inspect matching lines, and replace a selected line.

## Setup

Add your Gemini API key to `.env`:

```env
GOOGLE_API_KEY=your-key
```

Then run from this directory:

```powershell
uv sync
uv run main.py
```

The tools are implemented in `tools.py`. They only read and modify `.java`
files inside this project directory.
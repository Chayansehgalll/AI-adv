import json
import os
import re

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient


# Same hard cap as before. The single LLM can call tools at most 4 times.
MAX_ATTEMPTS = 4
TEST_QUERY = (
    "Find the latest population estimates for Paris and London. "
    "Include the year and source link for each. Make sure both use "
    "comparable boundaries, such as city proper or metro area. Use the "
    "calculator to find the population difference and combined population. "
    "Explain the result simply."
)

# ---------------------------------------------------------------------------
# KEY DIFFERENCE FROM THE PREVIOUS VERSION:
# The old code had three separate tool lists (SEARCH_TOOL for the search
# helper, CALCULATOR_TOOL for the maths helper, MANAGER_TOOLS for the
# manager). This version collapses everything into ONE flat list. A single
# LLM gets both tools directly — no manager, no delegation, no helpers.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current facts and sources.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate using basic arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def calculator(expression):
    # Identical to the previous version. Regex ensures only safe math chars,
    # length cap prevents abuse, eval() is sandboxed with no built-ins.
    if len(expression) > 100 or not re.fullmatch(r"[0-9+*/().%\-\s]+", expression):
        raise ValueError("Use only numbers and basic arithmetic symbols.")
    if "**" in expression:
        raise ValueError("Powers are not supported.")
    return eval(expression, {"__builtins__": {}}, {})


# ---------------------------------------------------------------------------
# NEW: Single tool dispatcher.
# The old code had separate ask_search_llm() and ask_maths_llm() functions,
# each spinning up their own LLM call. This replaces both with one function
# that just runs the tool directly in Python — no sub-LLM involved.
# The LLM says "I want to call web_search with query X" and this function
# actually executes it.
# ---------------------------------------------------------------------------
def run_tool(name, arguments, tavily):
    if name == "web_search":
        # Hit the Tavily API, get up to 5 results, return as JSON string.
        results = tavily.search(query=arguments["query"], max_results=5)
        return json.dumps(results)
    if name == "calculator":
        # Run the sandboxed Python calculator and return the result as a string.
        return str(calculator(arguments["expression"]))
    # Fallback in case the LLM hallucinates a tool name that doesn't exist.
    return "Unknown tool."


def ask_llm(query):
    # Everything is initialized inside this function now. The old code did
    # this in __main__ and passed client/tavily/model around as arguments.
    # Simpler, but means a new client is created every time you call ask_llm().
    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    usage = {"prompt": 0, "completion": 0, "total": 0}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful assistant. Use web_search for current facts and "
                "calculator for math. Include source URLs. You can use tools at most "
                "four times."
            ),
        },
        {"role": "user", "content": query},
    ]

    attempts = 0

    # Same loop structure as the old manager, but simpler because there's
    # only one LLM. It calls tools directly instead of delegating to helpers.
    while attempts < MAX_ATTEMPTS:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0.1
        )

        # Token tracking is done inline here instead of via a separate
        # add_tokens() helper function. Same result, just less indirection.
        usage["prompt"] += response.usage.prompt_tokens
        usage["completion"] += response.usage.completion_tokens
        usage["total"] += response.usage.total_tokens

        message = response.choices[0].message

        # Append the assistant's message to conversation history so the LLM
        # remembers what it said and what tools it called on previous turns.
        messages.append(message.model_dump(exclude_none=True))

        # No tool calls = the LLM thinks it has enough info and wrote a
        # final text answer. Return it along with token usage.
        # NOTE: This returns a tuple (answer, usage), unlike the old code
        # which returned just the answer string.
        if not message.tool_calls:
            return message.content, usage

        # The LLM wants to use one or more tools. Execute each one.
        for tool_call in message.tool_calls:
            if attempts >= MAX_ATTEMPTS:
                result = "Tool call limit reached."
            else:
                attempts += 1
                arguments = json.loads(tool_call.function.arguments)
                try:
                    # run_tool() handles both web_search and calculator.
                    # The old code routed to entirely separate LLM-powered
                    # helper functions here. Now it's just a direct Python call.
                    result = run_tool(tool_call.function.name, arguments, tavily)
                except Exception as error:
                    # Catch calculator errors (bad expression, etc.) so the
                    # pipeline doesn't crash. The error message gets fed back
                    # to the LLM so it can try again or explain the failure.
                    result = f"Tool error: {error}"

            # Same as before: the API requires a tool result message for
            # every tool_call, matched by tool_call_id. Without this the
            # next API call will throw an error.
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )

    # --- FALLBACK: all 4 attempts used up, LLM never stopped calling tools ---
    # KEY DIFFERENCE from the old code: the old version created a completely
    # fresh conversation with a new system prompt and no tools. This version
    # just reuses the existing conversation history but omits the `tools`
    # parameter. Without `tools=TOOLS`, the LLM literally cannot request
    # any more tool calls — it has no choice but to write a text answer.
    response = client.chat.completions.create(model=model, messages=messages)
    usage["prompt"] += response.usage.prompt_tokens
    usage["completion"] += response.usage.completion_tokens
    usage["total"] += response.usage.total_tokens
    return response.choices[0].message.content, usage


if __name__ == "__main__":
    question = input("Ask a question (Enter uses the test question): ").strip()
    if not question:
        question = TEST_QUERY

    # Unpack the tuple returned by ask_llm(). The old code passed a shared
    # dict for token tracking; this version returns it directly.
    answer, tokens = ask_llm(question)
    print("\nAnswer:\n", answer)
    print("\nToken usage:", tokens)
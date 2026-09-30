import json
import os
import re

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient


MAX_ATTEMPTS = 4
TEST_QUERY = (
    "Find the latest population estimates for Paris and London. "
    "Include the year and source link for each. Make sure both use "
    "comparable boundaries, such as city proper or metro area. Use the "
    "calculator to find the population difference and combined population. "
    "Explain the result simply."
)

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
    # Only allow numbers and arithmetic symbols; built-ins are disabled too.
    if len(expression) > 100 or not re.fullmatch(r"[0-9+*/().%\-\s]+", expression):
        raise ValueError("Use only numbers and basic arithmetic symbols.")
    if "**" in expression:
        raise ValueError("Powers are not supported.")
    return eval(expression, {"__builtins__": {}}, {})


def run_tool(name, arguments, tavily):
    if name == "web_search":
        results = tavily.search(query=arguments["query"], max_results=5)
        return json.dumps(results)
    if name == "calculator":
        return str(calculator(arguments["expression"]))
    return "Unknown tool."


def ask_llm(query):
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
    while attempts < MAX_ATTEMPTS:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0.1
        )
        usage["prompt"] += response.usage.prompt_tokens
        usage["completion"] += response.usage.completion_tokens
        usage["total"] += response.usage.total_tokens
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content, usage

        for tool_call in message.tool_calls:
            if attempts >= MAX_ATTEMPTS:
                result = "Tool call limit reached."
            else:
                attempts += 1
                arguments = json.loads(tool_call.function.arguments)
                try:
                    result = run_tool(tool_call.function.name, arguments, tavily)
                except Exception as error:
                    result = f"Tool error: {error}"
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )

    response = client.chat.completions.create(model=model, messages=messages)
    usage["prompt"] += response.usage.prompt_tokens
    usage["completion"] += response.usage.completion_tokens
    usage["total"] += response.usage.total_tokens
    return response.choices[0].message.content, usage


if __name__ == "__main__":
    question = input("Ask a question (Enter uses the test question): ").strip()
    if not question:
        question = TEST_QUERY
    answer, tokens = ask_llm(question)
    print("\nAnswer:\n", answer)
    print("\nToken usage:", tokens)
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

SEARCH_TOOL = {
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
}
CALCULATOR_TOOL = {
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
}
MANAGER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_search_llm",
            "description": "Give a web research task to the search helper.",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_maths_llm",
            "description": "Give a calculation task to the maths helper.",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        },
    },
]


def calculator(expression):
    if len(expression) > 100 or not re.fullmatch(r"[0-9+*/().%\-\s]+", expression):
        raise ValueError("Use only numbers and basic arithmetic symbols.")
    if "**" in expression:
        raise ValueError("Powers are not supported.")
    return eval(expression, {"__builtins__": {}}, {})


def add_tokens(usage, response):
    usage["prompt"] += response.usage.prompt_tokens
    usage["completion"] += response.usage.completion_tokens
    usage["total"] += response.usage.total_tokens


def search_web(tavily, query):
    return json.dumps(tavily.search(query=query, max_results=5))


def save_note(helper, task, answer):
    with open("manager_notes.txt", "a", encoding="utf-8") as file:
        file.write(f"\nHelper: {helper}\nTask: {task}\nResponse: {answer}\n")


def ask_search_llm(client, tavily, model, task, usage):
    messages = [
        {"role": "system", "content": "Search the web and return facts with source URLs."},
        {"role": "user", "content": task},
    ]
    response = client.chat.completions.create(
        model=model, messages=messages, tools=[SEARCH_TOOL], temperature=0.1
    )
    add_tokens(usage, response)
    message = response.choices[0].message

    if message.tool_calls:
        results = []
        for tool_call in message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            results.append(search_web(tavily, arguments["query"]))
        answer = "\n".join(results)
    else:
        answer = message.content or "No search answer was returned."

    save_note("Search", task, answer)
    return answer


def ask_maths_llm(client, model, task, usage):
    messages = [
        {"role": "system", "content": "Use the calculator for arithmetic. Explain your formulas."},
        {"role": "user", "content": task},
    ]
    response = client.chat.completions.create(
        model=model, messages=messages, tools=[CALCULATOR_TOOL], temperature=0.1
    )
    add_tokens(usage, response)
    message = response.choices[0].message

    if message.tool_calls:
        results = []
        for tool_call in message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            try:
                result = str(calculator(arguments["expression"]))
            except Exception as error:
                result = f"Calculator error: {error}"
            results.append(f"{arguments['expression']} = {result}")
        answer = "\n".join(results)
    else:
        answer = message.content or "No maths answer was returned."

    save_note("Maths", task, answer)
    return answer


def ask_manager_llm(client, tavily, model, query, usage):
    messages = [
        {
            "role": "system",
            "content": (
                "You are the manager. Use ask_search_llm for web research and "
                "ask_maths_llm for calculations. Use helper responses as notes and "
                "combine them into a sourced final answer. Delegate at most four times."
            ),
        },
        {"role": "user", "content": query},
    ]
    delegations = 0
    helper_notes = []

    while delegations < MAX_ATTEMPTS:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=MANAGER_TOOLS, temperature=0.1
        )
        add_tokens(usage, response)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content or "No manager answer was returned."

        for tool_call in message.tool_calls:
            if delegations >= MAX_ATTEMPTS:
                result = "Delegation limit reached."
            else:
                delegations += 1
                arguments = json.loads(tool_call.function.arguments)
                task = arguments["task"]
                if tool_call.function.name == "ask_search_llm":
                    result = ask_search_llm(client, tavily, model, task, usage)
                    helper_name = "Search helper"
                else:
                    result = ask_maths_llm(client, model, task, usage)
                    helper_name = "Maths helper"
                helper_notes.append(f"{helper_name} task: {task}\nResult: {result}")
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )

    final_messages = [
        {
            "role": "system",
            "content": (
                "Write the final answer using the helper notes. Do not request more "
                "helpers or tools. If a fact is missing, say so clearly."
            ),
        },
        {"role": "user", "content": query},
        {"role": "user", "content": "Helper notes:\n" + "\n\n".join(helper_notes)},
    ]
    response = client.chat.completions.create(
        model=model, messages=final_messages, temperature=0.1
    )
    add_tokens(usage, response)
    return response.choices[0].message.content or "The manager could not finish."


if __name__ == "__main__":
    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    question = input("Ask a question (Enter uses the test question): ").strip()
    if not question:
        question = TEST_QUERY
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    answer = ask_manager_llm(client, tavily, model, question, tokens)
    print("\nAnswer:\n", answer)
    print("\nToken usage:", tokens)
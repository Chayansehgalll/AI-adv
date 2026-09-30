import json
import os
import re

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient


# Hard cap on how many times the manager can delegate to helpers.
# Without this, the manager could loop forever, draining your API credits.
MAX_ATTEMPTS = 4
TEST_QUERY = (
    "Find the latest population estimates for Paris and London. "
    "Include the year and source link for each. Make sure both use "
    "comparable boundaries, such as city proper or metro area. Use the "
    "calculator to find the population difference and combined population. "
    "Explain the result simply."
)

# ---------------------------------------------------------------------------
# TOOL SCHEMAS
# These don't execute anything. They are JSON descriptions handed to the LLM
# so it knows what tools exist and what arguments they expect. When the LLM
# "calls" a tool, it just outputs structured JSON — your Python code does
# the real work.
# ---------------------------------------------------------------------------

# Given to the search helper so it can request web searches.
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

# Given to the maths helper so it can request calculations.
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

# Given to the manager so it can delegate work to the two helpers.
# The manager never searches or calculates directly — it calls these
# "tools" and your Python code routes them to the real helper functions.
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
    # Reject anything over 100 chars to prevent abuse.
    # The regex ensures ONLY digits, basic operators, parens, dots, and spaces
    # are allowed. This is critical because we use eval() below.
    if len(expression) > 100 or not re.fullmatch(r"[0-9+*/().%\-\s]+", expression):
        raise ValueError("Use only numbers and basic arithmetic symbols.")

    # Belt-and-suspenders: block exponentiation even though the regex
    # already prevents most forms of it.
    if "**" in expression:
        raise ValueError("Powers are not supported.")

    # eval() executes the string as Python code. DANGEROUS in general,
    # but safe here because:
    #   1. The regex above guarantees only math characters exist.
    #   2. {"__builtins__": {}} strips ALL Python built-in functions,
    #      so even a bypass couldn't call __import__, open(), etc.
    #   3. The empty {} for locals means no variables are in scope.
    return eval(expression, {"__builtins__": {}}, {})


def add_tokens(usage, response):
    # Accumulates token counts from each LLM call into a shared dict.
    # The dict is passed by reference, so it updates in place across
    # all calls (manager + helpers).
    usage["prompt"] += response.usage.prompt_tokens
    usage["completion"] += response.usage.completion_tokens
    usage["total"] += response.usage.total_tokens


def search_web(tavily, query):
    # Hits the Tavily search API, gets up to 5 results, and converts
    # the response to a JSON string so it can be fed back to the LLM as text.
    return json.dumps(tavily.search(query=query, max_results=5))


def save_note(helper, task, answer):
    # Appends every helper interaction to a log file for debugging/auditing.
    # "a" mode = append, so previous runs aren't overwritten.
    with open("manager_notes.txt", "a", encoding="utf-8") as file:
        file.write(f"\nHelper: {helper}\nTask: {task}\nResponse: {answer}\n")


def ask_search_llm(client, tavily, model, task, usage):
    # Fresh two-message conversation for the search helper.
    # It has no memory of previous tasks — each call is independent.
    messages = [
        {"role": "system", "content": "Search the web and return facts with source URLs."},
        {"role": "user", "content": task},
    ]

    # temperature=0.1 keeps output factual and deterministic.
    # The SEARCH_TOOL is the only tool available to this helper.
    response = client.chat.completions.create(
        model=model, messages=messages, tools=[SEARCH_TOOL], temperature=0.1
    )
    add_tokens(usage, response)
    message = response.choices[0].message

    # The LLM doesn't search the web itself. It outputs a tool_call request
    # like {"query": "Paris population 2024"}. We intercept that here,
    # run the actual Tavily search, and collect the results.
    if message.tool_calls:
        results = []
        for tool_call in message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            results.append(search_web(tavily, arguments["query"]))
        answer = "\n".join(results)
    else:
        # If the LLM didn't call the tool (thought it already knew),
        # just use whatever text it generated.
        answer = message.content or "No search answer was returned."

    save_note("Search", task, answer)
    return answer


def ask_maths_llm(client, model, task, usage):
    # Same pattern as the search helper but with the calculator tool.
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
                # Run the actual Python calculator function.
                result = str(calculator(arguments["expression"]))
            except Exception as error:
                # If the expression is invalid, return an error message
                # instead of crashing the whole pipeline.
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

    # Main delegation loop. Each iteration sends the full conversation
    # (including previous tool results) back to the manager LLM so it
    # can decide what to do next.
    while delegations < MAX_ATTEMPTS:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=MANAGER_TOOLS, temperature=0.1
        )
        add_tokens(usage, response)
        message = response.choices[0].message

        # model_dump() converts the Pydantic message object to a plain dict.
        # exclude_none=True drops null fields so the API doesn't reject them.
        # This appends the assistant's reply to the conversation history.
        messages.append(message.model_dump(exclude_none=True))

        # If the manager didn't call any tools, it's done — return its text.
        # This is the normal/happy exit path.
        if not message.tool_calls:
            return message.content or "No manager answer was returned."

        # The manager called one or more helper tools. Process each one.
        for tool_call in message.tool_calls:
            if delegations >= MAX_ATTEMPTS:
                result = "Delegation limit reached."
            else:
                delegations += 1
                arguments = json.loads(tool_call.function.arguments)
                task = arguments["task"]

                # Route to the correct helper based on the function name
                # the manager requested.
                if tool_call.function.name == "ask_search_llm":
                    result = ask_search_llm(client, tavily, model, task, usage)
                    helper_name = "Search helper"
                else:
                    result = ask_maths_llm(client, model, task, usage)
                    helper_name = "Maths helper"
                helper_notes.append(f"{helper_name} task: {task}\nResult: {result}")

            # CRITICAL: The Groq/OpenAI API requires that every tool_call
            # gets a matching tool result with the same tool_call_id.
            # Without this, the next API call will fail. The LLM reads
            # these results to decide what to do next.
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )

    # --- FALLBACK: only reached if all 4 delegations were used up ---
    # The manager never voluntarily finished, so we force a final answer.
    # We create a brand-new conversation with NO tools available, so the
    # LLM literally cannot delegate anymore. We dump all helper notes in
    # as context and tell it to write the answer now.
    final_messages = [
        {
            "role": "system",
            "content": (
                "Write the final answer using the helper notes. Do not request more "
                "helpers or tools. If a fact is missing, say so clearly."
            ),
        },
        {"role": "user", "content": query},
        # Two user messages in a row is valid in the API. This one carries
        # all the research/math results the helpers gathered.
        {"role": "user", "content": "Helper notes:\n" + "\n\n".join(helper_notes)},
    ]
    response = client.chat.completions.create(
        model=model, messages=final_messages, temperature=0.1
    )
    add_tokens(usage, response)
    return response.choices[0].message.content or "The manager could not finish."


if __name__ == "__main__":
    # Load API keys from a .env file (GROQ_API_KEY, TAVILY_API_KEY, etc.)
    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    # Fall back to a default model if GROQ_MODEL isn't set in .env
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    question = input("Ask a question (Enter uses the test question): ").strip()
    if not question:
        question = TEST_QUERY

    # Shared dict that accumulates token usage across ALL LLM calls
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    answer = ask_manager_llm(client, tavily, model, question, tokens)
    print("\nAnswer:\n", answer)
    print("\nToken usage:", tokens)
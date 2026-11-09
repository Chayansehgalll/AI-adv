
"""Simple Java coding agent powered by Groq."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from tools import make_tools

load_dotenv(Path(__file__).resolve().with_name(".env"))

MAX_RETRIES = 3
MAX_TOOL_CALLS = 12


def print_step(message):
    print(f"\n=== {message} ===")


def ask_llm_to_fix(workspace, bug_description):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("GROQ_API_KEY is missing in .env")
        return False, True

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    client = Groq(
        api_key=api_key,
        timeout=60,
        max_retries=0
    )

    tool_functions, tool_definitions = make_tools(workspace)

    messages = [
        {
            "role": "system",
            "content": """
You are a senior Java debugging agent.

Your job is to identify and fix simple Java bugs.

Follow these steps:

1. Use search_files to identify relevant Java files.
2. Use search_in_file to inspect relevant code.
3. If necessary, use an empty keyword to inspect the entire file.
4. Identify the root cause of the bug.
5. Use replace_line to fix the bug.
6. Use search_in_file again to verify the changes.

Rules:
- Make minimal changes.
- Do not modify unrelated code.
- Do not invent file paths or line numbers.
- Use only the provided tools to modify files.
- Never claim a fix unless the replacement succeeded.
- Explain what you changed.
- If you cannot identify the bug, explain why.
"""
        },
        {
            "role": "user",
            "content": f"Bug description: {bug_description}"
        }
    ]

    changed = False
    verified = False

    for _ in range(MAX_TOOL_CALLS):

        print_step(f"Tool call: Groq LLM ({model})")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_definitions,
                tool_choice="auto",
                temperature=0
            )

        except Exception as exc:
            status = getattr(exc, "status_code", None)

            print(f"Groq API error: {exc}")

            if status in (401, 403):
                print("Check your API key, model access and account permissions.")
                return False, True

            return False, False

        assistant_message = response.choices[0].message

        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:

            if assistant_message.content:
                print("\nAgent:", assistant_message.content)

            return changed and verified, False

        for tool_call in tool_calls:

            tool_name = tool_call.function.name

            print_step(f"Tool call: {tool_name}")

            if tool_name not in tool_functions:
                result = f"Unknown tool: {tool_name}"

            else:
                try:
                    arguments = json.loads(
                        tool_call.function.arguments or "{}"
                    )

                    result = tool_functions[tool_name](**arguments)

                except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
                    result = f"Tool execution failed: {exc}"

            print(result)

            if (
                tool_name == "replace_line"
                and result.startswith("Line ")
                and " replaced." in result
            ):
                changed = True

            if tool_name == "search_in_file" and changed:
                verified = True

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    print(f"Maximum tool calls ({MAX_TOOL_CALLS}) reached.")
    return False, False


def main():

    workspace = Path(__file__).resolve().parent

    print("Java coding agent ready.")
    print(f"Workspace: {workspace}")

    bug_description = input(
        "Describe the bug in the Java file: "
    ).strip()

    if not bug_description:
        print("No bug description provided.")
        return

    java_files = list(workspace.rglob("*.java"))

    if not java_files:
        print("No Java files found.")
        return

    print(f"Found {len(java_files)} Java file(s).")

    for attempt in range(1, MAX_RETRIES + 1):

        print_step(f"Attempt {attempt}/{MAX_RETRIES}")

        success, fatal = ask_llm_to_fix(
            workspace,
            bug_description
        )

        if success:
            print("\nBug fixed and changes verified successfully.")
            return

        if fatal:
            print("\nStopping because of an API configuration error.")
            return

        if attempt < MAX_RETRIES:
            print(f"Attempt {attempt} unsuccessful. Retrying...")

    print("\nI cannot solve the bug after 3 attempts.")


if __name__ == "__main__":
    main()

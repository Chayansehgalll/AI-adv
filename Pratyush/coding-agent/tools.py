
"""Workspace-scoped tools for the Java coding agent."""

from pathlib import Path
from typing import Callable


CODE_EXTENSIONS = {".java"}


def make_tools(workspace: Path):

    workspace = workspace.resolve()

    def relative_path(path):
        return path.relative_to(workspace).as_posix()

    def resolve(path):

        full_path = (workspace / path).resolve()

        if not full_path.is_relative_to(workspace):
            return None

        if not full_path.is_file():
            return None

        if full_path.suffix.lower() not in CODE_EXTENSIONS:
            return None

        return full_path

    def code_files():

        return sorted(
            path for path in workspace.rglob("*")
            if path.is_file()
            and path.suffix.lower() in CODE_EXTENSIONS
            and path.resolve().is_relative_to(workspace)
        )

    # TOOL 1: SEARCH FILES

    def search_files(keyword: str):

        files = code_files()

        if not files:
            return "No Java files found."

        keyword = keyword.lower().strip()

        found = []

        for path in files:

            content = path.read_text(encoding="utf-8")

            matches = content.lower().count(keyword) if keyword else 0

            if (
                not keyword
                or matches
                or keyword in path.name.lower()
            ):
                found.append(
                    f"{relative_path(path)} ({matches} matches)"
                )

        if found:
            return "\n".join(found)

        return (
            f"No direct match for '{keyword}'.\n"
            "Available Java files:\n"
            + "\n".join(relative_path(p) for p in files)
        )

    # TOOL 2: SEARCH INSIDE FILE

    def search_in_file(path: str, keyword: str):

        full_path = resolve(path)

        if full_path is None:
            return f"Java file '{path}' not found."

        lines = full_path.read_text(
            encoding="utf-8"
        ).splitlines()

        keyword = keyword.lower().strip()

        matches = [
            f"{number:>4}: {line}"
            for number, line in enumerate(lines, start=1)
            if not keyword or keyword in line.lower()
        ]

        if not matches:
            return (
                f"'{keyword}' not found in {path}. "
                "Use an empty keyword to display the entire file."
            )

        return "\n".join(matches)

    # TOOL 3: REPLACE LINE

    def replace_line(
        path: str,
        line_number: int,
        new_line: str
    ):

        full_path = resolve(path)

        if full_path is None:
            return f"Java file '{path}' not found."

        if isinstance(line_number, bool) or not isinstance(line_number, int):
            return "line_number must be an integer."

        if "\n" in new_line or "\r" in new_line:
            return "new_line must contain exactly one line."

        with full_path.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:
            lines = file.readlines()

        if not 1 <= line_number <= len(lines):
            return f"Invalid line number: {line_number}"

        old_line = lines[line_number - 1]

        old_text = old_line.rstrip("\r\n")
        line_ending = old_line[len(old_text):]

        indentation = old_text[
            :len(old_text) - len(old_text.lstrip())
        ]

        if new_line.strip() and not new_line[0].isspace():
            new_line = indentation + new_line

        if new_line == old_text:
            return "The replacement is identical to the existing line."

        lines[line_number - 1] = new_line + line_ending

        with full_path.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as file:
            file.writelines(lines)

        return (
            f"Line {line_number} of {path} replaced.\n"
            f"Before: {old_text.strip()}\n"
            f"After:  {new_line.strip()}"
        )

    tools = {
        "search_files": search_files,
        "search_in_file": search_in_file,
        "replace_line": replace_line
    }

    # GROQ / OPENAI-COMPATIBLE TOOL DEFINITIONS

    definitions = [

        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": (
                    "Find Java files related to a bug using a keyword. "
                    "If no match is found, list all available Java files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string"
                        }
                    },
                    "required": ["keyword"]
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "search_in_file",
                "description": (
                    "Search inside a Java file and return matching "
                    "lines with their line numbers. Use an empty "
                    "keyword to display the entire file."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string"
                        },
                        "keyword": {
                            "type": "string"
                        }
                    },
                    "required": ["path", "keyword"]
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "replace_line",
                "description": (
                    "Replace a complete Java source line using "
                    "its 1-based line number."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string"
                        },
                        "line_number": {
                            "type": "integer"
                        },
                        "new_line": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "path",
                        "line_number",
                        "new_line"
                    ]
                }
            }
        }

    ]

    return tools, definitions

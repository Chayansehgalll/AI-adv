We want to build a coding agent that can solve simple bugs.

We will use the Grok LLM for this. Check our previous files to see which LLM setup we will use.

The LLM will have access to a few tools that will help it solve bugs:

1. **Search Files:** Lists the paths of files that are relevant to the bug, based on the LLM's understanding.
2. **Search in File:** Searches within a file and returns the relevant lines along with their line numbers.
3. **Replace:** Replaces an existing line with a new line.

To test this, I want to try it myself. Prepare a simple `java` file with boilerplate code. Do not introduce any bugs for now. I will introduce the bugs myself and test the agent.

I want to describe the bug in simple English through the CLI, and the LLM, along with its tools, should be able to identify and fix it.

Additional requirements:

- Do not use LangGraph. Keep the implementation simple.
- Set a maximum retry limit of 3 attempts. If the LLM cannot solve the bug within these attempts, it should print a message saying that it cannot solve the bug.
- Include clear print statements indicating which tool is being called at each step. We want to understand the entire execution flow.
- Whenever I modify the java file and run the agent, it should ask me to describe the bug.
- Once I provide the bug description, the agent should start analysing and solving it.
- Do not write any test cases for now.

If anything is unclear, ask me before proceeding.
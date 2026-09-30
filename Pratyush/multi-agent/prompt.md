I want to make a multi model agent system but first i want to make single model 

It will have one LLM (groq) which will do all the work 
it will have 2 tools
1 tavily for web search and 2nd python calculator 

there will be 1 ask_llm function which will decide which tool to use and how many times, max attempts will be 4

then we'll convert it to multi agent system without langgraph

there will be 3 ask_llm functions
1st ask manager which will decide which helpeer llm to call
2nd will be ask_search_llm which will do the web searches for us
3rd will be ask_maths_llm which will have access to calculator

role of each llm can be given by ystem prompt
manager llm will also have a notes file which will keep track of all responses by helper llm
finally we will have token usage also and we'll compare both the systems to see which uses more tokens

for test prepare a complex query which involves search and calculator maybe more than once, ask if you have doubts
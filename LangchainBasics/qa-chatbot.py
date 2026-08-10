"""
Simple LangChain Streamlit App with Groq
A beginner-friendly version focusing on core concepts
"""
# What is langchain ? It is a framework for developing applications powered by language models.
# It provides a standard interface for all LLMs, as well as tools for prompt management, memory, and more.
# It allows developers to build applications that can interact with users in natural language, 
# making it easier to create chatbots, question-answering systems, and other AI-powered applications.

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
import os

## Page config
st.set_page_config(page_title="Simple LangChain Chatbot with Groq", page_icon="🚀")

# Title
st.title("🚀 Simple LangChain Chat with Groq")
st.markdown("Learn LangChain basics with Groq's ultra-fast inference!")

with st.sidebar:
    st.header("Settings")

    ## APi Key
    api_key=st.text_input("GROQ API Key", type="password",help="GET Free API Key at console.groq.com")

    ## Model Selection
    model_name=st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "openai/gpt-oss-20b"],
         index=0
    )

    # Clear button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

## Initialize LLM
@st.cache_resource  #@st.cache_resource tells Streamlit:
                    # "Run this function ONCE, save the result.
                    #  Next time it's called with same inputs, 
                    #  DON'T run again, just give saved result"
def get_chain(api_key,model_name):
    if not api_key:
        return None
    
    ## Initialize the GROQ Model
    llm=ChatGroq(groq_api_key=api_key,     
             model_name=model_name,
             temperature=0.7,
             streaming=True)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant powered by Groq. Answer questions clearly and concisely."),
        ("user", "{question}")
    ])

    ## create chain
    chain=prompt| llm | StrOutputParser()

    return chain

## get chain
my_bot_engine = get_chain(api_key,model_name)

if not my_bot_engine :
    st.warning("👆 Please enter your Groq API key in the sidebar to start chatting!")
    st.markdown("[Get your free API key here](https://console.groq.com)")

else:
##   Display the chat messages
#    st.session_state.messages  →  Our list of all messages
#                                Looks like this:
# [
#   {"role": "user", "content": "Hello!"},
#   {"role": "assistant", "content": "Hi there!"},
#   {"role": "user", "content": "How are you?"}
# ]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  #role = "user"      → show user style bubble
                                                #role = "assistant" → show assistant style bubble
            st.write(message["content"])        # Display the message content

    
    ## chat input
    if question:= st.chat_input("Ask me anything"):
        ## Add user message to session state
        st.session_state.messages.append({"role":"user","content":question}) # add user message in notebook / add to session state
        with st.chat_message("user"): # print user message in chat bubble
            st.write(question)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream response from Groq
                for chunk in my_bot_engine.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

## Examples

st.markdown("---")
st.markdown("### 💡 Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq's LPU technology")
with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a haiku about AI")

# Footer
st.markdown("---")
st.markdown("Built with LangChain & Groq | Experience the speed! ⚡")
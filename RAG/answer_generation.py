from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# 1. Load Embedding Model
# -----------------------------

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# -----------------------------
# 2. Load Chroma Vector Store
# -----------------------------

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# -----------------------------
# 3. User Query
# -----------------------------

query = "How much did Microsoft pay to acquire GitHub?"

# -----------------------------
# 4. Retrieve Relevant Documents
# -----------------------------

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)

# -----------------------------
# 5. Display Retrieved Context
# -----------------------------

print(f"User Query: {query}")

print("\n--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:")
    print(doc.page_content)
    print()


# -----------------------------
# 6. Combine Query + Documents
# -----------------------------

combined_input = f"""
Based on the following documents, please answer this question:

Question:
{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear and helpful answer using only the information
from these documents.

If you can't find the answer in the documents, say:
"I don't have enough information to answer that question based on the provided documents."
"""


# -----------------------------
# 7. Create LLM
# -----------------------------

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# -----------------------------
# 8. Create Messages
# -----------------------------

messages = [
    SystemMessage(
        content="You are a helpful assistant that answers questions using the provided documents."
    ),
    HumanMessage(
        content=combined_input
    ),
]


# -----------------------------
# 9. Generate Answer
# -----------------------------

result = model.invoke(messages)


# -----------------------------
# 10. Display Final Answer
# -----------------------------

print("\n--- Generated Response ---")

print(result.content)
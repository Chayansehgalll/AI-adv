from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from collections import defaultdict

load_dotenv()

# Setup
persistent_directory = "db/chroma_db"
llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0)
embedding_model=HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]
# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# ──────────────────────────────────────────────────────────────────
# Step 1: Generate Multiple Query Variations
# ──────────────────────────────────────────────────────────────────

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "="*60)

# ──────────────────────────────────────────────────────────────────
# Step 2: Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────

retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more docs for better RRF
all_retrieval_results = []  # Store all results for RRF

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)  # Store for RRF calculation
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")
print("Notice how different query variations retrieved different documents.")

# ──────────────────────────────────────────────────────────────────
# Step 3: Apply Reciprocal Rank Fusion (RRF)
# ──────────────────────────────────────────────────────────────────

# chunk_lists = [
#     [doc1, doc2, doc3, doc4, doc5],  # Results from Query 1
#     [doc3, doc1, doc6, doc7, doc2],  # Results from Query 2
#     [doc2, doc5, doc1, doc8, doc4],  # Results from Query 3
# ]

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Stores {chunk_content: total_rrf_score}
    all_unique_chunks = {}           # Stores: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}                 # Stores {chunk_content: "Chunk_1", "Chunk_2"...}
    chunk_counter = 1                 # Just a counter for naming chunks
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier coz we want to avoid duplicates and documents don't have unique IDs by default
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks

# Apply RRF to our retrieval results
fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=True)

# ──────────────────────────────────────────────────────────────────
# Step 4: Display Final Fused Results
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FINAL RRF RANKING")
print("="*60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(f"\n✅ RRF Complete! Fused {len(fused_results)} unique documents from {len(query_variations)} query variations.")
print("\n💡 Key benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score") 
print("   • Balanced fusion using k=60 for gentle position penalties")

# ──────────────────────────────────────────────────────────────────
# Optional: Quick Usage Examples
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("USAGE EXAMPLES")
print("="*60)

#########################################################
# VISUAL REPRESENTATION OF EVERYTHING
# #######################################################


# ORIGINAL QUESTION: "How does Tesla make money?"
#                           │
#                           ▼
# ┌─────────────────────────────────────────────────────┐
# │                    LLM (Groq)                       │
# │         Generates 3 Query Variations                │
# └─────────────────────────────────────────────────────┘
#                           │
#           ┌───────────────┼───────────────┐
#           ▼               ▼               ▼
#     "Tesla revenue   "Tesla business  "Tesla income
#       streams"         model"           sources"
#           │               │               │
#           ▼               ▼               ▼
#     ┌──────────┐    ┌──────────┐    ┌──────────┐
#     │ChromaDB  │    │ChromaDB  │    │ChromaDB  │
#     │ Search   │    │ Search   │    │ Search   │
#     └──────────┘    └──────────┘    └──────────┘
#           │               │               │
#           ▼               ▼               ▼
#     Query A Results  Query B Results  Query C Result
#     ┌───────────┐    ┌───────────┐    ┌───────────┐
#     │1. Chunk1  │    │1. Chunk2  │    │1. Chunk3  │
#     │2. Chunk2  │    │2. Chunk3  │    │2. Chunk1  │
#     │3. Chunk4  │    │3. Chunk1  │    │3. Chunk2  │
#     │4. Chunk3  │    │4. Chunk6  │    │4. Chunk7  │
#     │5. Chunk5  │    │5. Chunk8  │    │5. Chunk4  │
#     └───────────┘    └───────────┘    └───────────┘
#           │               │               │
#           └───────────────┼───────────────┘
#                           │
#                           ▼
# ┌─────────────────────────────────────────────────────┐
# │              RECIPROCAL RANK FUSION                 │
# │                                                     │
# │  score = 1 / (60 + position)                        │
# │  Same chunk found again? += add new score on top    │
# │                                                     │
# │  Chunk1: 1/(60+1) + 1/(60+3) + 1/(60+2) = 0.0484  │
# │  Chunk2: 1/(60+2) + 1/(60+1) + 1/(60+3) = 0.0484  │
# │  Chunk3: 1/(60+4) + 1/(60+2) + 1/(60+1) = 0.0481  │
# │  Chunk4: 1/(60+3) + 0        + 1/(60+5) = 0.0312  │
# │  Chunk5: 1/(60+5) + 0        + 0        = 0.0154  │
# │  Chunk6: 0        + 1/(60+4) + 0        = 0.0156  │
# │  Chunk7: 0        + 0        + 1/(60+4) = 0.0156  │
# │  Chunk8: 0        + 1/(60+5) + 0        = 0.0154  │
# └─────────────────────────────────────────────────────┘
#                           │
#                           ▼
# ┌─────────────────────────────────────────────────────┐
# │                 FINAL RANKING                       │
# │                                                     │
# │  🏆 Rank 1 → Chunk1 (0.0484) ◄── in ALL 3 queries  │
# │  🥈 Rank 2 → Chunk2 (0.0484) ◄── in ALL 3 queries  │
# │  🥉 Rank 3 → Chunk3 (0.0481) ◄── in ALL 3 queries  │
# │     Rank 4 → Chunk4 (0.0312) ◄── in 2 queries      │
# │     Rank 5 → Chunk6 (0.0156) ◄── in 1 query only   │
# │     Rank 6 → Chunk7 (0.0156) ◄── in 1 query only   │
# │     Rank 7 → Chunk5 (0.0154) ◄── in 1 query only   │
# │     Rank 8 → Chunk8 (0.0154) ◄── in 1 query only   │
# └─────────────────────────────────────────────────────┘
#                           │
#                           ▼
#               TOP CHUNKS SENT TO LLM
#               TO GENERATE FINAL ANSWER


# KEY TAKEAWAY:
# ─────────────────────────────────────────────────────
#   Appears in 3 queries → 3 scores added → HIGHEST RANK
#   Appears in 2 queries → 2 scores added → MIDDLE RANK
#   Appears in 1 query   → 1 score added  → LOWEST RANK
# ─────────────────────────────────────────────────────
#   Position 1 in ONE query < Appearing in ALL queries
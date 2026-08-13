# MultiModel RAG

A **Multimodal RAG (Retrieval-Augmented Generation)** pipeline that processes PDF documents containing **text, tables, and images**.

The project uses **Unstructured** for PDF parsing, **Gemini** for multimodal understanding, **Hugging Face BGE embeddings** for semantic search, and **ChromaDB** for vector storage.

---

## Features

- Extract text, tables, and images from PDFs
- High-resolution document parsing with Unstructured
- Intelligent title-based chunking
- AI-enhanced summaries for mixed content
- Preserve original text, table HTML, and images
- Semantic search using Hugging Face embeddings
- Persistent vector storage with ChromaDB
- Multimodal answer generation using Gemini
- Export processed chunks and retrieval results to JSON

---

## Architecture

```text
                        PDF Document
                             │
                             ▼
                   Unstructured Parser
                             │
                  ┌──────────┼──────────┐
                  ▼          ▼          ▼
                Text       Tables      Images
                  │          │          │
                  └──────────┼──────────┘
                             ▼
                     Title-Based Chunking
                             │
                             ▼
                   Content Type Detection
                             │
                             ▼
                  Gemini AI Enhancement
                             │
                             ▼
            HuggingFace Embeddings (BGE)
                             │
                             ▼
                          ChromaDB
                             │
                             ▼
                         User Query
                             │
                             ▼
                    Semantic Retrieval
                             │
                             ▼
                   Gemini Final Answer
```

---

## Tech Stack

| Technology                 | Purpose                          |
| -------------------------- | -------------------------------- |
| **Unstructured**           | PDF, table, and image extraction |
| **LangChain**              | RAG pipeline                     |
| **Gemini Flash**           | Multimodal summaries and answers |
| **BAAI/bge-small-en-v1.5** | Document embeddings              |
| **ChromaDB**               | Vector database                  |
| **PyTorch**                | ML backend                       |
| **OpenCV**                 | Image processing                 |

---

## Project Structure

```text
MultiModel-RAG/
│
├── docs/
│   └── attention-is-all-you-need.pdf
│
├── dbv1/
│   └── chroma_db/
│
├── dbv2/
│   └── chroma_db/
│
├── multi_modal_rag.ipynb
├── chunks_export.json
├── rag_results.json
├── requirements.txt
├── test.py
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Chayansehgalll/AI-adv.git
cd AI-adv/MultiModel-RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## System Dependencies

### Linux

```bash
sudo apt-get update
sudo apt-get install poppler-utils tesseract-ocr libmagic-dev
```

### macOS

```bash
brew install poppler tesseract libmagic
```

For Windows, install **Poppler** and **Tesseract OCR** and add them to your system `PATH`.

---

## Environment Setup

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

---

## Embedding Model

The project uses a local Hugging Face embedding model:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

---

## Pipeline

```text
1. Partition PDF
        ↓
2. Extract Text, Tables & Images
        ↓
3. Create Title-Based Chunks
        ↓
4. Detect Content Types
        ↓
5. Enhance Mixed Content with Gemini
        ↓
6. Generate Hugging Face Embeddings
        ↓
7. Store in ChromaDB
        ↓
8. Retrieve Relevant Chunks
        ↓
9. Generate Final Answer with Gemini
```

---

## Main Functions

| Function                            | Description                                                      |
| ----------------------------------- | ---------------------------------------------------------------- |
| `partition_document()`              | Extracts PDF elements including text, tables, and images.        |
| `create_chunks_by_title()`          | Creates intelligent chunks based on document structure.          |
| `separate_content_types()`          | Separates text, tables, and images from chunks.                  |
| `create_ai_enhanced_summary()`      | Uses Gemini to create searchable descriptions for mixed content. |
| `summarise_chunks()`                | Processes chunks and preserves original multimodal metadata.     |
| `create_vector_store()`             | Generates embeddings and stores them in ChromaDB.                |
| `run_complete_ingestion_pipeline()` | Runs the complete ingestion pipeline.                            |
| `generate_final_answer()`           | Generates answers from retrieved multimodal content.             |

---

## Example Query

```python
query = "What are the two main components of the Transformer architecture?"

retriever = db.as_retriever(search_kwargs={"k": 3})
chunks = retriever.invoke(query)

final_answer = generate_final_answer(chunks, query)

print(final_answer)
```

---

## Key Idea

Unlike traditional text-only RAG systems, this project preserves **tables and images** alongside text.

```text
Traditional RAG:
PDF → Text → Chunks → Embeddings → Vector DB → Answer

MultiModel RAG:
PDF → Text + Tables + Images
    → AI Enhancement
    → Embeddings
    → ChromaDB
    → Multimodal Retrieval
    → Gemini Answer
```

---

## Author

**Chayan Sehgal**

Applied AI | Generative AI | RAG | LLM Applications

**Repository:** https://github.com/Chayansehgalll/AI-adv/tree/main/MultiModel-RAG

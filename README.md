# Simple RAG Pipeline using LangChain, ChromaDB & Ollama

A beginner Retrieval-Augmented Generation (RAG) project built to understand how modern RAG systems work.

This project demonstrates the complete RAG workflow:

* Loading text documents
* Splitting documents into chunks
* Creating embeddings using Hugging Face
* Storing embeddings in ChromaDB
* Retrieving relevant documents using semantic search
* Passing the retrieved context to a local LLM (Ollama) for answer generation

---

## Project Structure

```
.
├── docs/                     # Knowledge base (.txt documents)
├── db/
│   └── chroma_db/            # Chroma vector database
├── ingestion_pipline.py      # Document ingestion pipeline
├── reterival_pipeline.py     # Retrieval + LLM pipeline
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Ollama
* CharacterTextSplitter

---

## Pipeline

### 1. Document Loading

Documents are loaded from the `docs/` directory using LangChain's `DirectoryLoader`.

Supported format:

* `.txt`

---

### 2. Text Chunking

Documents are split into chunks using:

* CharacterTextSplitter
* Chunk Size: **1000**
* Chunk Overlap: **0**

---

### 3. Embedding Generation

Embedding model used:

```
sentence-transformers/all-MiniLM-L6-v2
```

---

### 4. Vector Storage

Embeddings are stored in **ChromaDB** using **Cosine Similarity** for retrieval.

---

### 5. Retrieval

Relevant chunks are retrieved using semantic similarity search.

Current configuration:

```
Top K = 3
```

---

### 6. Answer Generation

The retrieved context is passed to a locally running Ollama model.

Current model:

```
qwen2.5-32k
```

The model is instructed to answer **only from the retrieved documents**.

If the answer is not present in the retrieved context, it responds with:

```
IDK
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

Create a virtual environment

```bash
python -m venv env
```

Activate it

Linux / macOS

```bash
source env/bin/activate
```

Windows

```bash
env\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1

Place your text documents inside the `docs/` folder.

### Step 2

Run the ingestion pipeline

```bash
python ingestion_pipline.py
```

This will:

* Load documents
* Split them into chunks
* Generate embeddings
* Store them inside ChromaDB

### Step 3

Run the retrieval pipeline

```bash
python reterival_pipeline.py
```

The program retrieves the most relevant chunks and asks the local LLM to answer the query.

---

## Dataset Notes

This repository includes a small collection of company documents that were created for learning and experimentation.

**Important:**

* The **NVIDIA** document is the most complete and highest-quality document in this dataset.
* The remaining company documents (Amazon, Apple, Microsoft, Alphabet, etc.) were created only for practice and are **not comprehensive or production-quality**.
* If you want to evaluate retrieval quality, it is recommended to use the NVIDIA document or replace the existing documents with higher-quality sources.

---

## Future Improvements

Some ideas for improving this project:

* RecursiveCharacterTextSplitter
* Better prompt engineering
* Metadata filtering
* PDF support
* Hybrid Search (BM25 + Vector Search)
* Reranking
* Conversational memory
* Better chunking strategies
* Source citation in responses
* Streamlit or Gradio interface

---

## Learning Purpose

This project was built for learning Retrieval-Augmented Generation (RAG) and understanding how document ingestion, embeddings, vector databases, retrieval, and local language models work together.

Feedback and suggestions are always welcome.

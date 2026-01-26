# 🤖 Agentic RAG Pipeline: Local Document Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Llama 3.2](https://img.shields.io/badge/LLM-Llama%203.2-orange.svg)](https://ollama.com/library/llama3.2)
[![Framework: LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![Vector DB: Chroma](https://img.shields.io/badge/VectorDB-Chroma-yellow.svg)](https://www.trychroma.com/)

An intelligent, privacy-first Document Assistant that uses **Agentic Routing** to determine whether to retrieve information from local PDFs or answer directly using the LLM's internal knowledge. Built with **Llama 3.2** and **LangChain**.

---

## 🌟 Overview
Most RAG (Retrieval-Augmented Generation) systems perform a vector search for every single query, which is computationally expensive for general questions. This project implements a **Routing Agent** that analyzes user intent to optimize the pipeline.

### Key Features
- **Privacy-First:** Entirely local execution via **Ollama**; no data leaves your machine.
- **Smart Routing:** Classifies user intent to choose between `Vector Search` or `Direct LLM Response`.
- **Semantic Retrieval:** Uses `nomic-embed-text` to find context based on meaning, not just keywords.
- **Persistent Storage:** Uses **ChromaDB** to save document embeddings locally.

---

## 🏗️ Technical Architecture

```mermaid
graph TD
    A[User Query] --> B{Agentic Router}
    B -- "Keywords: PDF, Data, Report" --> C[Vector Store: ChromaDB]
    B -- "General Knowledge" --> D[LLM: Llama 3.2]
    C --> E[Context Retrieval]
    E --> D
    D --> F[Final Response]

---

## 🛠️ Setup Instructions

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.com) installed and the following models downloaded:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text

###2. Clone the repository:
```bash
git clone [https://github.com/Mohammadyakub221/agentic-rag-pipeline-with-llama3.2.git](https://github.com/Mohammadyakub221/agentic-rag-pipeline-with-llama3.2.git)
cd agentic-rag-pipeline-with-llama3.2

###3.Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

###4.Install dependencies:
```bash
pip install -r requirements.txt

##🚀 Usage:

1.Place your PDF documents in the /data directory.
2.Run the pipeline:
```bash
python src/main.py



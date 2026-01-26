# 🤖 Agentic RAG Pipeline: Local Document Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Llama 3.2](https://img.shields.io/badge/LLM-Llama%203.2-orange.svg)](https://ollama.com/library/llama3.2)
[![Framework: LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![Vector DB: Chroma](https://img.shields.io/badge/VectorDB-Chroma-yellow.svg)](https://www.trychroma.com/)

An intelligent, privacy-first Document Assistant that uses **Agentic Routing** to determine whether to retrieve information from local PDFs or answer directly using the LLM's internal knowledge.

---

## 🌟 Overview
Standard RAG systems often perform unnecessary searches for general queries. This project implements a **Routing Agent** that acts as a traffic controller, optimizing compute resources and response accuracy.

### Key Features
- **Privacy-First:** Entirely local execution via Ollama; no data leaves your system.
- **Smart Routing:** Classifies user intent to choose between `Vector Search` or `Direct LLM Response`.
- **High-Performance Embeddings:** Utilizes `nomic-embed-text` for superior semantic retrieval.
- **Stateful Retrieval:** Persistent vector storage using ChromaDB.

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

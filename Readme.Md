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



1. **User Query**: Input from the terminal.
2. **Agent Router**: A logic layer that checks for keywords related to documents/data.
3. **Route A (RAG)**: Retrieves relevant snippets from the local PDF store using semantic search.
4. **Route B (Direct)**: Bypasses the database for general knowledge questions (e.g., "How do I bake a cake?").
5. **Response**: Llama 3.2 synthesizes the final answer.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.com) installed and the following models downloaded:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text


Agentic RAG Pipeline with Llama 3.2
An intelligent Document Assistant that uses Agentic Routing to decide whether to search through local documents or answer directly using the LLM's internal knowledge.

🚀 Key Features
Local-First & Private: Uses Ollama to run Llama 3.2 and Nomic-Embeddings locally—no data leaves your machine.

Agentic Routing: Implements a controller that analyzes user intent to choose between RAG retrieval or direct response, saving compute resources.

Vector Storage: Uses ChromaDB for persistent semantic search.

🛠️ Setup
Install Ollama.

ollama pull llama3.2 and ollama pull nomic-embed-text.

pip install -r requirements.txt.

Place your PDFs in the /data folder.

Run python main.py.
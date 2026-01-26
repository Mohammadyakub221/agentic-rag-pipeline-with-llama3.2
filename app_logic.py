from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# 1. Initialize Llama 3.2
llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Loading and Chunking


def process_pdfs(folder_path):
    loader_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            loader_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(loader_docs)

# 3. Vector Store Setup


def get_retriever(chunks):
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Agent Router


def agent_router(query):
    query_lower = query.lower()
    search_keywords = ["pdf", "document", "data", "report", "context", "find"]
    if any(word in query_lower for word in search_keywords):
        return "search"
    return "direct"

# 5. Execution Loop


def run_agentic_rag(query, retriever):
    route = agent_router(query)

    if route == "search":
        print(f"🔍 Agent Routing: Document Search")
        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])
        prompt = f"Using ONLY the following context:\n{context}\n\nQuestion: {query}"
    else:
        print(f"🤖 Agent Routing: Direct LLM Knowledge")
        prompt = query

    response = llm.invoke(prompt)
    return response.content

# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain_community.vectorstores import Chroma
# from flask import Flask, request, jsonify
# import os

# app = Flask(__name__)
# from flask_cors import CORS  # Import CORS
# CORS(app, resources={r"/chat": {"origins": "*"}})
# # Load and process documents
# def load_documents():
#     docs = []
#     for filename in os.listdir("docs"):
#         filepath = os.path.join("docs", filename)
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(filepath)
#         elif filename.endswith(".txt"):
#             loader = TextLoader(filepath)
#         else:
#             continue
#         docs.extend(loader.load())
#     return docs

# # Split documents into chunks
# documents = load_documents()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = text_splitter.split_documents(documents)

# # Create embeddings and vector store
# embeddings = OllamaEmbeddings(model="llama3.1")
# vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# # Initialize the chat model
# llm = ChatOllama(model="llama3.1")

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     query = data.get("message", "")
    
#     # Search for relevant document chunks
#     relevant_docs = vector_store.similarity_search(query, k=3)
#     context = "\n".join([doc.page_content for doc in relevant_docs])
    
#     # Build the prompt
#     prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"
    
#     # Get response from Ollama
#     response = llm.invoke(prompt)
    
#     return jsonify({"reply": response.content})

# if __name__ == "__main__":
#     # Ensure Ollama is running in another terminal: `ollama serve`
#     # app.run(port=5000, debug=True)
#     app.run(host="0.0.0.0", port=5000, debug=False)

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
import os
import shutil

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})

# Load and process documents
def load_documents():
    docs = []
    for filename in os.listdir("docs"):
        filepath = os.path.join("docs", filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue
        docs.extend(loader.load())
    return docs

# Configurable chunk settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIR = "./chroma_db"

# Initialize document store with check and recreation
def initialize_vector_store():
    # Check if ChromaDB directory exists and remove it
    if os.path.exists(PERSIST_DIR):
        print(f"Removing existing ChromaDB at {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)

    # Load documents
    documents = load_documents()
    if not documents:
        print("No documents found in 'docs/' folder.")
        return None

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks")

    # Create new embeddings and vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    print(f"Created new ChromaDB at {PERSIST_DIR}")
    return vector_store

# Initialize vector store and LLM
vector_store = initialize_vector_store()
llm = ChatOllama(model="llama3.1")

@app.route('/chat', methods=['POST'])
def chat():
    if vector_store is None:
        return jsonify({"reply": "No documents available to chat with."}), 503

    data = request.get_json()
    query = data.get("message", "")
    
    # Search for relevant document chunks
    relevant_docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Build prompt with context
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"
    
    # Get response from llama3.1
    response = llm.invoke(prompt)
    
    return jsonify({"reply": response.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
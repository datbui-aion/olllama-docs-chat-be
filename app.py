from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
from flask_cors import CORS  # Import CORS
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

# Split documents into chunks
documents = load_documents()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3.1")
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# Initialize the chat model
llm = ChatOllama(model="llama3.1")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get("message", "")
    
    # Search for relevant document chunks
    relevant_docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Build the prompt
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"
    
    # Get response from Ollama
    response = llm.invoke(prompt)
    
    return jsonify({"reply": response.content})

if __name__ == "__main__":
    # Ensure Ollama is running in another terminal: `ollama serve`
    # app.run(port=5000, debug=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
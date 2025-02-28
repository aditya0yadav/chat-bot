from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os

app = Flask(__name__)

# Configuration
GEMINI_API_KEY = "AIzaSyBlET66vMBvzxA_20YKxfexLDNervnH6hI"  # Replace with your actual API key
INDEX_DIRECTORY = "./cdp_docs_index"

# Initialize the index and query engine
def initialize_index():
    try:
        # Configure Gemini LLM
        llm = Gemini(api_key=GEMINI_API_KEY)
        
        # Configure embedding model
        embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        
        # Set up LlamaIndex settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # Load existing index
        if os.path.exists(INDEX_DIRECTORY):
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIRECTORY)
            index = load_index_from_storage(storage_context)
            return index
        else:
            return None
    except Exception as e:
        print(f"Error initializing index: {str(e)}")
        return None

# Global variables
index = initialize_index()
query_engine = index.as_query_engine() if index else None

@app.route('/api/chat', methods=['POST'])
def chat():
    if not query_engine:
        return jsonify({"error": "Index not initialized"}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        # Get response from the query engine
        response = query_engine.query(query)
        
        return jsonify({
            "query": query,
            "response": str(response),
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": f"Error processing query: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "index_available": index is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

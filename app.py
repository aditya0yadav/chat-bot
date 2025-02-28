from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import os
import json
import pandas as pd
from typing import List, Optional
import shutil
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import Document, Settings, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configuration
INDEX_DIRECTORY = "./cdp_docs_index"
UPLOAD_DIRECTORY = "./uploads"

# Ensure directories exist
os.makedirs(INDEX_DIRECTORY, exist_ok=True)
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="CDP Documentation Retrieval API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for API requests and responses
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    sources: List[dict] = []

class WebsiteSource(BaseModel):
    name: str
    url: str

class IndexInfo(BaseModel):
    doc_count: int
    sources: List[str]

# Configure LLM and embeddings
@app.on_event("startup")
async def startup_event():
    # Set your API key (consider using environment variables)
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    
    # Configure Gemini LLM
    llm = Gemini(api_key=os.environ["GEMINI_API_KEY"])
    
    # Configure embedding model
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Configure LlamaIndex settings
    Settings.llm = llm
    Settings.embed_model = embed_model

# Helper functions
def fetch_text(url):
    """Fetch and clean text from a given URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            return soup.get_text(separator=' ')  # Extract and normalize text
        print(f"Failed to fetch {url}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
    return ""

def clean_text(text):
    """Tokenize text into sentences and clean them."""
    sentences = sent_tokenize(text)
    # Filter out very short sentences and those with mostly non-text characters
    cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return cleaned_sentences

def get_index():
    """Load the vector index if it exists, otherwise return None."""
    if os.path.exists(INDEX_DIRECTORY):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIRECTORY)
            index = load_index_from_storage(storage_context)
            return index
        except Exception as e:
            print(f"Error loading index: {str(e)}")
    return None

def process_file(file_path, source_name):
    """Process a file and return documents."""
    documents = []
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        for i, row in df.iterrows():
            if 'text' in df.columns:
                text = str(row['text'])
            else:
                text = ' '.join([f"{col}: {val}" for col, val in row.items()])
                
            doc = Document(
                text=text,
                metadata={"source": source_name, "type": "csv", "row_id": i}
            )
            documents.append(doc)
            
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    text = json.dumps(item)
                    doc = Document(
                        text=text,
                        metadata={"source": source_name, "type": "json", "item_id": i}
                    )
                    documents.append(doc)
        elif isinstance(data, dict):
            for key, value in data.items():
                text = f"{key}: {json.dumps(value)}"
                doc = Document(
                    text=text,
                    metadata={"source": source_name, "type": "json", "key": key}
                )
                documents.append(doc)
                
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        sentences = clean_text(text)
        for i, sentence in enumerate(sentences):
            doc = Document(
                text=sentence,
                metadata={"source": source_name, "type": "txt", "sentence_id": i}
            )
            documents.append(doc)
    
    return documents

def process_website(name, url):
    """Process a website and return documents."""
    documents = []
    raw_text = fetch_text(url)
    if raw_text:
        cleaned_sentences = clean_text(raw_text)
        for i, sentence in enumerate(cleaned_sentences):
            doc = Document(
                text=sentence,
                metadata={"source": name, "url": url, "type": "web", "sentence_id": i}
            )
            documents.append(doc)
    return documents

def add_to_index(documents):
    """Add documents to the index, creating a new one if needed."""
    index = get_index()
    
    if index:
        # Add to existing index
        for doc in documents:
            index.insert(doc)
    else:
        # Create new index
        index = VectorStoreIndex.from_documents(documents)
    
    # Save the updated index
    index.storage_context.persist(INDEX_DIRECTORY)
    return index

# API endpoints
@app.get("/")
async def root():
    return {"message": "CDP Documentation Retrieval API is running"}

@app.post("/query", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    index = get_index()
    if not index:
        raise HTTPException(status_code=404, detail="No index found. Please upload documents first.")
    
    query_engine = index.as_query_engine(response_mode="tree_summarize")
    response = query_engine.query(request.query)
    
    # Extract source nodes for citation
    sources = []
    if hasattr(response, 'source_nodes'):
        for node in response.source_nodes:
            sources.append({
                "text": node.node.text[:100] + "..." if len(node.node.text) > 100 else node.node.text,
                "source": node.node.metadata.get("source", "Unknown"),
                "score": node.score
            })
    
    return QueryResponse(response=str(response), sources=sources)

@app.get("/index-info", response_model=IndexInfo)
async def get_index_info():
    index = get_index()
    if not index:
        return IndexInfo(doc_count=0, sources=[])
    
    # Get document count and unique sources
    doc_count = len(index.docstore.docs)
    
    # Extract unique sources
    sources = set()
    for doc_id in index.docstore.docs:
        doc = index.docstore.docs[doc_id]
        source = doc.metadata.get("source", "Unknown")
        sources.add(source)
    
    return IndexInfo(doc_count=doc_count, sources=list(sources))

@app.post("/upload-file")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_name: str = Form(...)
):
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the file in the background
    background_tasks.add_task(process_and_index_file, file_path, source_name)
    
    return JSONResponse(content={
        "message": f"File uploaded successfully. Processing as '{source_name}'.",
        "filename": file.filename
    })

@app.post("/add-website")
async def add_website(background_tasks: BackgroundTasks, source: WebsiteSource):
    # Process the website in the background
    background_tasks.add_task(process_and_index_website, source.name, source.url)
    
    return JSONResponse(content={
        "message": f"Website '{source.name}' added to processing queue.",
        "url": source.url
    })

@app.post("/reset-index")
async def reset_index():
    if os.path.exists(INDEX_DIRECTORY):
        shutil.rmtree(INDEX_DIRECTORY)
        os.makedirs(INDEX_DIRECTORY)
    
    return JSONResponse(content={"message": "Index reset successfully"})

# Background processing functions
def process_and_index_file(file_path, source_name):
    """Process a file and add it to the index as a background task."""
    try:
        documents = process_file(file_path, source_name)
        if documents:
            add_to_index(documents)
            print(f"Added {len(documents)} documents from {source_name} to the index")
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")

def process_and_index_website(name, url):
    """Process a website and add it to the index as a background task."""
    try:
        documents = process_website(name, url)
        if documents:
            add_to_index(documents)
            print(f"Added {len(documents)} documents from website {name} to the index")
    except Exception as e:
        print(f"Error processing website {url}: {str(e)}")

# Run the server
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

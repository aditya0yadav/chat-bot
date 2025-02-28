import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import os
import json
import pandas as pd
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import Document, Settings, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# CDP Documentation URLs - expanded with more sources
doc_urls = {
    "Segment": "https://segment.com/docs/?ref=nav",
    "mParticle": "https://docs.mparticle.com/",
    "Lytics": "https://docs.lytics.com/",
    "Zeotap": "https://docs.zeotap.com/home/en-us/",
    "Tealium": "https://docs.tealium.com/",
    "RudderStack": "https://www.rudderstack.com/docs/",
    "Treasure Data": "https://docs.treasuredata.com/",
    "Bloomreach": "https://documentation.bloomreach.com/"
}

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

def load_additional_data(file_path, source_name):
    """Load data from various file formats."""
    documents = []
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return documents
        
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        for i, row in df.iterrows():
            # Convert row to text - adjust the columns as needed
            if 'text' in df.columns:
                text = str(row['text'])
            else:
                # Use all columns if no specific text column
                text = ' '.join([f"{col}: {val}" for col, val in row.items()])
                
            doc = Document(
                text=text,
                metadata={"source": source_name, "type": "csv", "row_id": i}
            )
            documents.append(doc)
            
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
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
        with open(file_path, 'r') as f:
            text = f.read()
        sentences = clean_text(text)
        for i, sentence in enumerate(sentences):
            doc = Document(
                text=sentence,
                metadata={"source": source_name, "type": "txt", "sentence_id": i}
            )
            documents.append(doc)
            
    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents

# Configure Gemini LLM
llm = Gemini(api_key=gemini_api_key)

# Configure the embedding model using HuggingFace/SentenceTransformers
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# Configure LlamaIndex settings
Settings.llm = llm
Settings.embed_model = embed_model

# Check if an existing index is available
index_directory = "./cdp_docs_index"
all_documents = []

# Function to add documents to an existing index
def add_documents_to_index(index, documents):
    for doc in documents:
        index.insert(doc)
    return index

# Try to load existing index
if os.path.exists(index_directory):
    try:
        print("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=index_directory)
        index = load_index_from_storage(storage_context)
        print("Successfully loaded existing index")
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        print("Creating new index instead...")
        index = None
else:
    print("No existing index found. Creating new index...")
    index = None

# If no existing index was loaded, process the web docs
if index is None:
    # Extract and clean documentation texts from websites
    doc_texts = {}
    
    for name, url in doc_urls.items():
        print(f"Fetching {name} documentation...")
        raw_text = fetch_text(url)
        if raw_text:
            cleaned_sentences = clean_text(raw_text)
            doc_texts[name] = cleaned_sentences
            
            # Create documents with metadata to track source
            for i, sentence in enumerate(cleaned_sentences):
                doc = Document(
                    text=sentence,
                    metadata={"source": name, "url": url, "type": "web", "sentence_id": i}
                )
                all_documents.append(doc)
            
            print(f"  Processed {len(cleaned_sentences)} sentences from {name}")
        else:
            print(f"  No content retrieved from {name}")
    
    # Create new index with the documents
    if all_documents:
        print(f"Creating new vector index from {len(all_documents)} documents...")
        index = VectorStoreIndex.from_documents(all_documents)
    else:
        print("No documents to index!")
        index = VectorStoreIndex([])  # Empty index

# Add additional data from files
additional_data_sources = [
    {"path": "combined_content.txt", "name": "Best Practices"}
]

for data_source in additional_data_sources:
    print(f"Processing additional data source: {data_source['name']}...")
    docs = load_additional_data(data_source['path'], data_source['name'])
    if docs:
        if index:
            print(f"Adding {len(docs)} documents to existing index")
            index = add_documents_to_index(index, docs)
        else:
            all_documents.extend(docs)

# If we only have additional documents and no index yet, create one
if index is None and all_documents:
    print(f"Creating new vector index from {len(all_documents)} documents...")
    index = VectorStoreIndex.from_documents(all_documents)

if index:
    print("Saving index...")
    index.storage_context.persist(index_directory)
    print("Index saved successfully.")

    query_engine = index.as_query_engine()
    print("\nExample query:")
    example_query = "How do I track user events across different CDPs?"
    response = query_engine.query(example_query)
    print(f"Query: {example_query}")
    print(f"Response: {response}")
else:
    print("No index was created. Please check your data sources.")
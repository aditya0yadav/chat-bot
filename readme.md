# CDP Documentation Chatbot (Assignment)

![CDP Documentation Assistant](https://img.shields.io/badge/CDP-Documentation%20Assistant-4361ee)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

This is an **assignment**, not a full project, focused on developing a chatbot that answers questions about CDP (Customer Data Platform) documentation using AI-powered semantic search and natural language processing.

## 📌 Assignment Details
- **Task**: Build a chatbot interface that can understand and retrieve relevant documentation
- **Scope**: Backend implementation using Python and Flask
- **Deployment**: The `app.py` file serves as the backend and can be deployed separately
- **Key Technologies**: Flask, LlamaIndex, Gemini API, HuggingFace Embeddings

## 🚀 Key Features
- **Semantic Search Engine**: Retrieves relevant results based on meaning, not just keywords
- **AI-Powered Responses**: Uses Google's Gemini API for language understanding
- **Efficient Document Retrieval**: Pre-indexed vectorized documentation for fast responses
- **Backend-Only Deployment**: The `app.py` file can be deployed independently for API-based chatbot interactions

## ⚡ Limitations & Justifications
Due to the limited time available (2 days for the assignment), I opted for a lesser **Retrieval-Augmented Generation (RAG) approach** instead of more advanced methods. 

- **Web scraping was not possible** due to sidebar issues in the documentation structure.
- Instead, I used **synthetic data** for testing and evaluation.
- Please **judge the assignment based on these constraints.**

## 🛠️ Technical Overview
### Backend
- **Flask API**: Handles all user queries and responses
- **LlamaIndex**: Manages document indexing and retrieval
- **Gemini API**: Provides AI-powered responses
- **HuggingFace Embeddings**: Uses `all-MiniLM-L6-v2` for text embeddings
- **Vector Store**: Stores pre-computed document embeddings for fast lookups

## 🚀 Deployment Instructions
To deploy the backend (`app.py`):

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up the Gemini API key:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
3. Run the backend server:
   ```bash
   python app.py
   ```

The backend will be running and accessible for frontend or API calls.

## 🔧 Customization
To adapt this chatbot for different documentation:
1. Replace the CDP documentation with your own content.
2. Rebuild the vectorized index.
3. Adjust API responses and configurations as needed.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Google Gemini API for natural language capabilities
- LlamaIndex for the semantic search framework
- HuggingFace for the embedding models



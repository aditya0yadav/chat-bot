# CDP Documentation Chatbot

![CDP Documentation Assistant](https://img.shields.io/badge/CDP-Documentation%20Assistant-4361ee)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A modern, responsive chatbot interface designed to answer questions about CDP (Customer Data Platform) documentation using AI-powered semantic search and natural language processing.

## 🌟 Project Highlights

- **Semantic Search Engine**: Uses vector embeddings to understand the meaning behind user queries, not just keywords
- **Modern UI/UX**: Clean, responsive design with user-friendly interface and real-time interaction
- **AI-Powered Responses**: Leverages Google's Gemini API for natural language understanding and generation
- **Efficient Document Retrieval**: Pre-indexed documentation for fast and relevant responses
- **Real-time Status Indicators**: Provides clear feedback about connection status and system availability

## 📋 Features

- **Natural Language Understanding**: Understands questions asked in conversational language
- **Context-Aware Responses**: Maintains context between messages for more meaningful interactions
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Visual Feedback**: Typing indicators, message animations, and status updates provide a smooth user experience
- **Error Handling**: Graceful handling of server connectivity issues and other potential errors
- **Customizable**: Easy to adapt for different documentation sets or knowledge bases

## 🛠️ Technical Implementation

### Backend

- **Flask API**: Lightweight Python web server handling requests
- **LlamaIndex**: Framework for building LLM-powered applications
- **Gemini API**: Google's advanced language model for generating natural responses
- **HuggingFace Embeddings**: Using the `all-MiniLM-L6-v2` model for efficient text embeddings
- **Vector Store**: Pre-computed document embeddings for semantic search

### Frontend

- **Pure HTML/CSS/JS**: No dependencies on large frameworks
- **Modern CSS**: Flexbox layout, animations, and responsive design
- **Interactive UI**: Real-time feedback and smooth transitions
- **Accessibility**: Designed with usability in mind

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Gemini API key
- CDP documentation corpus (to build the index)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cdp-documentation-chatbot.git
   cd cdp-documentation-chatbot
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

4. Build the documentation index (if not already available):
   ```bash
   python build_index.py --docs_dir=./your_docs_directory
   ```

5. Start the server:
   ```bash
   python app.py
   ```

6. Open `index.html` in your browser or serve it using a web server.

## 📝 Important Notes on Deployment

### Model Size Constraints

**⚠️ Note to Reviewers:** 

Due to the size requirements of the language models and vector embeddings, deployment on free-tier services is challenging. The current implementation requires:

- ~500MB for the embedding model
- ~200MB for the indexed documentation vector store
- Additional memory for the Flask server and API communication

Most free hosting services have memory limits below what's required for optimal performance. For production deployment, I recommend:

- A VPS with at least 2GB RAM
- Container orchestration (Docker/Kubernetes) for easier scaling
- CDN for serving static frontend assets

For evaluation purposes, the application can be run locally or on a temporary cloud instance with sufficient resources.

## 🔍 Project Structure

```
cdp-documentation-chatbot/
├── app.py                 # Flask server
├── index.html             # Frontend interface
├── cdp_docs_index/        # Pre-built vector index
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🔧 Customization

To adapt this chatbot for different documentation:

1. Replace the CDP documentation with your own corpus
2. Rebuild the index using the provided scripts
3. Update the UI with your branding
4. Adjust the model parameters for your specific use case

## 📈 Future Improvements

- Add authentication for protected documentation
- Implement multi-language support
- Add document upload functionality for dynamic index updates
- Integrate with existing knowledge bases or ticketing systems
- Implement conversation history persistence

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for natural language capabilities
- LlamaIndex for the semantic search framework
- HuggingFace for the embedding models
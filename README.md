# DocuMind

A simple Document Question Answering System using Retrieval-Augmented Generation (RAG). Built using Python, Streamlit, LangChain, FAISS, and Hugging Face.

## Features
- Upload PDF documents.
- Chunks texts and creates embeddings using Hugging Face's `all-MiniLM-L6-v2`.
- Stores embeddings in a local, in-memory FAISS vector index.
- Answers queries using context-aware RetrievalQA using the Hugging Face Inference API.

## Setup Instructions

1. **Install Python dependencies:**
   Make sure to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API Key:**
   - Copy `.env.example` to a new file named `.env`.
   - Edit the `.env` file and place your Hugging Face API Token inside.
   *(You can obtain a free API Token by creating an account on Hugging Face and visiting `Settings` > `Access Tokens`)*

3. **Run the Application:**
   Starts the Streamlit web application interface:
   ```bash
   streamlit run app.py
   ```

## Folder Structure
- `app.py`: Streamlit entry point / frontend
- `core/doc_processor.py`: PDF load and chunk split
- `core/vector_store.py`: FAISS embedding initialization
- `core/llm_pipeline.py`: Retrievers and Inference Engine logic
- `.temp/`: A directory temporarily created to store uploaded PDF files.
# RAG

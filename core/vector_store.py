import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()


def _get_hf_token() -> str | None:
    """Return the first supported Hugging Face token from the environment."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

def create_vector_store(chunks):
    """Creates a FAISS vector store from text chunks using HuggingFace embeddings."""
    hf_token = _get_hf_token()

    # Use the lightweight MiniLM model for fast embedding generation
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": hf_token} if hf_token else {},
    )
    
    # Build the FAISS index
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

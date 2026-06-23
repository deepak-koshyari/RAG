import streamlit as st
import os
from dotenv import load_dotenv

# Import our core logic
from core.doc_processor import process_pdf
from core.vector_store import create_vector_store
from core.llm_pipeline import build_rag_pipeline

# Initialize environment variables (specifically HUGGINGFACEHUB_API_TOKEN)
load_dotenv()


def has_hf_token() -> bool:
    """Return True when a non-placeholder Hugging Face token is configured."""
    load_dotenv(override=True)  # Reload environment dynamically
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
    return bool(token and token != "your_huggingface_api_token")

st.set_page_config(page_title="DocuMind - AI PDF Q&A", layout="centered")

st.title("🧠 DocuMind")
st.write("A simple Document Question Answering System using Retrieval-Augmented Generation (RAG).")

# Sidebar for file upload
with st.sidebar:
    st.header("1. Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        # Check if we already processed this file to save time across UI refreshes
        if "processed_doc" not in st.session_state or st.session_state.processed_doc != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                temp_file_path = None
                try:
                    # Save uploaded file to a temporary directory
                    temp_dir = ".temp"
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                        
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the file (chunking)
                    chunks = process_pdf(temp_file_path)
                    
                    # Build Vector Store
                    vector_store = create_vector_store(chunks)
                    
                    # Save to session state
                    st.session_state.vector_store = vector_store
                    st.session_state.processed_doc = uploaded_file.name
                    
                    st.success("File indexed successfully!")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                finally:
                    # Clean up the temp file immediately
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except Exception:
                            pass
    else:
        # Reset session state when file upload is cleared
        if "vector_store" in st.session_state:
            del st.session_state.vector_store
        if "processed_doc" in st.session_state:
            del st.session_state.processed_doc

# Main logic for querying
st.header("2. Ask Questions")
if "vector_store" in st.session_state:
    query = st.text_input("Enter your question about the document:")
    
    if st.button("Get Answer"):
        if query:
            # Check for generic API token to prevent crash
            if not has_hf_token():
                st.error(
                    "Missing a valid Hugging Face API key. Please add "
                    "HUGGINGFACEHUB_API_TOKEN or HF_TOKEN to your .env file."
                )
            else:
                with st.spinner("Thinking..."):
                    try:
                        # Build the locally stored RAG pipeline dynamically
                        qa_chain = build_rag_pipeline(st.session_state.vector_store)
                        
                        # Get answer from LLM (modern chain expects input key)
                        result = qa_chain.invoke({"input": query})
                        
                        # The default output key for modern create_retrieval_chain is 'answer'
                        answer = result.get("answer", result) 
                        
                        # Display output
                        st.subheader("Answer:")
                        st.write(answer)
                        
                        # Show exact text source snippets (The RAG piece)
                        with st.expander("Show source context"):
                            for i, doc in enumerate(result.get("context", [])):
                                st.markdown(f"**Snippet {i+1}:**")
                                st.text(doc.page_content)
                    except Exception as e:
                        st.error(
                            "Error during query execution. Check your Hugging Face "
                            f"token/model configuration. Error Details: {str(e)}"
                        )
        else:
            st.warning("Please enter a question first.")
else:
    st.info("Please upload a PDF document first in the sidebar.")

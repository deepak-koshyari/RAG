from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path: str):
    """Loads a PDF and splits it into chunks."""
    # Load PDF using PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split text recursively
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        raise ValueError("Could not extract any text from the PDF. The file might be scanned or empty.")
        
    return chunks


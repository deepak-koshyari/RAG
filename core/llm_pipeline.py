import os

from dotenv import load_dotenv
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_TASK = "conversational"

def _get_hf_token() -> str | None:
    """Return the first supported Hugging Face token from the environment."""
    return os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")


def build_rag_pipeline(vector_store):
    """Build the retrieval-augmented generation pipeline."""
    hf_token = _get_hf_token()
    if not hf_token:
        raise ValueError(
            "Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN."
        )

    model_id = os.getenv("HUGGINGFACEHUB_MODEL_ID", DEFAULT_MODEL_ID)
    task = os.getenv("HUGGINGFACEHUB_TASK", DEFAULT_TASK)
    provider = os.getenv("HUGGINGFACEHUB_PROVIDER")

    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task=task,
        provider=provider or None,
        temperature=0.1,
        max_new_tokens=512,
        do_sample=False,
        huggingfacehub_api_token=hf_token,
    )

    llm = endpoint
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # PromptTemplate for text-generation models (not chat models)
    prompt_template = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you don't know the answer, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n\n"
        "Answer:"
    )
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=prompt_template,
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


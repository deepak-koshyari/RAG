import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
model_id = os.environ.get("HUGGINGFACEHUB_MODEL_ID", "openai/gpt-oss-20b")

if not hf_token:
    raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN or HF_TOKEN")

endpoint = HuggingFaceEndpoint(
    repo_id=model_id,
    task="conversational",
    max_new_tokens=64,
    do_sample=False,
    huggingfacehub_api_token=hf_token,
)

chat = ChatHuggingFace(llm=endpoint)
print(chat.invoke("What is the capital of France?").content)

import os
import traceback
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
model_id = os.environ.get("HUGGINGFACEHUB_MODEL_ID", "openai/gpt-oss-20b")
task = os.environ.get("HUGGINGFACEHUB_TASK", "conversational")

if not hf_token:
    raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN or HF_TOKEN")

print("Token starts with:", hf_token[:5])
print("Model:", model_id)
print("Task:", task)

try:
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task=task,
        temperature=0.1,
        max_new_tokens=64,
        do_sample=False,
        huggingfacehub_api_token=hf_token,
    )
    llm = ChatHuggingFace(llm=endpoint) if task == "conversational" else endpoint
    result = llm.invoke("What is the capital of France?")
    print("Success:", result)
except Exception as e:
    traceback.print_exc()

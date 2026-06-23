from dotenv import load_dotenv
load_dotenv()
import os
print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
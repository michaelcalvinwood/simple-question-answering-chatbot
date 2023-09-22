import my_huggingface_key

huggingface_key = my_huggingface_key.huggingface_key

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_key

from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id = "google/flan-t5-large")

query = "What is the currency of India?"
response = llm(query)

print(response)
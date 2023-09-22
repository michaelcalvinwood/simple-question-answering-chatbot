import os
import my_openai_key
from langchain.llms import OpenAI # langchain wrapper for openai

os.environ["OPENAI_API_KEY"] = my_openai_key.open_ai_key

llm = OpenAI(model_name="text-davinci-003") # create an instance of an OpenAI model

query = "What is the currency of India?"
response = llm(query)

print(response)

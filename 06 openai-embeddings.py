import os
import my_openai_key
os.environ["OPENAI_API_KEY"] = my_openai_key.open_ai_key

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

our_Text = "Hey buddy"
text_embedding = embeddings.embed_query(our_Text)
print (f"Our embedding is {text_embedding}")
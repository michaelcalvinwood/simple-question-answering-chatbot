import os
import my_openai_key
os.environ["OPENAI_API_KEY"] = my_openai_key.open_ai_key

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
import pandas as pd
df = pd.read_csv('myData.csv')
print(df)

df['embedding'] = df['Words'].apply(lambda x: embeddings.embed_query(x))
df.to_csv('word_embeddings.csv')

new_df = pd.read_csv('word_embeddings.csv')
print(new_df)

our_Text = "Mango"
text_embedding = embeddings.embed_query(our_Text)
#print (f"Our embedding is {text_embedding}")

from openai.embeddings_utils import cosine_similarity

df["similarity score"] = df['embedding'].apply(lambda x: cosine_similarity(x, text_embedding))
df = df.sort_values("similarity score", ascending=False).head(10)
print(df[["Words", "similarity score"]])
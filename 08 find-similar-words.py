import os
import my_openai_key
os.environ["OPENAI_API_KEY"] = my_openai_key.open_ai_key

from langchain.embeddings import OpenAIEmbeddings
import streamlit as st

from langchain.embeddings import OpenAIEmbeddings

#FAISS is an open-source library developed by Facebook AI Research for efficient similarity search and clustering of large-scale datasets, particularly with high-dimensional vectors. 
#It provides optimized indexing structures and algorithms for tasks like nearest neighbor search and recommendation systems.
from langchain.vectorstores import FAISS

#load_dotenv() is a function that loads variables from a .env file into environment variables in a Python script. 
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Educate Kids", page_icon=":robot:")
st.header("Hey, Ask me something & I will give out similar things")

embeddings = OpenAIEmbeddings()

from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})
data = loader.load()
st.write(data)

db = FAISS.from_documents(data, embeddings)

#Function to receive input from user and store it in a variable
def get_text():
    input_text = st.text_input("You: ", key= input)
    return input_text

user_input=get_text()
submit = st.button('Find similar Things')  

if submit:
    docs = db.similarity_search(user_input)
    #print(docs)
    st.subheader("Top Matches:")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)

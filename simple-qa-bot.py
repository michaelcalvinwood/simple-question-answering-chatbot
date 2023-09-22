import os
import my_openai_key
from langchain.llms import OpenAI # langchain wrapper for openai

os.environ["OPENAI_API_KEY"] = my_openai_key.open_ai_key

import streamlit as st
st.set_page_config(page_title="Simple Q/A Bot", page_icon=":robot:")
st.header("My Bot:")

from langchain.llms import OpenAI

def load_answer(question):
    llm = OpenAI(model_name="text-davinci-003", temperature=0)
    answer = llm(question)
    return answer

def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()
response = load_answer(user_input)

submit = st.button("Generate")

if submit:
    st.subheader("Answer:")
    st.write(response)
import os
import my_openai_key
os.environ["OPENAI_API_KEY"] = my_openai_key.open_ai_key

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("Hey, I'm your Chat GPT")

# If st.session_state is uninitialized then initialize it
if "sessionMessages" not in st.session_state:
     st.session_state.sessionMessages = [
        SystemMessage(content="You are a helpful assistant.")
    ]
    
def load_answer(question):
    st.session_state.sessionMessages.append(HumanMessage(content=question))
    assistant_answer  = chat(st.session_state.sessionMessages )
    st.session_state.sessionMessages.append(AIMessage(content=assistant_answer.content))
    return assistant_answer.content

def get_text(label="You:", key="user"):
    input_text = st.text_input(label, key=key)
    return input_text

chat = ChatOpenAI(temperature=0)

user_input=get_text()
submit = st.button('Generate')  

if submit:    
    response = load_answer(user_input)
    st.subheader("Answer:")
    st.write(response,key= 1)
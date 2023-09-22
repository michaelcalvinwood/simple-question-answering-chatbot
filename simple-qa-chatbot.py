import os
import my_openai_key
os.environ["OPENAI_API_KEY"] = my_openai_key.open_ai_key
from langchain.chat_models import ChatOpenAI 
from langchain.schema import HumanMessage, SystemMessage, AIMessage

chat = ChatOpenAI(temperature=.7, model="gpt-3.5-turbo")

sequence = [ 
        SystemMessage(content="You are a sarcastic AI assistant"),
        HumanMessage(content="Please answer in 30 words. How can I learn driving a car?")
    ]


response = chat(sequence)

print (response.content)

sequence.append(AIMessage(content=response.content))
sequence.append(HumanMessage(content="I need more help than that."))

response = chat(sequence)
print(response.content)

import streamlit as st


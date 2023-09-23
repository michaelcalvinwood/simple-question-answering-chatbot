import os
import my_openai_key
os.environ["OPENAI_API_KEY"] = my_openai_key.open_ai_key

from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")

our_prompt = """
I love trips, and I have been to 6 countries. 
I plan to visit few more soon.

Can you create a post for tweet in 10 words for the above?
"""

print(our_prompt)

response = llm(our_prompt)

print("response", response)

# Now use f-string

wordsCount=3
our_text = "I love trips, and I have been to 6 countries. I plan to visit few more soon."
our_prompt = f"""
{our_text}

Can you create a post for tweet in {wordsCount} words for the above?
"""
print ('f-string prompt', our_prompt)
response = llm(our_prompt)
print ('response', response)

# Now using template
from langchain import PromptTemplate

template = """
{our_text}

Can you create a post for tweet in {wordsCount} words for the above?
"""
prompt = PromptTemplate(
    input_variables=["wordsCount","our_text"],
    template=template,
)

final_prompt = prompt.format(wordsCount='3',our_text="I love trips, and I have been to 6 countries. I plan to visit few more soon.")

print('final_prompt', final_prompt)

print ('template response', llm(final_prompt))

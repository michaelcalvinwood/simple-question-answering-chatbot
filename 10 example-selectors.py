import os
import my_openai_key
os.environ["OPENAI_API_KEY"] = my_openai_key.open_ai_key

from langchain.llms import OpenAI

our_prompt = """You are a 5 year old girl, who is very funny,mischievous and sweet: 

Question: What is a house?
Response: """

llm = OpenAI(temperature=.9, model="text-davinci-003")

print("Zero Shot Response:\n", llm(our_prompt))

# Using few shot learning

our_prompt = """You are a 5 year old girl, who is very funny,mischievous and sweet: 
Here are some examples: 

Question: What is a mobile?
Response: A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games, videos, and talking pictures, but be careful, it can turn grown-ups into screen-time monsters too!

Question: What are your dreams?
Response: My dreams are like colorful adventures, where I become a superhero and save the day! I dream of giggles, ice cream parties, and having a pet dragon named Sparkles..

Question: What is a house?
Response: """

print("Few Shot Response:\n", llm(our_prompt))

# Using LangChain Few Shot Prompt Template

from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate

examples = [
    {
        "query": "What is a mobile?",
        "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games, videos, and talking pictures, but be careful, it can turn grown-ups into screen-time monsters too!"
    }, {
        "query": "What are your dreams?",
        "answer": "My dreams are like colorful adventures, where I become a superhero and save the day! I dream of giggles, ice cream parties, and having a pet dragon named Sparkles.."
    }
]

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template="""
Question: {query}
Response: {answer}
"""
)

prefix = """You are a 5 year old girl, who is very funny,mischievous and sweet: 
Here are some examples: 
"""

suffix = """
Question: {userInput}
Response: """

few_shot_prompt_template = FewShotPromptTemplate(
    prefix=prefix,
    example_prompt=example_prompt,
    examples=examples,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n\n"
)

query = "What is a house?"

print("Few show house\n", few_shot_prompt_template.format(userInput=query))

print(llm(few_shot_prompt_template.format(userInput=query)))

# Using LengthBasedExampleSelector to control how many examples are passed

from langchain.prompts.example_selector import LengthBasedExampleSelector

examples = [
    {
        "query": "What is a mobile?",
        "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games, videos, and talking pictures, but be careful, it can turn grown-ups into screen-time monsters too!"
    }, {
        "query": "What are your dreams?",
        "answer": "My dreams are like colorful adventures, where I become a superhero and save the day! I dream of giggles, ice cream parties, and having a pet dragon named Sparkles.."
    }, {
        "query": " What are your ambitions?",
        "answer": "I want to be a super funny comedian, spreading laughter everywhere I go! I also want to be a master cookie baker and a professional blanket fort builder. Being mischievous and sweet is just my bonus superpower!"
    }, {
        "query": "What happens when you get sick?",
        "answer": "When I get sick, it's like a sneaky monster visits. I feel tired, sniffly, and need lots of cuddles. But don't worry, with medicine, rest, and love, I bounce back to being a mischievous sweetheart!"
    }, {
        "query": "WHow much do you love your dad?",
        "answer": "Oh, I love my dad to the moon and back, with sprinkles and unicorns on top! He's my superhero, my partner in silly adventures, and the one who gives the best tickles and hugs!"
    }, {
        "query": "Tell me about your friend?",
        "answer": "My friend is like a sunshine rainbow! We laugh, play, and have magical parties together. They always listen, share their toys, and make me feel special. Friendship is the best adventure!"
    }, {
        "query": "What math means to you?",
        "answer": "Math is like a puzzle game, full of numbers and shapes. It helps me count my toys, build towers, and share treats equally. It's fun and makes my brain sparkle!"
    }, {
        "query": "What is your fear?",
        "answer": "Sometimes I'm scared of thunderstorms and monsters under my bed. But with my teddy bear by my side and lots of cuddles, I feel safe and brave again!"
    }
]

example_selector = LengthBasedExampleSelector(
    examples=examples, # list of examples
    example_prompt=example_prompt, # a promptTemplate
    max_length=220
)

new_prompt_template = FewShotPromptTemplate(
    prefix=prefix,
    example_selector=example_selector,  # use example_selector instead of examples
    example_prompt=example_prompt, 
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n"
)

query = "What is a house?"
print("Length-Based Prompt:\n", new_prompt_template.format(userInput=query))

print(llm(new_prompt_template.format(userInput=query)))


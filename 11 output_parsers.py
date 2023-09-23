import os
import my_openai_key
os.environ["OPENAI_API_KEY"] = my_openai_key.open_ai_key

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# CSV Output

from langchain.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

print("Format Instructions: ", format_instructions)

prompt = PromptTemplate(
    template="Provide 5 examples of {query}.\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions}
)

llm = OpenAI(temperature=.9, model="text-davinci-003")

prompt = prompt.format(query="Currencies")

print("Comma Prompt:\n", prompt)

output = llm(prompt)
print("Comma Response:\n", output)

# JSON Output

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="currency", description="answer to the user's question"),
    ResponseSchema(name="abbreviation", description="Whats the abbreviation of that currency")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print("JSON Format Instructions:\n", format_instructions)

prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions}
)
prompt = prompt.format(query="what's the currency of America?")

print("JSON Prompt:\n", prompt)

output = llm(prompt)
print("JSON Output:\n", output)





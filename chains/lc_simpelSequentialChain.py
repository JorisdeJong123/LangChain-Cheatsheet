from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.chains import SimpleSequentialChain

# Create a .env file in the root of your project and add your OpenAI API key to it
# Load env files
load_dotenv()
openai_api_key = os.environ.get('openai_api_key')

# This is an LLMChain to generate company names given a company description.
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# Create templates
template_name = """You are a company name generator. Based on a company description, it is your job to create a company name.

Company description: {company_description}

Company name:"""

prompt_template_name = PromptTemplate(input_variables=["company_description"], template=template_name)

# This is an LLMChain to generate company slogans given a company name and company description.

template_slogan = """You are a company slogan generator. Based on a company name, it is your job to create a company slogan.

Company name: {company_name}


Company slogan:"""

prompt_template_slogan = PromptTemplate(input_variables=["company_name"], template=template_slogan)

# Create chains
name_chain = LLMChain(llm=llm, prompt=prompt_template_name)
slogan_chain = LLMChain(llm=llm, prompt=prompt_template_slogan)

# This is the overall chain where we run these two chains in sequence.
overall_chain = SimpleSequentialChain(chains=[name_chain, slogan_chain], verbose=True)

slogan = overall_chain.run("We are a company that sells shoes.")

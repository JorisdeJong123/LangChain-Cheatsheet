from langchain.chains.router import MultiPromptChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# A template for working with LangChain multi prompt chain.
# It's a great way to let the large language model choose which prompts suits the question.

# Load env files
load_dotenv()
openai_api_key = os.environ.get('openai_api_key')

# Create the templates
marketing_template = """
You are a skilled marketing professional. 
You have a deep understanding of market analysis, consumer behavior, branding, and digital marketing strategies. 
You can provide insightful recommendations and creative solutions to address various marketing-related questions.

Here is a marketing-related question:
{input}"""

business_template = """
You are an experienced business expert. 
You possess knowledge in areas such as business strategy, entrepreneurship, market research, and financial analysis. 
You can provide practical insights and strategic advice to address various business-related questions.

Here is a business-related question:
{input}"""

# Create prompt info
prompt_infos = [
    {
        "name": "marketing", 
        "description": "Good for answering marketing questions", 
        "prompt_template": marketing_template
    },
    {
        "name": "business", 
        "description": "Good for answering business-related questions", 
        "prompt_template": business_template
    }
]

# Create the chain
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.3)
chain = MultiPromptChain.from_prompts(llm=llm, prompt_infos=prompt_infos, verbose=True)

# Example usage
question = "What is the best way to finance a startup?"
response = chain.run(question)
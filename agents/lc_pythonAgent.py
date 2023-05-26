from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Load env files
load_dotenv()
openai_api_key = os.environ.get('openai_api_key')

# Create a Python agent
agent_executor = create_python_agent(
    llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo"),
    tool=PythonREPLTool(),
    verbose=True
)

# Run the agent
agent_executor.run("What is the closet prime number to 102?")
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# A LangChain CSV Agent is an agent that uses a CSV file as a knowledge base.

# Create a .env file in the root of your project and add your OpenAI API key to it

# Load env files
load_dotenv()
openai_api_key = os.environ.get('openai_api_key')

# Create a CSV agent, change model_name to the name of the model you want to use.
csv_file_path = "file/path/to/your/csv/file"
agent = create_csv_agent(ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"), path=csv_file_path, verbose=True)

# Run the agent
agent.run("What is the average salary of a Full time ML Engineer in the US?")


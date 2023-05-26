from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# This is a boilerplate file to summarize a PDF document.
# Create a .env file in the same directory as this file and add your OpenAI API key to it.

# Load env files
load_dotenv()
openai_api_key = os.environ.get('openai_api_key')

# Set the path to your PDF file
pdf_file_path = "/path/to/your/pdf/file.pdf"
 
# Create a PDFLoader instance
pdf_loader = PyPDFLoader(pdf_file_path)

# Load the documents
documents = pdf_loader.load()

# Split the documents into chunks.
# I use the TokenTextSplitter as an example, but you can use any text splitter you want.
text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Initilialize LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.3)

# Create the chain
# There's three options for chain_type: 
# "stuff", "map_reduce" and "refine".
chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)

# Run the chain
summary = chain.run(texts)



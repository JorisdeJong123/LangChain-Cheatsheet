from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Load env files
load_dotenv()
openai_api_key = os.environ.get('openai_api_key')

# Load the documents
pdf_file_path = "path/to/your/pdf/file"
loader = PyPDFLoader(pdf_file_path)
documents = loader.load()

# Split the documents into chunks
text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
docsearch = FAISS.from_documents(texts, embeddings)

# Create LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.3)

# Create QA chain
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo"), 
                                 chain_type="stuff", retriever=docsearch.as_retriever())

# Run the agent
query = "What's a reason I might struggle with learning AI?"
answer = qa.run(query)
print(answer)
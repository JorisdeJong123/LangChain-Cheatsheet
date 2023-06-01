from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import os

# Load env files
load_dotenv()
openai_api_key = os.environ.get('openai_api_key')

# Load YouTube transcript
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=D4JkWoUovdo")
transcript = loader.load()

# Split the transcript into chunks.
# I use the TokenTextSplitter as an example, but you can use any text splitter you like.
text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=200)
texts = text_splitter.split_documents(transcript)

# Initilialize LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.3)

# Create the chain
# There's three options for chain_type: 
# "stuff", "map_reduce" and "refine".
chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)

# Run the chain
summary = chain.run(texts)
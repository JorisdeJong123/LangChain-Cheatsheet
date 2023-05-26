from langchain.callbacks import OpenAICallbackHandler, get_openai_callback

# This is a boilerplate template to showcase how to use the OpenAICallbackHandler.

with get_openai_callback() as cb:
    # Include the code you want to track here
    # ...

    
    # Print the OpenAI cost
    print(cb) 
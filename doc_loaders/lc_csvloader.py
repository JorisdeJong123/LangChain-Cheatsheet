from langchain.document_loaders.csv_loader import CSVLoader

# Instantiate a CSVLoader object with the path to your CSV file
loader = CSVLoader(file_path="/Users/jorisdejong/Documents/GitHub/LangChain-cheat/files/datasets/ds_salaries.csv")

# Call the load() method to load the data from the CSV file into a list of Document objects
documents = loader.load()

# Print out the contents of each Document object
for doc in documents:
    print(doc.page_content)
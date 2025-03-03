import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.settings import Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.gemini import Gemini
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Define the folder containing PDF files
PDF_FOLDER_PATH = "mydata"

# Step 2: Load a local embedding model using Langchain's HuggingFaceEmbeddings
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# Step 3: Load a local LLM (optional, for generating responses)
# Replace with your preferred model, e.g., GPT-2, LLaMA, etc.
llm = Gemini(model="models/gemini-1.5-pro")

# Step 4: Set up the embed model
Settings.embed_model = embed_model

# Step 5: Load PDF files from the folder
def load_pdf_files(folder_path):
    reader = SimpleDirectoryReader(input_dir=folder_path, recursive=True)
    documents = reader.load_data()
    return documents

# Step 6: Create a vector index
def create_index(documents):
    index = VectorStoreIndex.from_documents(documents)
    return index

# Step 7: Query the index
def query_index(index, query, llm):
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(query)
    return response

# Step 8: Main function
def main():
    # Load PDF files
    print("Loading PDF files...")
    documents = load_pdf_files(PDF_FOLDER_PATH)

    # Create index
    print("Creating index...")
    index = create_index(documents)

    # Example query
    query = "What is the main topic discussed in the documents?"
    print(f"Query: {query}")
    response = query_index(index, query, llm)
    print(f"Answer: {response}")

if __name__ == "__main__":
    main()
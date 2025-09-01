import pandas as pd
import sys, os, traceback
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
import google.generativeai as genai


def preprocess_excel(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Create LlamaIndex documents with clear price focus
    documents = []
    for _, row in df.iterrows():
        product_name = row.get('product info', 'Unknown Product')
        price = row.get('price(rs)', 'Price not available')
        text = f"The price of {product_name} is {price}."
        documents.append(Document(text=text))
    
    return documents

def main():
    print("\n\nProduct Price Finder is ready. Initializing AI models...")

    # Retrieve Google API Key from environment variables
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API Key not found. Please set 'GOOGLE_API_KEY' as an environment variable.")

    genai.configure(api_key=GOOGLE_API_KEY)

    # dataset path
    excel_file_path = r"data_set/data_file.xlsx"
    documents = preprocess_excel(excel_file_path)

    # Initialize Gemini models
    model = Gemini(models="gemini-pro", api_key=GOOGLE_API_KEY)
    gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=GOOGLE_API_KEY)

    # Configure Settings
    Settings.llm = model
    Settings.embed_model = gemini_embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)

    # creating the Vector Store Index for retrieve information
    index = VectorStoreIndex.from_documents(documents, service_context=Settings)
    index.storage_context.persist()

    # query engine for generating response
    query_engine = index.as_query_engine()

    # Interactive loop for querying product prices
    while True:
        # Get user input
        user_input = input("Enter product query (or 'exit' to quit) 📊 : ")
        
        # Check for exit condition
        if user_input.lower() in ['exit', 'quit', 'bye','ex']:
            print("Thank you for using the Product Price Finder. Goodbye!")
            break
        try:
            response = query_engine.query(
                f"Find the exact price and product details for: {user_input}. "
                f"Respond with clear, concise product information."
            )
            
            # Validate and display response
            if response.response and len(response.response.strip()) > 10:
                print("\n📊 Product Details:")
                print(response.response)
            else:
                print(f"❌ No specific information found for '{user_input}'.")
        
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

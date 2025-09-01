import os
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.readers.file import PandasExcelReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


# Settings for the chatbot
class ChatbotSettings:
    llm = None
    embed_model = None
    node_parser = None

def initialize_chatbot():
    """
    Initialize the chatbot by setting up the Gemini API, embeddings, and vector store index.
    """
    # Load Google API Key from environment variables
    google_api_key = os.getenv("Google_Api_Key")    
    if not google_api_key:
        raise ValueError("Google API Key not found. Please set 'GOOGLE_API_KEY' as an environment variable.")
    
    # Initialize Gemini model and embedding model
    model = Gemini(models="gemini-pro", api_key=google_api_key)
    gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=google_api_key)

    # Set up the settings
    ChatbotSettings.llm = model
    ChatbotSettings.embed_model = gemini_embed_model
    ChatbotSettings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    # Load data from Excel file
    reader = PandasExcelReader()
    documents = reader.load_data(file="./data_file.xlsx")  # Ensure this file exists in the same directory

    # Create a vector store index from the documents
    index = VectorStoreIndex.from_documents(documents, service_context=ChatbotSettings)
    
    # Persist the index for future use
    index.storage_context.persist()

    return index

def main():
    """
    Main function to handle interactive user input for querying product prices.
    """
    print("Initializing chatbot...")
    
    try:
        # Initialize the chatbot and load the vector store index
        index = initialize_chatbot()
        query_engine = index.as_query_engine()

        print("Welcome to the Product Price Chatbot! Ask me about product prices.")
        print("Type 'exit' to end the chat.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            # Query the engine with user input and return a response
            response = query_engine.query(user_input)
            print(f"Chatbot: {response.response}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

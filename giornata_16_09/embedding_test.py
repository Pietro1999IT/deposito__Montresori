from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()
azure_endpoint = os.getenv("EMBEDDING_MODEL_ENDPOINT")
api_key = os.getenv("EMBEDDING_MODEL_KEY")
user_input = input("inserisci la frase:")
 
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version="2024-12-01-preview"
)
 
response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=user_input
        )
embedding = response.data[0].embedding

print(f"L'embedding generato è:\n\n{embedding}")
 

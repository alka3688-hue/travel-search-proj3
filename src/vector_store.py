import os
from src.config import Config
from langchain_community.vectorstores import AzureSearch

def get_vector_store(embedding_function):
    """
    Returns Azure AI Search vector store (no ChromaDB option).
    
    Args:
        embedding_function: The LangChain embedding function to use.
    """
    endpoint = Config.AZURE_SEARCH_ENDPOINT
    key = Config.AZURE_SEARCH_KEY
    index_name = Config.AZURE_SEARCH_INDEX_NAME
    
    if not endpoint or not key:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY must be set.")
        
    vector_store = AzureSearch(
        azure_search_endpoint=endpoint,
        azure_search_key=key,
        index_name=index_name,
        embedding_function=embedding_function.embed_query
    )
    
    print(f"Initialized Azure AI Search (LangChain) for index '{index_name}'")
    return vector_store
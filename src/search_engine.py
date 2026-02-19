import mlflow
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config import Config
from governance.governance_gate import GovernanceGate
from src.vector_store import get_vector_store

class TravelSearchEngine:
    """RAG-powered search engine for travel queries (text-only, no vision)"""
    
    def __init__(self):
        self.governance_gate = GovernanceGate()
        
        self.llm = AzureChatOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            deployment_name=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=1
        )
        
        self.embeddings = AzureOpenAIEmbeddings(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            azure_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        )
        
        # Initialize Vector Store (Azure Search only)
        self.vector_store = get_vector_store(self.embeddings)
    
    def search_by_text(self, query_text: str, k: int = 5):
        """
        Search for travel information using a text query.
        """
        print(f"DEBUG: Starting mlflow: {query_text}")
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        print(f"DEBUG: Starting search for query: {query_text}")
        with mlflow.start_run(run_name="search_travel_info"):
            print(f"DEBUG: Text Query: {query_text}")
            
            # Governance check on input
            gov_check = self.governance_gate.validate_input(query_text)
            print(f"DEBUG: Governance check result for query '{query_text}': {gov_check}")
            if not gov_check['passed']:
                mlflow.log_event("GovernanceCheckFailed", {"violations": gov_check['violations']})
                return [], "Query blocked by security checks."

            mlflow.log_param("k", k)
            mlflow.log_param("query_text", query_text)
            
            # Perform similarity search
            docs = self.vector_store.similarity_search(query_text, k=k)
            
            mlflow.log_metric("results_count", len(docs))
            
            return docs, query_text

    def synthesize_response(self, docs, user_query):
        """
        Generate a conversational response based on retrieved documents.
        """
        #mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="synthesize_response"):
            if not docs:
                return "No relevant travel information found for your query."
            
            context = "\n".join([f"- {doc.page_content} (Source: {doc.metadata.get('source', 'Unknown')})" for doc in docs])
            
            prompt = f"""
            You are a helpful travel assistant for Wanderlust Travels, an online travel agency.
            Use the following information from our knowledge base to answer the customer's question.
            
            Knowledge Base Information:
            {context}
            
            Customer Question: "{user_query}"
            
            Please provide a clear, helpful, and accurate answer based on the information above.
            If the information is not sufficient, let the customer know and provide general guidance.
            """
            
            response = self.llm.invoke(prompt).content
            
            # Governance Check on Output
            gov_check = self.governance_gate.validate_output(response)
            if not gov_check['passed']:
                return f"I generated a response but it didn't pass safety checks. Please rephrase your question."
            
            #mlflow.log_text(response, "final_response.txt")
            return response
import os
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class LLMFactory:
    """Factory class to create different LLM instances"""
    
    @staticmethod
    def create_llm(llm_type: str = "gemini", **kwargs):
        if llm_type == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7,
                **kwargs
            )
        elif llm_type == "openai":
            return ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.7,
                **kwargs
            )
        elif llm_type == "llama":
            return LlamaCpp(
                model_path=kwargs.get("model_path", "models/llama-2-7b-chat.gguf"),
                temperature=0.7,
                max_tokens=2000,
                top_p=1,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

class IMDBQueryEngine:
    def __init__(
        self,
        collection_name: str = "imdb_movies",
        llm_type: str = "gemini",
        **llm_kwargs
    ):
        # Initialize vector store
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY", "")
        )
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embeddings=self.embeddings
        )
        
        # Initialize LLM and chain
        self.llm = LLMFactory.create_llm(llm_type, **llm_kwargs)
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self) -> RetrievalQA:
        prompt_template = """You are a helpful movie expert assistant. Use the following context to answer the question.
        If you don't have enough information, just say so.
        
        Context: {context}
        
        Question: {query}
        
        Provide a detailed and accurate answer based on the context provided. If relevant, include:
        - Movie titles
        - Release years
        - Directors
        - Ratings
        - Box office numbers
        - Other relevant details from the context
        
        If you don't have enough information, please say so rather than making assumptions.
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "query"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False
        )

    def query(self, query: str) -> Dict:
        """Process a single question using RAG"""
        try:
            response = self.qa_chain.invoke({"query": query})
            
            return {
                "answer": response.get("result", "I couldn't find an answer to that question.")
            }
            
        except Exception as e:
            print(f"Error in query processing: {e}")
            return {
                "answer": "Sorry, I encountered an error. Please try rephrasing your question."
            } 
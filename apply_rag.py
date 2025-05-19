"""
Main RAG implementation class for document ingestion and querying.
"""

import logging
import os
from typing import Optional, Dict, Any, List, Tuple, Literal

from rag_utils import (
    TextChunk,
    ProviderType,
    split_text_into_chunks,
    get_embedding,
    semantic_search,
    get_completion,
    get_completion_from_openai
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleDocumentRAG:
    """
    RAG implementation for a single document.
    Ingests a document, vectorizes it, and provides querying capabilities.
    Supports different providers: OpenAI, Azure OpenAI, and Groq.
    """
    
    def __init__(
        self,
        document_text: str,
        provider: ProviderType = "openai",
        api_key: Optional[str] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        similarity_top_k: int = 3,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_embedding_deployment: Optional[str] = None,
        azure_api_version: Optional[str] = None,
    ):
        """
        Initialize the RAG system with a document.
        
        Args:
            document_text: The text content of the document to ingest
            provider: The provider to use ("openai", "azure", or "groq")
            api_key: Provider API key (optional, will use env var if not provided)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            similarity_top_k: Number of top similar chunks to consider during retrieval
            model: Name of the LLM model to use (provider-specific)
            embedding_model: Name of the embedding model to use (provider-specific)
            azure_endpoint: Azure OpenAI endpoint (for Azure only)
            azure_deployment: Azure OpenAI deployment name for chat (for Azure only)
            azure_embedding_deployment: Azure OpenAI deployment name for embeddings (for Azure only)
            azure_api_version: Azure OpenAI API version (for Azure only)
        """
        self.document_text = document_text
        self.provider = provider
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        
        # Set default models based on provider
        if provider == "openai":
            self.model = model or "gpt-3.5-turbo-0125"
            self.embedding_model = embedding_model or "text-embedding-3-small"
        elif provider == "azure":
            self.model = model
            self.embedding_model = embedding_model
            self.azure_endpoint = azure_endpoint
            self.azure_deployment = azure_deployment
            self.azure_embedding_deployment = azure_embedding_deployment or azure_deployment
            self.azure_api_version = azure_api_version or "2023-05-15"
        elif provider == "groq":
            self.model = model or "llama3-8b-8192"
            # Groq doesn't have embedding models, use a fallback method or OpenAI if available
            self.embedding_model = embedding_model or "text-embedding-ada-002"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Process the document
        self._process_document()
        
        logger.info(f"SingleDocumentRAG initialized successfully with provider '{provider}'")
    
    def _process_document(self) -> None:
        """
        Process the document: split into chunks and create embeddings.
        """
        # Split document into chunks
        self.chunks = split_text_into_chunks(
            self.document_text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Generate embeddings for each chunk
        for chunk in self.chunks:
            try:
                # Set up embedding parameters based on provider
                kwargs = {
                    "text": chunk.text,
                    "provider": self.provider,
                    "api_key": self.api_key,
                    "model": self.embedding_model,
                }
                
                # Add Azure-specific parameters if needed
                if self.provider == "azure":
                    kwargs.update({
                        "azure_endpoint": self.azure_endpoint,
                        "azure_deployment": self.azure_embedding_deployment,
                        "azure_api_version": self.azure_api_version,
                    })
                    
                chunk.embedding = get_embedding(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for chunk: {str(e)}")
                # Continue with unembedded chunk - will be embedded on-the-fly during retrieval
        
        logger.info(f"Document processed: {len(self.chunks)} chunks indexed")
    
    def query(self, query_text: str, similarity_top_k: Optional[int] = None) -> str:
        """
        Query the document with a natural language question.
        
        Args:
            query_text: The query or question to ask
            similarity_top_k: Override the default number of top similar chunks to retrieve
            
        Returns:
            Answer text
        """
        try:
            if not query_text.strip():
                return "Query is empty. Please provide a valid question."
            
            # Set the number of chunks to retrieve
            top_k = similarity_top_k or self.similarity_top_k
            
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query_text, top_k)
            
            if not relevant_chunks:
                return "No relevant information found in the document."
            
            # Build context from the retrieved chunks
            context = "\n\n".join([chunk[0].text for chunk in relevant_chunks])
            
            # Set up completion parameters based on provider
            kwargs = {
                "prompt": query_text,
                "context": context,
                "provider": self.provider,
                "api_key": self.api_key,
                "model": self.model,
            }
            
            # Add Azure-specific parameters if needed
            if self.provider == "azure":
                kwargs.update({
                    "azure_endpoint": self.azure_endpoint,
                    "azure_deployment": self.azure_deployment,
                    "azure_api_version": self.azure_api_version,
                })
            
            # Get completion from provider
            response = get_completion(**kwargs)
            
            logger.info(f"Query executed successfully using {self.provider}: '{query_text}'")
            return response
        
        except Exception as e:
            error_msg = f"Error during query execution: {str(e)}"
            logger.error(error_msg)
            return f"An error occurred: {error_msg}"
    
    def retrieve_relevant_chunks(self, query_text: str, similarity_top_k: Optional[int] = None) -> List[Tuple[TextChunk, float]]:
        """
        Retrieve relevant chunks for a query without generating an answer.
        Useful for debugging or understanding retrieval process.
        
        Args:
            query_text: The query text
            similarity_top_k: Override the default number of top similar chunks to retrieve
            
        Returns:
            List of (chunk, score) tuples
        """
        if not query_text.strip():
            return []
        
        top_k = similarity_top_k or self.similarity_top_k
        
        # Set up semantic search parameters based on provider
        kwargs = {
            "query": query_text,
            "chunks": self.chunks,
            "top_k": top_k,
            "provider": self.provider,
            "api_key": self.api_key,
            "model": self.embedding_model,
        }
        
        # Add Azure-specific parameters if needed
        if self.provider == "azure":
            kwargs.update({
                "azure_endpoint": self.azure_endpoint,
                "azure_deployment": self.azure_embedding_deployment,
                "azure_api_version": self.azure_api_version,
            })
            
        # Perform semantic search
        return semantic_search(**kwargs)
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ingested document.
        
        Returns:
            Dictionary with document statistics
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "total_chunks": len(self.chunks),
            "total_text_length": sum(len(chunk.text) for chunk in self.chunks),
            "avg_chunk_size": sum(len(chunk.text) for chunk in self.chunks) / max(1, len(self.chunks)),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedded_chunks": sum(1 for chunk in self.chunks if chunk.embedding is not None),
        }

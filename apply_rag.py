"""
Main RAG implementation class for document ingestion and querying.
"""

import logging
import os
import json
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
        document_id: str,
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
        vector_store_folder: str = "vector_store",
        document_text: Optional[str] = None
    ):
        """
        Initialize the RAG system with a document.
        
        Args:
            document_id: Unique identifier for the document
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
            vector_store_folder: Folder to store/load vector embeddings
            document_text: Optional document text (if None, will be loaded from document_id)
        """
        self.document_id = document_id
        self.document_text = document_text or self._get_document_text(document_id)
        self.provider = provider
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.vector_store_folder = vector_store_folder
        
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
        
        # Ensure vector store folder exists
        os.makedirs(self.vector_store_folder, exist_ok=True)
        
        # Process the document or load existing vectors
        self._process_document()
        
        logger.info(f"SingleDocumentRAG initialized successfully with provider '{provider}'")
    
    def _get_document_text(self, document_id: str) -> str:
        """
        Convert document ID to text. This is a placeholder method that you can 
        replace with actual implementation to fetch documents from a database or filesystem.
        
        Args:
            document_id: The unique identifier for the document
            
        Returns:
            The text content of the document
        """
        # Placeholder implementation - in a real system, this would fetch the document
        # from a database, API, or file system based on the ID
        logger.info(f"Fetching document text for ID: {document_id}")
        
        # Example placeholder implementation:
        # You could replace this with actual database queries or file loading
        if document_id == "sample_doc":
            return """# Artificial Intelligence

Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to human or animal intelligence. 
"AI" may also refer to the field of science concerned with building artificial intelligence systems.

## Types of AI

AI can be categorized as either weak AI or strong AI:

1. **Weak AI** (also called narrow AI) is designed and trained to complete a specific task. Industrial robots and virtual personal assistants, such as Apple's Siri, use weak AI.

2. **Strong AI** (or artificial general intelligence) is a theoretical type of AI where the machine would have an intelligence equal to humans; it would have a self-aware consciousness with the ability to solve problems, learn, and plan for the future.

## Machine Learning

Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to perform specific tasks without explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence.

### Deep Learning

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised."""
        else:
            # Return a default message for unknown document IDs
            return f"This is placeholder text for document ID: {document_id}"
    
    def _get_vector_store_path(self) -> str:
        """
        Get the path to the vector store file for the current document.
        
        Returns:
            Path to the vector store file
        """
        # Create a safe filename from document ID
        safe_id = "".join(c if c.isalnum() else "_" for c in self.document_id)
        return os.path.join(self.vector_store_folder, f"{safe_id}_vectors.json")
    
    def _save_vectors(self) -> None:
        """
        Save vector embeddings to disk.
        """
        vector_path = self._get_vector_store_path()
        
        # Prepare data for serialization
        serializable_chunks = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                serializable_chunks.append({
                    "id": chunk.id,
                    "text": chunk.text,
                    "embedding": chunk.embedding,
                    "metadata": chunk.metadata
                })
        
        # Save to file
        with open(vector_path, 'w') as f:
            json.dump(serializable_chunks, f)
            
        logger.info(f"Saved vector store to {vector_path} with {len(serializable_chunks)} chunks")
    
    def _load_vectors(self) -> List[TextChunk]:
        """
        Load vector embeddings from disk.
        
        Returns:
            List of TextChunk objects with embeddings
        """
        vector_path = self._get_vector_store_path()
        
        if not os.path.exists(vector_path):
            logger.info(f"No existing vector store found at {vector_path}")
            return []
        
        try:
            with open(vector_path, 'r') as f:
                data = json.load(f)
            
            chunks = []
            for item in data:
                chunk = TextChunk(
                    text=item["text"],
                    embedding=item["embedding"],
                    metadata=item.get("metadata", {})
                )
                chunks.append(chunk)
                
            logger.info(f"Loaded {len(chunks)} chunks from vector store at {vector_path}")
            return chunks
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return []
    
    def _process_document(self) -> None:
        """
        Process the document: split into chunks and create embeddings.
        If vectors for this document ID already exist, load them instead.
        """
        # Try to load existing vectors first
        existing_chunks = self._load_vectors()
        
        if existing_chunks:
            # Use existing vectors if available
            self.chunks = existing_chunks
            logger.info(f"Using existing vector store with {len(self.chunks)} chunks")
            return
        
        # No existing vectors found, create new ones
        logger.info(f"Creating new vector store for document ID: {self.document_id}")
        
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
        
        # Save the vectors for future use
        self._save_vectors()
        
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

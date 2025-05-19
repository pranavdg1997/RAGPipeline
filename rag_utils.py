"""
Utility functions for RAG implementation using OpenAI/Azure/Groq and simple vector search.
"""
import os
import logging
import re
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Literal
import json
import openai
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default models
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo-0125"
DEFAULT_AZURE_OPENAI_MODEL = "gpt-35-turbo"  # Example deployment name for Azure
DEFAULT_GROQ_MODEL = "llama3-8b-8192"

# Provider types
ProviderType = Literal["openai", "azure", "groq"]

class TextChunk:
    """
    A class to represent a chunk of text with its embedding.
    """
    def __init__(self, text: str, embedding: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
        # Create a unique ID for the chunk
        self.id = hashlib.md5(text.encode()).hexdigest()

def get_embedding(
    text: str, 
    provider: ProviderType = "openai",
    api_key: Optional[str] = None, 
    model: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_deployment: Optional[str] = None,
    azure_api_version: Optional[str] = None
) -> List[float]:
    """
    Get embedding vector for text using various provider APIs.
    
    Args:
        text: The text to get embedding for
        provider: The provider to use ("openai", "azure", or "groq")
        api_key: Provider API key (optional, will use env var if not provided)
        model: The embedding model to use
        azure_endpoint: Azure OpenAI endpoint (for Azure only)
        azure_deployment: Azure OpenAI deployment name (for Azure only)
        azure_api_version: Azure OpenAI API version (for Azure only)
        
    Returns:
        Embedding vector
    """
    # Clean up text
    text = text.replace("\n", " ")
    
    if provider == "openai":
        # OpenAI embeddings
        model = model or "text-embedding-3-small"
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not provided and not found in environment, using fallback embedding")
            return create_basic_embedding(text)
        
        try:
            openai.api_key = api_key
            try:
                response = openai.embeddings.create(input=[text], model=model)
                return response.data[0].embedding
            except (AttributeError, openai.OpenAIError) as e:
                # Fall back to older OpenAI API version or catch API errors
                logger.warning(f"Error with OpenAI embeddings API: {e}")
                try:
                    response = openai.Embedding.create(input=[text], model=model)
                    return response['data'][0]['embedding']
                except Exception as e2:
                    logger.warning(f"Failed to get OpenAI embeddings: {e2}")
                    return create_basic_embedding(text)
        except Exception as e:
            logger.warning(f"OpenAI embedding error: {e}, using fallback embedding")
            return create_basic_embedding(text)
        
    elif provider == "azure":
        # Azure OpenAI embeddings
        azure_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        
        if not azure_api_key:
            raise ValueError("Azure OpenAI API key not provided and not found in environment")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint not provided and not found in environment")
        if not azure_deployment:
            raise ValueError("Azure OpenAI deployment name not provided and not found in environment")
        
        # Configure Azure OpenAI client
        client = openai.AzureOpenAI(
            api_key=azure_api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint
        )
        
        try:
            response = client.embeddings.create(
                input=[text],
                model=azure_deployment
            )
            return response.data[0].embedding
        except AttributeError:
            # Fall back to older OpenAI API version
            openai.api_type = "azure"
            openai.api_key = azure_api_key
            openai.api_base = azure_endpoint
            openai.api_version = azure_api_version
            
            response = openai.Embedding.create(
                input=[text],
                engine=azure_deployment
            )
            return response['data'][0]['embedding']
        
    elif provider == "groq":
        # Using LlamaIndex integration for Groq
        logger.info("Using LlamaIndex integration for Groq")
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not provided and not found in environment")
            
        try:
            # Try to use OpenAI embeddings if available
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                # Import LlamaIndex OpenAI embedding
                from llama_index.embeddings.openai import OpenAIEmbedding
                
                embed_model = OpenAIEmbedding(
                    model="text-embedding-ada-002",
                    api_key=openai_api_key
                )
                
                # Get embedding
                embedding = embed_model.get_text_embedding(text)
                return embedding
            else:
                # Fall back to simple embedding
                logger.warning("No OpenAI API key found. Using basic text encoding for embeddings with Groq.")
                return create_basic_embedding(text)
        except Exception as e:
            logger.warning(f"Error with LlamaIndex integration for embeddings: {e}")
            logger.warning("Falling back to basic embedding method")
            return create_basic_embedding(text)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Generic error handling
    try:
        # Default handling if specific implementation fails
        pass
    except Exception as e:
        logger.error(f"Failed to get embedding with {provider}: {e}")
        raise

def semantic_search(
    query: str, 
    chunks: List[TextChunk], 
    top_k: int = 3, 
    provider: ProviderType = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_deployment: Optional[str] = None,
    azure_api_version: Optional[str] = None
) -> List[Tuple[TextChunk, float]]:
    """
    Perform semantic search to find most relevant chunks for a query.
    
    Args:
        query: Query text
        chunks: List of text chunks to search
        top_k: Number of top results to return
        provider: The provider to use ("openai", "azure", or "groq")
        api_key: Provider API key (optional, will use env var if not provided)
        model: The embedding model to use (provider-specific)
        azure_endpoint: Azure OpenAI endpoint (for Azure only)
        azure_deployment: Azure OpenAI deployment name (for Azure only)
        azure_api_version: Azure OpenAI API version (for Azure only)
        
    Returns:
        List of (chunk, score) tuples
    """
    if not chunks:
        return []
    
    # Get query embedding
    query_embedding = get_embedding(
        text=query,
        provider=provider,
        api_key=api_key,
        model=model,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        azure_api_version=azure_api_version
    )
    
    # Calculate similarity scores
    results = []
    for chunk in chunks:
        # Ensure chunk has embedding
        if chunk.embedding is None:
            chunk.embedding = get_embedding(
                text=chunk.text,
                provider=provider,
                api_key=api_key,
                model=model,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                azure_api_version=azure_api_version
            )
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_embedding, chunk.embedding)
        results.append((chunk, similarity))
    
    # Sort by similarity score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return results[:top_k]

def create_basic_embedding(text: str, dimension: int = 1536) -> List[float]:
    """
    Create a basic embedding vector from text when API access isn't available.
    This is a simplistic approach and not as effective as model-generated embeddings.
    
    Args:
        text: Text to create embedding for
        dimension: Size of the embedding vector
        
    Returns:
        A list of float values representing the embedding
    """
    # Clean and normalize text
    text = text.lower().strip()
    
    # Simple hash-based approach
    hash_embedding = []
    text_bytes = text.encode('utf-8')
    
    # Use multiple hash seed values to create varied dimensions
    for i in range(dimension):
        # Use different parts of the text and position index to vary the hash
        seed = i % 128  # Different seeds for different dimensions
        hash_input = text_bytes + str(i).encode('utf-8') + str(seed).encode('utf-8')
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16) % 10000 / 10000.0
        hash_embedding.append(hash_val)
    
    # Normalize the vector to have unit length (crucial for cosine similarity)
    embedding_array = np.array(hash_embedding)
    norm = np.linalg.norm(embedding_array)
    if norm > 0:
        embedding_array = embedding_array / norm
    
    return embedding_array.tolist()

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score
    """
    a_np = np.array(a)
    b_np = np.array(b)
    
    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    
    return dot_product / (norm_a * norm_b)

def split_text_into_chunks(text: str, chunk_size: int = 1024, chunk_overlap: int = 20) -> List[TextChunk]:
    """
    Split text into chunks with some overlap.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of TextChunk objects
    """
    chunks = []
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Calculate sentence boundaries (assuming sentences end with period, question mark, or exclamation mark)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed chunk size, add current chunk to results
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(TextChunk(current_chunk.strip()))
            
            # Start new chunk with overlap from previous chunk
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                # Extract last part of previous chunk for overlap
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(TextChunk(current_chunk.strip()))
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def get_completion(
    prompt: str, 
    context: str, 
    provider: ProviderType = "openai",
    api_key: Optional[str] = None, 
    model: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_deployment: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 500
) -> str:
    """
    Get completion from various provider APIs.
    
    Args:
        prompt: The prompt to send to the API
        context: The context information to include
        provider: The provider to use ("openai", "azure", or "groq")
        api_key: Provider API key (optional, will use env var if not provided)
        model: The model to use (provider-specific)
        azure_endpoint: Azure OpenAI endpoint (for Azure only)
        azure_deployment: Azure OpenAI deployment name (for Azure only)
        azure_api_version: Azure OpenAI API version (for Azure only)
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        Completion text
    """
    system_message = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "If the answer cannot be found in the context, say that you don't know the answer."
    )
    user_content = f"Context information:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]
    
    if provider == "openai":
        # OpenAI completions
        model = model or DEFAULT_OPENAI_MODEL
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and not found in environment")
        
        openai.api_key = api_key
        
        try:
            # Try using newer OpenAI API version
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except AttributeError:
            # Fall back to older OpenAI API version
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
            
    elif provider == "azure":
        # Azure OpenAI completions
        azure_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        
        if not azure_api_key:
            raise ValueError("Azure OpenAI API key not provided and not found in environment")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint not provided and not found in environment")
        if not azure_deployment:
            raise ValueError("Azure OpenAI deployment name not provided and not found in environment")
        
        try:
            # Try using newer OpenAI API version with AzureOpenAI client
            client = openai.AzureOpenAI(
                api_key=azure_api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint
            )
            
            response = client.chat.completions.create(
                model=azure_deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except AttributeError:
            # Fall back to older OpenAI API version
            openai.api_type = "azure"
            openai.api_key = azure_api_key
            openai.api_base = azure_endpoint
            openai.api_version = azure_api_version
            
            response = openai.ChatCompletion.create(
                engine=azure_deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
            
    elif provider == "groq":
        # Use LlamaIndex Groq integration
        model = model or DEFAULT_GROQ_MODEL
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not provided and not found in environment")
            
        try:
            # Import LlamaIndex Groq integration
            from llama_index.llms.groq import Groq
            
            # Create Groq LLM instance
            llm = Groq(
                model_name=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Format the prompt with system and user messages
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
            
            # Use LlamaIndex's completion method
            prompt_template = f"{system_msg}\n\n{user_msg}"
            response = llm.complete(prompt_template)
            
            return response.text
        except Exception as e:
            logger.warning(f"Error using LlamaIndex Groq integration: {e}. Falling back to direct Groq API.")
            
            # Fallback to direct Groq API
            try:
                import groq
                client = groq.Client(api_key=api_key)
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Failed to get completion from Groq: {e}")
                return f"Error getting response from Groq: {str(e)}"
    else:
        raise ValueError(f"Unsupported provider: {provider}")
        
    # Generic error handling if needed
    try:
        # Default handling if specific implementation fails
        pass 
    except Exception as e:
        logger.error(f"Failed to get completion from {provider}: {e}")
        raise

# Keep the original function name for backward compatibility
def get_completion_from_openai(
    prompt: str, 
    context: str, 
    api_key: Optional[str] = None, 
    model: str = DEFAULT_OPENAI_MODEL
) -> str:
    """
    Backward compatibility function that calls get_completion with OpenAI provider.
    """
    return get_completion(
        prompt=prompt,
        context=context,
        provider="openai",
        api_key=api_key,
        model=model
    )

"""
Test suite for the RAG implementation using pytest.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from rag_utils import (
    TextChunk,
    get_embedding,
    semantic_search,
    split_text_into_chunks,
    get_completion_from_openai,
    cosine_similarity
)
from apply_rag import SingleDocumentRAG

# Sample document for testing
SAMPLE_DOCUMENT = """
# Artificial Intelligence

Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to human or animal intelligence. 
"AI" may also refer to the field of science concerned with building artificial intelligence systems.

## Types of AI

AI can be categorized as either weak AI or strong AI:

1. **Weak AI** (also called narrow AI) is designed and trained to complete a specific task. Industrial robots and virtual personal assistants, such as Apple's Siri, use weak AI.

2. **Strong AI** (or artificial general intelligence) is a theoretical type of AI where the machine would have an intelligence equal to humans; it would have a self-aware consciousness with the ability to solve problems, learn, and plan for the future.

## Machine Learning

Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to perform specific tasks without explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence.

### Deep Learning

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.
"""

# Mock API key for testing
MOCK_API_KEY = "mock-api-key"
MOCK_EMBEDDING = [0.1] * 1536  # Mock embedding vector

@pytest.fixture
def mock_openai_embedding():
    """Mock OpenAI embedding for testing"""
    with patch('openai.Embedding.create') as mock_embed:
        mock_embed.return_value = {
            'data': [{'embedding': MOCK_EMBEDDING}]
        }
        yield mock_embed

    # Also patch the newer API version
    with patch('openai.embeddings.create') as mock_embed_new:
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=MOCK_EMBEDDING)]
        mock_embed_new.return_value = mock_response
        yield mock_embed_new

@pytest.fixture
def mock_openai_completion():
    """Mock OpenAI chat completion for testing"""
    with patch('openai.ChatCompletion.create') as mock_completion:
        mock_message = MagicMock()
        mock_message.content = "This is a mock response from the LLM."
        mock_choice = MagicMock()
        mock_choice.message = {"content": "This is a mock response from the LLM."}
        mock_completion.return_value = MagicMock(choices=[mock_choice])
        yield mock_completion
    
    # Also patch the newer API version
    with patch('openai.chat.completions.create') as mock_completion_new:
        mock_message = MagicMock()
        mock_message.content = "This is a mock response from the LLM."
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a mock response from the LLM."
        mock_completion_new.return_value = MagicMock(choices=[mock_choice])
        yield mock_completion_new

class TestRagUtils:
    """Tests for rag_utils.py"""
    
    def test_text_chunk_creation(self):
        """Test TextChunk creation"""
        text = "Test chunk"
        chunk = TextChunk(text)
        assert chunk.text == text
        assert chunk.embedding is None
        assert chunk.id is not None
    
    def test_get_embedding(self, mock_openai_embedding):
        """Test if get_embedding returns a vector"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": MOCK_API_KEY}):
            embedding = get_embedding("Test text")
            assert embedding is not None
            assert len(embedding) > 0
    
    def test_get_embedding_no_api_key(self):
        """Test if get_embedding raises an error when no API key is provided"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                get_embedding("Test text")
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        v3 = [1, 1, 0]
        
        # Orthogonal vectors should have similarity 0
        assert cosine_similarity(v1, v2) == 0
        
        # Same vector should have similarity 1
        assert cosine_similarity(v1, v1) == 1
        
        # Check other cases
        assert 0 < cosine_similarity(v1, v3) < 1
    
    def test_split_text_into_chunks(self):
        """Test if split_text_into_chunks splits text correctly"""
        chunks = split_text_into_chunks(SAMPLE_DOCUMENT, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(len(chunk.text) <= 200 for chunk in chunks)
    
    def test_semantic_search(self, mock_openai_embedding):
        """Test semantic search functionality"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": MOCK_API_KEY}):
            chunks = [
                TextChunk("AI is intelligence demonstrated by machines.", MOCK_EMBEDDING),
                TextChunk("Machine learning is a subset of AI.", MOCK_EMBEDDING),
                TextChunk("Deep learning uses neural networks.", MOCK_EMBEDDING)
            ]
            
            results = semantic_search("What is AI?", chunks, top_k=2)
            assert len(results) == 2
            assert all(isinstance(result[0], TextChunk) for result in results)
            assert all(isinstance(result[1], float) for result in results)
    
    def test_get_completion_from_openai(self, mock_openai_completion):
        """Test if get_completion_from_openai returns a completion"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": MOCK_API_KEY}):
            response = get_completion_from_openai(
                prompt="What is AI?",
                context="AI is intelligence demonstrated by machines."
            )
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0

class TestSingleDocumentRAG:
    """Tests for SingleDocumentRAG class"""
    
    @pytest.fixture
    def mock_rag_system(self, mock_openai_embedding, mock_openai_completion):
        """Create a RAG system for testing"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": MOCK_API_KEY}):
            # Mock the embedding creation to avoid API calls
            with patch('rag_utils.get_embedding', return_value=MOCK_EMBEDDING):
                return SingleDocumentRAG(SAMPLE_DOCUMENT, api_key=MOCK_API_KEY)
    
    def test_initialization(self, mock_rag_system):
        """Test if the RAG system initializes correctly"""
        assert mock_rag_system is not None
        assert mock_rag_system.chunks is not None
        assert len(mock_rag_system.chunks) > 0
    
    def test_query(self, mock_rag_system):
        """Test if query returns a response"""
        with patch('rag_utils.get_completion_from_openai', return_value="This is a test response"):
            query = "What is artificial intelligence?"
            response = mock_rag_system.query(query)
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0
    
    def test_empty_query(self, mock_rag_system):
        """Test if empty query returns an error message"""
        response = mock_rag_system.query("")
        assert "Query is empty" in response
    
    def test_retrieve_relevant_chunks(self, mock_rag_system):
        """Test if retrieve_relevant_chunks returns chunks"""
        with patch('rag_utils.semantic_search', return_value=[(mock_rag_system.chunks[0], 0.9)]):
            query = "What is machine learning?"
            chunks = mock_rag_system.retrieve_relevant_chunks(query)
            assert chunks is not None
            assert len(chunks) > 0
        
    def test_empty_retrieve_relevant_chunks(self, mock_rag_system):
        """Test if retrieve_relevant_chunks with empty query returns empty list"""
        chunks = mock_rag_system.retrieve_relevant_chunks("")
        assert chunks == []
    
    def test_get_document_stats(self, mock_rag_system):
        """Test if get_document_stats returns statistics"""
        stats = mock_rag_system.get_document_stats()
        assert stats is not None
        assert "total_chunks" in stats
        assert "total_text_length" in stats
        assert "avg_chunk_size" in stats
        assert "chunk_size" in stats
        assert "chunk_overlap" in stats
    
    def test_query_with_custom_similarity_top_k(self, mock_rag_system):
        """Test if query works with custom similarity_top_k"""
        with patch('rag_utils.get_completion_from_openai', return_value="This is a test response"):
            query = "What is deep learning?"
            response = mock_rag_system.query(query, similarity_top_k=2)
            assert response is not None
            assert isinstance(response, str)

class TestFunctionalIntegration:
    """Functional tests for RAG integration"""
    
    @pytest.mark.skipif(
        os.getenv("OPENAI_API_KEY") is None,
        reason="Skip integration tests when no API key is available"
    )
    def test_end_to_end_workflow(self):
        """
        Test the complete RAG workflow with actual API calls.
        Only runs if OPENAI_API_KEY environment variable is set.
        """
        # Create RAG system with the sample document
        rag = SingleDocumentRAG(SAMPLE_DOCUMENT)
        
        # Test document stats
        stats = rag.get_document_stats()
        assert stats["total_chunks"] > 0
        
        # Test querying
        response = rag.query("Explain the difference between weak AI and strong AI")
        assert response is not None
        assert len(response) > 0
        
        # Test chunk retrieval
        chunks = rag.retrieve_relevant_chunks("What is machine learning?")
        assert len(chunks) > 0

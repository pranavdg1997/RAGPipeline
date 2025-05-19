from flask import Flask, render_template, request, jsonify
import os
import logging
from dotenv import load_dotenv
from apply_rag import SingleDocumentRAG

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-key-for-testing")

# Initialize with a default document if needed
DEFAULT_DOCUMENT = """
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

rag_system = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize_rag():
    global rag_system
    
    try:
        document_text = request.form.get('document_text', DEFAULT_DOCUMENT)
        provider = request.form.get('provider', 'openai')
        
        # Initialize parameters based on provider
        if provider == 'openai':
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return jsonify({
                    'status': 'error',
                    'message': 'OPENAI_API_KEY environment variable is required'
                }), 400
                
            rag_system = SingleDocumentRAG(
                document_text=document_text,
                provider='openai',
                api_key=api_key
            )
            
        elif provider == 'azure':
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            embedding_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
            
            if not all([api_key, endpoint, deployment]):
                missing = []
                if not api_key: missing.append("AZURE_OPENAI_API_KEY")
                if not endpoint: missing.append("AZURE_OPENAI_ENDPOINT")
                if not deployment: missing.append("AZURE_OPENAI_DEPLOYMENT")
                
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required Azure OpenAI environment variables: {", ".join(missing)}'
                }), 400
                
            rag_system = SingleDocumentRAG(
                document_text=document_text,
                provider='azure',
                api_key=api_key,
                azure_endpoint=endpoint,
                azure_deployment=deployment,
                azure_embedding_deployment=embedding_deployment or deployment,
                azure_api_version=api_version
            )
            
        elif provider == 'groq':
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                return jsonify({
                    'status': 'error',
                    'message': 'GROQ_API_KEY environment variable is required'
                }), 400
                
            rag_system = SingleDocumentRAG(
                document_text=document_text,
                provider='groq',
                api_key=api_key
            )
            
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported provider: {provider}'
            }), 400
        
        stats = rag_system.get_document_stats()
        
        return jsonify({
            'status': 'success',
            'message': f'RAG system initialized successfully with {provider}',
            'stats': stats
        })
    
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error initializing RAG system: {str(e)}'
        }), 500

@app.route('/query', methods=['POST'])
def query_rag():
    global rag_system
    
    if not rag_system:
        return jsonify({
            'status': 'error',
            'message': 'RAG system not initialized yet'
        }), 400
    
    try:
        query_text = request.form.get('query_text', '')
        if not query_text:
            return jsonify({
                'status': 'error',
                'message': 'Query text is required'
            }), 400
        
        response = rag_system.query(query_text)
        
        return jsonify({
            'status': 'success',
            'response': response
        })
    
    except Exception as e:
        logger.error(f"Error querying RAG system: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error querying RAG system: {str(e)}'
        }), 500

@app.route('/provider_status', methods=['GET'])
def provider_status():
    """
    Check the status of available providers based on environment variables
    """
    providers = {
        'openai': {
            'available': bool(os.environ.get("OPENAI_API_KEY")),
            'required_env': ['OPENAI_API_KEY']
        },
        'azure': {
            'available': bool(os.environ.get("AZURE_OPENAI_API_KEY") and 
                               os.environ.get("AZURE_OPENAI_ENDPOINT") and
                               os.environ.get("AZURE_OPENAI_DEPLOYMENT")),
            'required_env': ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_DEPLOYMENT']
        },
        'groq': {
            'available': bool(os.environ.get("GROQ_API_KEY")),
            'required_env': ['GROQ_API_KEY']
        }
    }
    
    return jsonify({
        'status': 'success',
        'providers': providers
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
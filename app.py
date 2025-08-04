from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import logging
import os
from research_agent import ResearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize Research Agent
try:
    agent = ResearchAgent()
    logger.info("Research Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Research Agent: {str(e)}")
    agent = None

@app.route('/')
def index():
    """Serve the main page"""
    return """
    <h1>🔬 Research Agent API</h1>
    <p>IBM Granite-powered Research Assistant</p>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><strong>POST /ingest</strong> - Ingest research papers</li>
        <li><strong>POST /ask</strong> - Ask research questions</li>
        <li><strong>POST /summary</strong> - Generate research summaries</li>
        <li><strong>POST /hypotheses</strong> - Generate research hypotheses</li>
        <li><strong>GET /stats</strong> - Get knowledge base statistics</li>
        <li><strong>GET /papers</strong> - List ingested papers</li>
        <li><strong>GET /health</strong> - Health check</li>
    </ul>
    <p><a href="/static/index.html">Try the Web Interface</a></p>
    """

@app.route('/health')
def health():
    """Health check endpoint"""
    if agent is None:
        return jsonify({'status': 'unhealthy', 'error': 'Agent not initialized'}), 500
    
    try:
        stats = agent.get_knowledge_base_stats()
        return jsonify({
            'status': 'healthy',
            'agent_ready': True,
            'knowledge_base': stats
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/ingest', methods=['POST'])
def ingest_document():
    """Ingest a research paper"""
    if agent is None:
        return jsonify({'error': 'Research Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        source_id = data.get('source_id', '').strip()
        source_type = data.get('source_type', 'arxiv').lower()
        
        if not source_id:
            return jsonify({'error': 'source_id is required'}), 400
        
        if source_type not in ['arxiv', 'doi']:
            return jsonify({'error': 'source_type must be "arxiv" or "doi"'}), 400
        
        logger.info(f"Ingesting document: {source_id} ({source_type})")
        result = agent.ingest_document(source_id, source_type)
        
        status_code = 200 if result['status'] == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in ingest endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask a research question"""
    if agent is None:
        return jsonify({'error': 'Research Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        question = data.get('question', '').strip()
        k = min(int(data.get('k', 5)), 10)  # Max 10 documents
        
        if not question:
            return jsonify({'error': 'question is required'}), 400
        
        logger.info(f"Processing question: {question[:100]}...")
        result = agent.ask_question(question, k=k)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/summary', methods=['POST'])
def generate_summary():
    """Generate research summary"""
    if agent is None:
        return jsonify({'error': 'Research Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        topic = data.get('topic', '').strip()
        max_docs = min(int(data.get('max_docs', 10)), 15)
        
        if not topic:
            return jsonify({'error': 'topic is required'}), 400
        
        logger.info(f"Generating summary for: {topic}")
        summary = agent.generate_summary(topic, max_docs=max_docs)
        return jsonify({'summary': summary}), 200
        
    except Exception as e:
        logger.error(f"Error in summary endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/hypotheses', methods=['POST'])
def suggest_hypotheses():
    """Generate research hypotheses"""
    if agent is None:
        return jsonify({'error': 'Research Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        research_area = data.get('research_area', '').strip()
        
        if not research_area:
            return jsonify({'error': 'research_area is required'}), 400
        
        logger.info(f"Generating hypotheses for: {research_area}")
        hypotheses = agent.suggest_hypotheses(research_area)
        return jsonify({'hypotheses': hypotheses}), 200
        
    except Exception as e:
        logger.error(f"Error in hypotheses endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/stats')
def get_stats():
    """Get knowledge base statistics"""
    if agent is None:
        return jsonify({'error': 'Research Agent not initialized'}), 500
    
    try:
        stats = agent.get_knowledge_base_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/papers')
def list_papers():
    """List all ingested papers"""
    if agent is None:
        return jsonify({'error': 'Research Agent not initialized'}), 500
    
    try:
        papers = agent.list_ingested_papers()
        return jsonify({'papers': papers}), 200
    except Exception as e:
        logger.error(f"Error listing papers: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)


from typing import List, Dict
from models import GraniteModels
from vector_store import VectorStore
from document_processor import DocumentProcessor
import logging

logger = logging.getLogger(__name__)

class ResearchAgent:
    """Main Research Agent class"""
    
    def __init__(self):
        """Initialize Research Agent"""
        self.granite = GraniteModels()
        self.llm = self.granite.get_llm()
        self.vector_store = VectorStore()
        self.doc_processor = DocumentProcessor()
        
        # Try to load existing index
        self.vector_store.load_index("research_index")
        
        logger.info("Initialized Research Agent")
        
    def ingest_document(self, source_id: str, source_type: str = 'arxiv'):
        """Ingest a document into the knowledge base"""
        try:
            logger.info(f"Ingesting document: {source_id} ({source_type})")
            
            # Check if document already exists
            existing_docs = [doc for doc in self.vector_store.metadata 
                           if doc['source_id'] == source_id]
            if existing_docs:
                return {
                    'status': 'info',
                    'message': f"Document {source_id} already exists in knowledge base",
                    'chunks_added': 0
                }
            
            # Process document
            chunks = self.doc_processor.process_document(source_id, source_type)
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            # Save updated index
            self.vector_store.save_index("research_index")
            
            return {
                'status': 'success',
                'message': f"Successfully ingested: {chunks[0]['title']}",
                'chunks_added': len(chunks),
                'document_info': {
                    'title': chunks[0]['title'],
                    'authors': chunks[0]['authors'],
                    'source_type': source_type,
                    'source_id': source_id
                }
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document {source_id}: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'chunks_added': 0
            }
    
    def ask_question(self, question: str, k: int = 5) -> Dict:
        """Answer a research question using RAG"""
        try:
            logger.info(f"Answering question: {question[:100]}...")
            
            # Retrieve relevant chunks
            relevant_docs = self.vector_store.search(question, k=k)
            
            if not relevant_docs:
                return {
                    'answer': "I don't have enough information to answer this question. Please ingest some relevant papers first.",
                    'sources': [],
                    'context_used': 0
                }
            
            # Build context from relevant documents
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"[Source {i+1}] {doc['title']}\n{doc['content']}\n")
            
            context = "\n".join(context_parts)
            
            # Generate prompt
            prompt = f"""You are an expert research assistant. Based on the provided academic papers, answer the user's question comprehensively and accurately.

Research Context:
{context}

Question: {question}

Please provide a detailed answer based on the research context above. When referencing information, cite the sources using [Source X] format. Be specific and include relevant details from the papers."""

            # Generate answer
            answer = self.llm.invoke(prompt)
            
            # Prepare unique sources
            sources = []
            seen_sources = set()
            for doc in relevant_docs:
                source_key = (doc['source_id'], doc['title'])
                if source_key not in seen_sources:
                    sources.append({
                        'title': doc['title'],
                        'authors': doc['authors'],
                        'source_id': doc['source_id'],
                        'source_type': doc['source_type'],
                        'url': doc.get('url', ''),
                        'similarity_score': doc['similarity_score']
                    })
                    seen_sources.add(source_key)
            
            return {
                'answer': answer,
                'sources': sources[:3],  # Top 3 unique sources
                'context_used': len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                'answer': f"Sorry, I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'context_used': 0
            }
    
    def generate_summary(self, topic: str, max_docs: int = 10) -> str:
        """Generate a research summary on a topic"""
        try:
            logger.info(f"Generating summary for topic: {topic}")
            
            # Search for relevant documents
            relevant_docs = self.vector_store.search(topic, k=max_docs)
            
            if not relevant_docs:
                return "No relevant documents found for this topic. Please ingest some papers related to this topic first."
            
            # Build context from multiple documents
            context_parts = []
            unique_papers = {}
            
            for doc in relevant_docs:
                paper_key = doc['source_id']
                if paper_key not in unique_papers:
                    unique_papers[paper_key] = {
                        'title': doc['title'],
                        'authors': doc['authors'],
                        'content': []
                    }
                unique_papers[paper_key]['content'].append(doc['content'])
            
            # Create structured context
            for paper_id, paper_info in unique_papers.items():
                context_parts.append(f"Paper: {paper_info['title']}")
                context_parts.append(f"Authors: {', '.join(paper_info['authors'])}")
                context_parts.append(f"Key Content: {' '.join(paper_info['content'][:3])}")  # First 3 chunks
                context_parts.append("---")
            
            context = "\n".join(context_parts)
            
            prompt = f"""Based on the following research papers, write a comprehensive summary about "{topic}".

Research Papers:
{context}

Please provide a structured summary that includes:
1. Overview of the research area
2. Key findings and methodologies mentioned in the papers
3. Important trends or patterns
4. Research gaps that might exist
5. Potential future directions

Make the summary informative and well-organized."""

            summary = self.llm.invoke(prompt)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def suggest_hypotheses(self, research_area: str) -> List[str]:
        """Suggest research hypotheses based on literature"""
        try:
            logger.info(f"Generating hypotheses for: {research_area}")
            
            relevant_docs = self.vector_store.search(research_area, k=8)
            
            if not relevant_docs:
                return ["No relevant literature found. Please ingest papers in this research area first."]
            
            # Build context
            context = "\n\n".join([
                f"Paper: {doc['title']}\nContent: {doc['content']}"
                for doc in relevant_docs
            ])
            
            prompt = f"""Based on the following research literature in {research_area}, suggest 5 novel and testable research hypotheses that could advance the field.

Literature Review:
{context}

Please provide exactly 5 specific, testable hypotheses that:
1. Build upon the existing research
2. Explore new directions or fill gaps
3. Are feasible to test
4. Could contribute meaningfully to the field

Format each hypothesis as: "H1: [hypothesis statement]", "H2: [hypothesis statement]", etc."""

            response = self.llm.invoke(prompt)
            
            # Parse hypotheses
            hypotheses = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if any(line.startswith(f"H{i}:") for i in range(1, 10)):
                    hypotheses.append(line)
                elif any(line.startswith(f"{i}.") for i in range(1, 10)) and len(line) > 10:
                    hypotheses.append(line)
            
            return hypotheses if hypotheses else [response]
            
        except Exception as e:
            logger.error(f"Error generating hypotheses: {str(e)}")
            return [f"Error generating hypotheses: {str(e)}"]
    
    def get_knowledge_base_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        stats = self.vector_store.get_stats()
        return {
            'total_papers': stats['unique_documents'],
            'total_chunks': stats['total_chunks'],
            'index_size': stats['index_size']
        }
    
    def list_ingested_papers(self) -> List[Dict]:
        """List all ingested papers"""
        papers = {}
        for doc in self.vector_store.metadata:
            source_id = doc['source_id']
            if source_id not in papers:
                papers[source_id] = {
                    'source_id': source_id,
                    'title': doc['title'],
                    'authors': doc['authors'],
                    'source_type': doc['source_type'],
                    'url': doc.get('url', ''),
                    'chunk_count': 0
                }
            papers[source_id]['chunk_count'] += 1
        
        return list(papers.values())

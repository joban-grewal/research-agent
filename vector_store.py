import faiss
import numpy as np
import json
import os
import pickle
from typing import List, Dict, Tuple
from models import GraniteModels
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, dimension: int = 768):
        """Initialize vector store"""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self.embeddings_model = None
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embeddings model"""
        try:
            granite = GraniteModels()
            self.embeddings_model = granite.get_embeddings()
            logger.info("Successfully initialized embeddings model")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to vector store"""
        if not chunks:
            logger.warning("No chunks provided to add_documents")
            return
            
        try:
            texts = [chunk['content'] for chunk in chunks]
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            
            # Generate embeddings
            embeddings = self.embeddings_model.embed_documents(texts)
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store metadata
            self.metadata.extend(chunks)
            
            logger.info(f"Added {len(chunks)} chunks to vector store. Total: {len(self.metadata)}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
            
        try:
            logger.info(f"Searching for: {query[:100]}...")
            
            query_embedding = self.embeddings_model.embed_query(query)
            query_array = np.array([query_embedding]).astype('float32')
            
            # Ensure k doesn't exceed available documents
            k = min(k, self.index.ntotal)
            
            distances, indices = self.index.search(query_array, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata) and idx >= 0:
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(distances[0][i])
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def save_index(self, filepath: str):
        """Save FAISS index and metadata"""
        try:
            faiss.write_index(self.index, f"{filepath}.index")
            with open(f"{filepath}.metadata", 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved vector store to {filepath}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_index(self, filepath: str):
        """Load FAISS index and metadata"""
        try:
            if os.path.exists(f"{filepath}.index") and os.path.exists(f"{filepath}.metadata"):
                self.index = faiss.read_index(f"{filepath}.index")
                with open(f"{filepath}.metadata", 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded index with {len(self.metadata)} documents")
                return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
        return False
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            'total_chunks': len(self.metadata),
            'unique_documents': len(set([doc['source_id'] for doc in self.metadata])),
            'index_size': self.index.ntotal
        }

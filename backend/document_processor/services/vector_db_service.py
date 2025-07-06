"""
Vector Database Service for document classification.
Uses FAISS for efficient similarity search and sentence transformers for embeddings.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


class VectorDBService:
    """Service for vector database operations using FAISS."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize vector database service.
        
        Args:
            model_name (str): Name of the sentence transformer model
        """
        self.model_name = model_name
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.document_types = []
        self.file_paths = []
        
        # Paths for saving/loading
        self.vector_db_path = getattr(settings, 'VECTOR_DB_PATH', 'vector_db')
        self.index_path = os.path.join(self.vector_db_path, 'faiss_index.bin')
        self.metadata_path = os.path.join(self.vector_db_path, 'metadata.pkl')
        
        # Ensure directory exists
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Initialize model
        self._load_embedding_model()
        
        # Try to load existing index
        self._load_index()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Try to load sentence transformer with offline mode first
            try:
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                self.embedding_model = SentenceTransformer(self.model_name, local_files_only=True)
                self.use_tfidf = False
                logger.info("Embedding model loaded successfully (offline)")
                return
            except:
                logger.warning("Offline model not found, trying online download...")
                
            # Try online download
            try:
                if 'TRANSFORMERS_OFFLINE' in os.environ:
                    del os.environ['TRANSFORMERS_OFFLINE']
                self.embedding_model = SentenceTransformer(self.model_name)
                self.use_tfidf = False
                logger.info("Embedding model loaded successfully (downloaded)")
                return
            except:
                logger.warning("Online download failed")
                
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
        
        # Fallback to TF-IDF
        logger.warning("Using fallback TF-IDF embedding")
        self.embedding_model = None
        self.use_tfidf = True
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_fitted = False
    
    def _load_index(self):
        """Load existing FAISS index and metadata if available."""
        try:
            # Check if metadata exists (for both FAISS and TF-IDF modes)
            if os.path.exists(self.metadata_path):
                logger.info("Loading existing metadata...")
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.documents = metadata['documents']
                    self.document_types = metadata['document_types']
                    self.file_paths = metadata['file_paths']
                    
                    # Check if it was using TF-IDF
                    was_using_tfidf = metadata.get('use_tfidf', False)
                
                # Load FAISS index only if it exists and we're not using TF-IDF
                if os.path.exists(self.index_path) and not self.use_tfidf:
                    self.index = faiss.read_index(self.index_path)
                    logger.info(f"Loaded FAISS index with {len(self.documents)} documents")
                else:
                    logger.info(f"Loaded TF-IDF metadata with {len(self.documents)} documents")
                    
            else:
                logger.info("No existing metadata found, creating new index")
                self._create_new_index()
                
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        try:
            if self.use_tfidf:
                # For TF-IDF, we'll use a simple approach without FAISS initially
                self.index = None
                self.documents = []
                self.document_types = []
                self.file_paths = []
                logger.info("Created new simple index for TF-IDF")
            else:
                # Get embedding dimension
                sample_text = "sample text"
                sample_embedding = self.embedding_model.encode([sample_text])
                dimension = sample_embedding.shape[1]
                
                # Create FAISS index (using L2 distance)
                self.index = faiss.IndexFlatL2(dimension)
                
                # Initialize empty lists
                self.documents = []
                self.document_types = []
                self.file_paths = []
                
                logger.info(f"Created new FAISS index with dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Error creating new index: {e}")
            # Fallback to simple approach
            self.index = None
            self.documents = []
            self.document_types = []
            self.file_paths = []
            logger.info("Created fallback simple index")
    
    def add_document(self, text, document_type, file_path=None):
        """
        Add a document to the vector database.
        
        Args:
            text (str): Document text
            document_type (str): Type of document (invoice, receipt, etc.)
            file_path (str): Optional file path
        """
        try:
            if not text.strip():
                logger.warning("Empty text provided, skipping")
                return
            
            # Store metadata first
            self.documents.append(text)
            self.document_types.append(document_type)
            self.file_paths.append(file_path or "")
            
            if not self.use_tfidf and self.embedding_model:
                # Generate embedding and add to FAISS
                embedding = self.embedding_model.encode([text])
                self.index.add(embedding.astype('float32'))
            else:
                # For TF-IDF, we'll handle this in search
                pass
            
            logger.info(f"Added document of type '{document_type}' to index")
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    def search_similar_documents(self, query_text, k=5):
        """
        Search for similar documents.
        
        Args:
            query_text (str): Text to search for
            k (int): Number of results to return
            
        Returns:
            list: List of tuples (document_type, confidence, text)
        """
        try:
            if len(self.documents) == 0:
                logger.warning("No documents in index")
                return []
            
            if self.use_tfidf or not self.embedding_model:
                # Use simple keyword matching for fallback
                return self._search_with_keywords(query_text, k)
            
            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query_text])
            
            # Search in FAISS index
            distances, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(k, self.index.ntotal)
            )
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.document_types):
                    # Convert distance to similarity score (0-1)
                    # Lower distance = higher similarity
                    confidence = max(0, 1.0 - (distance / 10.0))  # Normalize distance
                    
                    results.append({
                        'document_type': self.document_types[idx],
                        'confidence': confidence,
                        'text': self.documents[idx][:200] + "..." if len(self.documents[idx]) > 200 else self.documents[idx],
                        'file_path': self.file_paths[idx],
                        'distance': float(distance)
                    })
            
            # Sort by confidence (highest first)
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _search_with_keywords(self, query_text, k=5):
        """Simple keyword-based search fallback."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            if len(self.documents) == 0:
                return []
            
            # Combine query with documents for TF-IDF
            all_texts = [query_text] + self.documents
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate similarities
            query_vec = tfidf_matrix[0:1]
            doc_vecs = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vec, doc_vecs).flatten()
            
            # Get top k results
            top_indices = similarities.argsort()[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    confidence = float(similarities[idx])
                    
                    results.append({
                        'document_type': self.document_types[idx],
                        'confidence': confidence,
                        'text': self.documents[idx][:200] + "..." if len(self.documents[idx]) > 200 else self.documents[idx],
                        'file_path': self.file_paths[idx],
                        'distance': 1.0 - confidence
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def classify_document(self, text):
        """
        Classify document type based on similarity search.
        
        Args:
            text (str): Document text to classify
            
        Returns:
            tuple: (document_type, confidence)
        """
        try:
            results = self.search_similar_documents(text, k=3)
            
            if not results:
                return "unknown", 0.0
            
            # Get the most confident result
            best_match = results[0]
            
            # If confidence is too low, return unknown
            if best_match['confidence'] < 0.3:
                return "unknown", best_match['confidence']
            
            return best_match['document_type'], best_match['confidence']
            
        except Exception as e:
            logger.error(f"Error classifying document: {e}")
            return "unknown", 0.0
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save metadata regardless of index type
            metadata = {
                'documents': self.documents,
                'document_types': self.document_types,
                'file_paths': self.file_paths,
                'model_name': self.model_name,
                'use_tfidf': getattr(self, 'use_tfidf', False)
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Save FAISS index only if using embeddings
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
                logger.info(f"Saved FAISS index with {len(self.documents)} documents")
            else:
                logger.info(f"Saved TF-IDF metadata with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def get_statistics(self):
        """Get statistics about the vector database."""
        try:
            if not self.document_types:
                return {}
            
            # Count documents by type
            type_counts = {}
            for doc_type in self.document_types:
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            return {
                'total_documents': len(self.documents),
                'document_types': type_counts,
                'index_size': self.index.ntotal if self.index else 0,
                'model_name': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def clear_index(self):
        """Clear all documents from the index."""
        try:
            self._create_new_index()
            logger.info("Index cleared")
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise
    
    def remove_document(self, index):
        """
        Remove a document from the index by its position.
        Note: FAISS doesn't support efficient removal, so this recreates the index.
        
        Args:
            index (int): Index of document to remove
        """
        try:
            if index < 0 or index >= len(self.documents):
                raise ValueError(f"Invalid index: {index}")
            
            # Remove from metadata
            self.documents.pop(index)
            self.document_types.pop(index)
            self.file_paths.pop(index)
            
            # Recreate index
            if self.documents:
                # Get embedding dimension
                sample_embedding = self.embedding_model.encode([self.documents[0]])
                dimension = sample_embedding.shape[1]
                
                # Create new index
                self.index = faiss.IndexFlatL2(dimension)
                
                # Re-add all documents
                for text in self.documents:
                    embedding = self.embedding_model.encode([text])
                    self.index.add(embedding.astype('float32'))
            else:
                self._create_new_index()
            
            logger.info(f"Removed document at index {index}")
            
        except Exception as e:
            logger.error(f"Error removing document: {e}")
            raise

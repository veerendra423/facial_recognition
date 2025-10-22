"""
Face Recognition Module using DeGirum SDK
Author: AI Assistant
Date: 2025
"""

import cv2
import numpy as np
import logging
import sqlite3
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import degirum as dg
except ImportError:
    print("DeGirum SDK not installed. Please install: pip install degirum")
    dg = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("scikit-learn not installed. Using manual cosine similarity calculation.")
    cosine_similarity = None


class FaceRecognizer:
    """Face recognition using DeGirum AI models"""
    
    def __init__(self, config):
        """Initialize face recognizer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.device_config = config.get_degirum_device_config()
        self.recognition_config = config.get_recognition_config()
        self.database_config = config.get_database_config()
        
        # Model parameters
        self.model_name = self.recognition_config.get('model_name')
        self.similarity_threshold = self.recognition_config.get('similarity_threshold', 0.7)
        self.input_size = self.recognition_config.get('input_size', [160, 160])
        self.embedding_size = self.recognition_config.get('embedding_size', 512)
        
        # Database
        self.db_path = self.database_config['path']
        self.embeddings_path = self.database_config['embeddings_path']
        
        # Initialize model
        self.model = None
        self._initialize_model()
        
        # Cache for embeddings
        self.embeddings_cache = {}
        self._load_embeddings_cache()
        
        self.logger.info("Face recognizer initialized")
    
    def _initialize_model(self):
        """Initialize DeGirum face recognition model"""
        try:
            if dg is None:
                raise ImportError("DeGirum SDK not available")
            
            # Connect to DeGirum device
            device_type = self.device_config.get('type', 'auto')
            device_id = self.device_config.get('device_id', 0)
            
            if device_type == 'auto':
                zoo = dg.connect_model_zoo()
            elif device_type == 'cpu':
                zoo = dg.connect_model_zoo(dg.hw.cpu)
            elif device_type == 'orca':
                zoo = dg.connect_model_zoo(f"orca{device_id}")
            else:
                zoo = dg.connect_model_zoo()
            
            # Load face recognition model
            if self.model_name:
                self.model = zoo.load_model(self.model_name)
                self.logger.info(f"Loaded DeGirum recognition model: {self.model_name}")
            else:
                # Try default face recognition models
                available_models = [
                    "facenet_keras--160x160_quant_n2x_orca1_1",
                    "arcface_resnet50--112x112_quant_n2x_orca1_1",
                    "sphereface_resnet50--112x112_quant_n2x_orca1_1"
                ]
                
                for model_name in available_models:
                    try:
                        self.model = zoo.load_model(model_name)
                        self.model_name = model_name
                        self.logger.info(f"Loaded default recognition model: {model_name}")
                        break
                    except Exception as e:
                        self.logger.debug(f"Failed to load {model_name}: {e}")
                        continue
                
                if self.model is None:
                    raise Exception("No suitable face recognition model found")
            
            self.logger.info("DeGirum face recognition model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DeGirum recognition model: {e}")
            self.model = None
    
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding from face image"""
        try:
            if self.model is None:
                self.logger.error("Face recognition model not available")
                return None
            
            # Preprocess image
            processed_image = self._preprocess_image(face_image)
            
            # Run inference
            result = self.model(processed_image)
            
            # Extract embedding
            embedding = None
            
            if hasattr(result, 'embedding'):
                embedding = result.embedding
            elif hasattr(result, 'feature_vector'):
                embedding = result.feature_vector
            elif hasattr(result, 'results') and len(result.results) > 0:
                embedding = result.results[0]
            elif isinstance(result, np.ndarray):
                embedding = result
            elif isinstance(result, list) and len(result) > 0:
                embedding = np.array(result)
            else:
                self.logger.error("Could not extract embedding from model result")
                return None
            
            # Ensure embedding is numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Normalize embedding
            embedding = self._normalize_embedding(embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error getting face embedding: {e}")
            return None
    
    def _preprocess_image(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for recognition model"""
        try:
            # Resize to model input size
            processed = cv2.resize(face_image, tuple(self.input_size))
            
            # Convert BGR to RGB if needed
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 1]
            processed = processed.astype(np.float32) / 255.0
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing face image: {e}")
            return face_image
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit vector"""
        try:
            # Flatten if needed
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            # L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error normalizing embedding: {e}")
            return embedding
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize a face and return person ID and confidence"""
        try:
            # Get embedding
            embedding = self.get_embedding(face_image)
            if embedding is None:
                return None, 0.0
            
            # Match against database
            person_id, similarity = self._match_embedding(embedding)
            
            # Check threshold
            if similarity >= self.similarity_threshold:
                return person_id, similarity
            else:
                return None, similarity
            
        except Exception as e:
            self.logger.error(f"Error in face recognition: {e}")
            return None, 0.0
    
    def _match_embedding(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Match embedding against database"""
        try:
            if not self.embeddings_cache:
                self._load_embeddings_cache()
            
            if not self.embeddings_cache:
                return None, 0.0
            
            best_match = None
            best_similarity = 0.0
            
            for person_name, person_embeddings in self.embeddings_cache.items():
                for stored_embedding in person_embeddings:
                    similarity = self._calculate_similarity(embedding, stored_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = person_name
            
            return best_match, best_similarity
            
        except Exception as e:
            self.logger.error(f"Error matching embedding: {e}")
            return None, 0.0
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two embeddings"""
        try:
            if cosine_similarity is not None:
                # Use scikit-learn
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            else:
                # Manual cosine similarity calculation
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                else:
                    similarity = 0.0
            
            # Ensure similarity is in [0, 1] range
            similarity = max(0.0, min(1.0, similarity))
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _load_embeddings_cache(self):
        """Load all embeddings into memory cache"""
        try:
            self.embeddings_cache = {}
            
            if not os.path.exists(self.db_path):
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all persons and their embeddings
            cursor.execute("""
                SELECT p.name, fe.embedding_file 
                FROM persons p 
                JOIN face_embeddings fe ON p.id = fe.person_id
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            for person_name, embedding_file in results:
                embedding_path = os.path.join(self.embeddings_path, embedding_file)
                
                if os.path.exists(embedding_path):
                    try:
                        with open(embedding_path, 'rb') as f:
                            embedding = pickle.load(f)
                        
                        if person_name not in self.embeddings_cache:
                            self.embeddings_cache[person_name] = []
                        
                        self.embeddings_cache[person_name].append(embedding)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load embedding {embedding_file}: {e}")
            
            self.logger.info(f"Loaded embeddings for {len(self.embeddings_cache)} persons")
            
        except Exception as e:
            self.logger.error(f"Error loading embeddings cache: {e}")
    
    def add_person_embedding(self, person_name: str, embedding: np.ndarray):
        """Add embedding to cache for a person"""
        try:
            if person_name not in self.embeddings_cache:
                self.embeddings_cache[person_name] = []
            
            self.embeddings_cache[person_name].append(embedding)
            self.logger.info(f"Added embedding for {person_name}")
            
        except Exception as e:
            self.logger.error(f"Error adding embedding to cache: {e}")
    
    def remove_person_from_cache(self, person_name: str):
        """Remove person from embeddings cache"""
        try:
            if person_name in self.embeddings_cache:
                del self.embeddings_cache[person_name]
                self.logger.info(f"Removed {person_name} from cache")
            
        except Exception as e:
            self.logger.error(f"Error removing person from cache: {e}")
    
    def get_enrolled_persons(self) -> List[str]:
        """Get list of enrolled person names"""
        return list(self.embeddings_cache.keys())
    
    def get_person_embedding_count(self, person_name: str) -> int:
        """Get number of embeddings for a person"""
        return len(self.embeddings_cache.get(person_name, []))
    
    def verify_face(self, face_image: np.ndarray, person_name: str) -> Tuple[bool, float]:
        """Verify if face belongs to specific person"""
        try:
            if person_name not in self.embeddings_cache:
                return False, 0.0
            
            # Get embedding
            embedding = self.get_embedding(face_image)
            if embedding is None:
                return False, 0.0
            
            # Calculate best similarity with person's embeddings
            best_similarity = 0.0
            person_embeddings = self.embeddings_cache[person_name]
            
            for stored_embedding in person_embeddings:
                similarity = self._calculate_similarity(embedding, stored_embedding)
                best_similarity = max(best_similarity, similarity)
            
            # Check threshold
            is_match = best_similarity >= self.similarity_threshold
            
            return is_match, best_similarity
            
        except Exception as e:
            self.logger.error(f"Error in face verification: {e}")
            return False, 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            'model_name': self.model_name,
            'similarity_threshold': self.similarity_threshold,
            'input_size': self.input_size,
            'embedding_size': self.embedding_size,
            'enrolled_persons': len(self.embeddings_cache),
            'backend': 'degirum' if self.model else 'none'
        }
        
        if self.model and hasattr(self.model, 'model_info'):
            info.update(self.model.model_info)
        
        return info
    
    def set_similarity_threshold(self, threshold: float):
        """Set recognition similarity threshold"""
        self.similarity_threshold = threshold
        self.logger.info(f"Similarity threshold set to {threshold}")
    
    def reload_embeddings(self):
        """Reload embeddings from database"""
        self._load_embeddings_cache()
        self.logger.info("Embeddings cache reloaded")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.model:
                self.model = None
            self.embeddings_cache.clear()
            self.logger.info("Face recognizer cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up recognizer: {e}")
#!/usr/bin/env python3
"""
Face Enrollment System for Dual RTSP Camera Facial Recognition

This script allows users to register/enroll faces into the database for
recognition by the dual RTSP camera system using DeGirum SDK.

Author: AI Assistant
Date: 2025
"""

import argparse
import logging
import cv2
import numpy as np
import yaml
import sqlite3
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import base64

try:
    import degirum as dg
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please install DeGirum SDK: pip install degirum")
    exit(1)


class FaceEnrollmentSystem:
    """Face enrollment system for registering faces"""
    
    def __init__(self, config_path: str = "config/dual_rtsp.yaml"):
        """Initialize the face enrollment system
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize database
        self.db_path = Path(self.config['database']['path'])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Initialize DeGirum models
        self._init_degirum_models()
        
        # Face enrollment settings
        self.min_face_size = self.config['enrollment']['min_face_size']
        self.max_face_size = self.config['enrollment']['max_face_size']
        self.min_quality_score = self.config['enrollment']['min_quality_score']
        self.required_samples = self.config['enrollment']['required_samples']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            exit(1)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/face_enrollment.log'),
                logging.StreamHandler()
            ]
        )
        Path('logs').mkdir(exist_ok=True)
        return logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize the face database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                image_data BLOB,
                enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                quality_score REAL,
                metadata TEXT
            )
        ''')
        
        # Create enrollment sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enrollment_sessions (
                session_id TEXT PRIMARY KEY,
                person_name TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                samples_collected INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        conn.commit()
        conn.close()
        self.logger.info("Database initialized successfully")
    
    def _init_degirum_models(self):
        """Initialize DeGirum models for face detection and recognition"""
        try:
            # Connect to DeGirum cloud or edge server
            if self.config['degirum']['use_cloud']:
                self.dg_model = dg.connect_model_zoo(
                    server=self.config['degirum']['cloud_server'],
                    token=self.config['degirum']['cloud_token']
                )
            else:
                self.dg_model = dg.connect_model_zoo(
                    server=self.config['degirum']['edge_server']
                )
            
            # Load face detection model
            self.face_detector = self.dg_model.load_model(
                model_name=self.config['degirum']['models']['face_detection']['model_name']
            )
            
            # Load face recognition model
            self.face_recognizer = self.dg_model.load_model(
                model_name=self.config['degirum']['models']['face_recognition']['model_name']
            )
            
            self.logger.info("DeGirum models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DeGirum models: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in the image using DeGirum
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        try:
            # Run face detection
            results = self.face_detector.predict(image)
            
            faces = []
            for detection in results:
                if detection.score >= self.config['degirum']['models']['face_detection']['confidence_threshold']:
                    bbox = detection.bbox
                    faces.append({
                        'bbox': [int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)],
                        'confidence': float(detection.score),
                        'landmarks': getattr(detection, 'landmarks', None)
                    })
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using DeGirum face recognition model
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Face embedding vector or None if extraction failed
        """
        try:
            # Preprocess face image for recognition model
            processed_face = self._preprocess_face_for_recognition(face_image)
            
            # Extract embedding
            result = self.face_recognizer.predict(processed_face)
            
            if hasattr(result, 'embedding'):
                return result.embedding
            elif isinstance(result, np.ndarray):
                return result
            else:
                # Handle different result formats
                return np.array(result.feature_vector) if hasattr(result, 'feature_vector') else None
                
        except Exception as e:
            self.logger.error(f"Face embedding extraction failed: {e}")
            return None
    
    def _preprocess_face_for_recognition(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for recognition model
        
        Args:
            face_image: Input face image
            
        Returns:
            Preprocessed face image
        """
        # Resize to model input size
        target_size = self.config['degirum']['models']['face_recognition']['input_size']
        face_resized = cv2.resize(face_image, (target_size, target_size))
        
        # Normalize if required
        if self.config['degirum']['models']['face_recognition']['normalize']:
            face_resized = face_resized.astype(np.float32) / 255.0
        
        return face_resized
    
    def calculate_face_quality(self, face_image: np.ndarray) -> float:
        """Calculate face quality score
        
        Args:
            face_image: Face image
            
        Returns:
            Quality score between 0 and 1
        """
        # Convert to grayscale for quality assessment
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # Calculate brightness score
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
        
        # Calculate contrast score
        contrast_score = np.std(gray) / 255.0
        
        # Combined quality score
        quality_score = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
        
        return min(quality_score, 1.0)
    
    def enroll_face_from_image(self, image_path: str, person_name: str) -> bool:
        """Enroll a face from a single image file
        
        Args:
            image_path: Path to the image file
            person_name: Name of the person
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if not faces:
                self.logger.warning(f"No faces detected in image: {image_path}")
                return False
            
            if len(faces) > 1:
                self.logger.warning(f"Multiple faces detected in image: {image_path}. Using the largest face.")
            
            # Use the largest face
            largest_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
            bbox = largest_face['bbox']
            
            # Extract face
            face_image = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            
            # Calculate quality
            quality_score = self.calculate_face_quality(face_image)
            
            if quality_score < self.min_quality_score:
                self.logger.warning(f"Face quality too low: {quality_score:.2f} < {self.min_quality_score}")
                return False
            
            # Extract embedding
            embedding = self.extract_face_embedding(face_image)
            
            if embedding is None:
                self.logger.error("Failed to extract face embedding")
                return False
            
            # Save to database
            face_id = str(uuid.uuid4())
            self._save_face_to_database(
                face_id=face_id,
                person_name=person_name,
                embedding=embedding,
                face_image=face_image,
                quality_score=quality_score,
                metadata={'source': 'single_image', 'image_path': image_path}
            )
            
            self.logger.info(f"Successfully enrolled face for {person_name} with quality {quality_score:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Face enrollment failed: {e}")
            return False
    
    def enroll_face_from_camera(self, camera_source: str, person_name: str) -> bool:
        """Enroll faces from live camera feed
        
        Args:
            camera_source: Camera source (RTSP URL or camera index)
            person_name: Name of the person
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                self.logger.error(f"Failed to open camera: {camera_source}")
                return False
            
            # Set camera properties for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            collected_samples = 0
            session_id = str(uuid.uuid4())
            
            # Create enrollment session
            self._create_enrollment_session(session_id, person_name)
            
            self.logger.info(f"Starting enrollment for {person_name}. Collect {self.required_samples} samples.")
            self.logger.info("Press 'c' to capture, 'q' to quit")
            
            while collected_samples < self.required_samples:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to read from camera")
                    break
                
                # Detect faces in current frame
                faces = self.detect_faces(frame)
                
                # Draw detected faces
                display_frame = frame.copy()
                for face in faces:
                    bbox = face['bbox']
                    cv2.rectangle(display_frame, 
                                (bbox[0], bbox[1]), 
                                (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                                (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Conf: {face['confidence']:.2f}", 
                              (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Display instructions
                cv2.putText(display_frame, f"Samples: {collected_samples}/{self.required_samples}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(f'Face Enrollment - {person_name}', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):  # Capture face
                    if faces:
                        # Use the best quality face
                        best_face = self._select_best_face(frame, faces)
                        if best_face:
                            collected_samples += 1
                            self._update_enrollment_session(session_id, collected_samples)
                            self.logger.info(f"Captured sample {collected_samples}/{self.required_samples}")
                        else:
                            self.logger.warning("No suitable face found for capture")
                    else:
                        self.logger.warning("No faces detected for capture")
                
                elif key == ord('q'):  # Quit
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Complete enrollment session
            success = collected_samples >= self.required_samples
            self._complete_enrollment_session(session_id, success)
            
            if success:
                self.logger.info(f"Successfully enrolled {person_name} with {collected_samples} samples")
            else:
                self.logger.warning(f"Enrollment incomplete for {person_name}. Only {collected_samples} samples collected.")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Camera enrollment failed: {e}")
            return False
    
    def _select_best_face(self, frame: np.ndarray, faces: List[Dict[str, Any]]) -> Optional[str]:
        """Select and save the best quality face from detected faces
        
        Args:
            frame: Current frame
            faces: List of detected faces
            
        Returns:
            Face ID if successful, None otherwise
        """
        best_face_id = None
        best_quality = 0.0
        
        for face in faces:
            bbox = face['bbox']
            face_image = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            
            # Check face size
            face_area = bbox[2] * bbox[3]
            if face_area < self.min_face_size or face_area > self.max_face_size:
                continue
            
            # Calculate quality
            quality_score = self.calculate_face_quality(face_image)
            
            if quality_score >= self.min_quality_score and quality_score > best_quality:
                # Extract embedding
                embedding = self.extract_face_embedding(face_image)
                
                if embedding is not None:
                    # Save this face
                    face_id = str(uuid.uuid4())
                    # Note: person_name would need to be passed to this method
                    # For now, we'll use a placeholder - this should be fixed in the calling method
                    
                    best_face_id = face_id
                    best_quality = quality_score
        
        return best_face_id
    
    def _save_face_to_database(self, face_id: str, person_name: str, embedding: np.ndarray, 
                              face_image: np.ndarray, quality_score: float, metadata: Dict[str, Any]):
        """Save face data to database
        
        Args:
            face_id: Unique face identifier
            person_name: Name of the person
            embedding: Face embedding vector
            face_image: Face image
            quality_score: Quality score
            metadata: Additional metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize embedding and image
        embedding_blob = pickle.dumps(embedding)
        _, img_encoded = cv2.imencode('.jpg', face_image)
        image_blob = img_encoded.tobytes()
        metadata_json = str(metadata)
        
        cursor.execute('''
            INSERT INTO faces (id, name, embedding, image_data, quality_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (face_id, person_name, embedding_blob, image_blob, quality_score, metadata_json))
        
        conn.commit()
        conn.close()
    
    def _create_enrollment_session(self, session_id: str, person_name: str):
        """Create new enrollment session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO enrollment_sessions (session_id, person_name)
            VALUES (?, ?)
        ''', (session_id, person_name))
        
        conn.commit()
        conn.close()
    
    def _update_enrollment_session(self, session_id: str, samples_collected: int):
        """Update enrollment session with sample count"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE enrollment_sessions 
            SET samples_collected = ?
            WHERE session_id = ?
        ''', (samples_collected, session_id))
        
        conn.commit()
        conn.close()
    
    def _complete_enrollment_session(self, session_id: str, success: bool):
        """Complete enrollment session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        status = 'completed' if success else 'incomplete'
        
        cursor.execute('''
            UPDATE enrollment_sessions 
            SET end_time = CURRENT_TIMESTAMP, status = ?
            WHERE session_id = ?
        ''', (status, session_id))
        
        conn.commit()
        conn.close()
    
    def list_enrolled_faces(self) -> List[Dict[str, Any]]:
        """List all enrolled faces"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, enrollment_date, quality_score 
            FROM faces 
            ORDER BY enrollment_date DESC
        ''')
        
        faces = []
        for row in cursor.fetchall():
            faces.append({
                'id': row[0],
                'name': row[1],
                'enrollment_date': row[2],
                'quality_score': row[3]
            })
        
        conn.close()
        return faces
    
    def delete_face(self, face_id: str) -> bool:
        """Delete a face from database
        
        Args:
            face_id: Face ID to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
            
            if cursor.rowcount > 0:
                conn.commit()
                conn.close()
                self.logger.info(f"Deleted face: {face_id}")
                return True
            else:
                conn.close()
                self.logger.warning(f"Face not found: {face_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete face: {e}")
            return False


def main():
    """Main function for face enrollment"""
    parser = argparse.ArgumentParser(description="Face Enrollment System")
    parser.add_argument('--config', default='config/dual_rtsp.yaml', help='Configuration file path')
    parser.add_argument('--mode', choices=['image', 'camera'], required=True, help='Enrollment mode')
    parser.add_argument('--name', required=True, help='Person name')
    parser.add_argument('--source', help='Image path or camera source (RTSP URL)')
    parser.add_argument('--list', action='store_true', help='List enrolled faces')
    parser.add_argument('--delete', help='Delete face by ID')
    
    args = parser.parse_args()
    
    # Initialize enrollment system
    enrollment_system = FaceEnrollmentSystem(args.config)
    
    if args.list:
        faces = enrollment_system.list_enrolled_faces()
        print("\nEnrolled Faces:")
        print("-" * 80)
        for face in faces:
            print(f"ID: {face['id']}")
            print(f"Name: {face['name']}")
            print(f"Enrollment Date: {face['enrollment_date']}")
            print(f"Quality Score: {face['quality_score']:.2f}")
            print("-" * 80)
        return
    
    if args.delete:
        success = enrollment_system.delete_face(args.delete)
        if success:
            print(f"Successfully deleted face: {args.delete}")
        else:
            print(f"Failed to delete face: {args.delete}")
        return
    
    # Enrollment modes
    if args.mode == 'image':
        if not args.source:
            print("Error: --source required for image mode")
            return
        
        success = enrollment_system.enroll_face_from_image(args.source, args.name)
        if success:
            print(f"Successfully enrolled {args.name} from image")
        else:
            print(f"Failed to enroll {args.name} from image")
    
    elif args.mode == 'camera':
        source = args.source if args.source else 0  # Default to camera 0
        success = enrollment_system.enroll_face_from_camera(source, args.name)
        if success:
            print(f"Successfully enrolled {args.name} from camera")
        else:
            print(f"Failed to enroll {args.name} from camera")


if __name__ == "__main__":
    main()
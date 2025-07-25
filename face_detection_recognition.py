#!/usr/bin/env python3
"""
Face Detection and Recognition System for Dual RTSP Cameras

This script performs real-time face detection and recognition using two RTSP cameras
with DeGirum AI SDK for high accuracy on Raspberry Pi 5.

Author: AI Assistant
Date: 2025
"""

import argparse
import logging
import time
import threading
import queue
import cv2
import numpy as np
import yaml
import sqlite3
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

try:
    import degirum as dg
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please install DeGirum SDK: pip install degirum")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/face_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FaceRecognitionDatabase:
    """Database interface for face recognition"""
    
    def __init__(self, db_path: str = "data/faces.db"):
        """Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.faces_cache = {}
        self.last_cache_update = 0
        self.cache_ttl = 60  # Cache time-to-live in seconds
        
        # Load faces into memory for fast recognition
        self._update_faces_cache()
    
    def _update_faces_cache(self):
        """Update the in-memory faces cache"""
        try:
            if not self.db_path.exists():
                logger.warning(f"Database file not found: {self.db_path}")
                return
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT person_id, name, embedding, confidence_threshold, metadata
                    FROM faces
                ''')
                
                self.faces_cache = {}
                for row in cursor.fetchall():
                    person_id, name, embedding_blob, confidence_threshold, metadata = row
                    embedding = pickle.loads(embedding_blob)
                    
                    self.faces_cache[person_id] = {
                        'name': name,
                        'embedding': embedding,
                        'confidence_threshold': confidence_threshold,
                        'metadata': metadata
                    }
                
                self.last_cache_update = time.time()
                logger.info(f"Loaded {len(self.faces_cache)} faces into cache")
                
        except Exception as e:
            logger.error(f"Error updating faces cache: {e}")
    
    def get_all_faces(self) -> Dict[str, Dict]:
        """Get all faces from cache
        
        Returns:
            Dictionary of person_id -> face_data
        """
        # Update cache if it's stale
        if time.time() - self.last_cache_update > self.cache_ttl:
            self._update_faces_cache()
        
        return self.faces_cache.copy()
    
    def update_last_seen(self, person_id: str):
        """Update last seen timestamp for a person
        
        Args:
            person_id: Person's unique identifier
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE faces SET last_seen = CURRENT_TIMESTAMP 
                    WHERE person_id = ?
                ''', (person_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating last seen: {e}")
    
    def log_recognition_event(self, person_id: str, camera_id: str, 
                            confidence: float, bbox: List[int]):
        """Log a recognition event
        
        Args:
            person_id: Person's unique identifier
            camera_id: Camera that detected the person
            confidence: Recognition confidence
            bbox: Bounding box coordinates
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create recognition_events table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recognition_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        bbox TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (person_id) REFERENCES faces (person_id)
                    )
                ''')
                
                # Insert recognition event
                bbox_str = json.dumps(bbox)
                cursor.execute('''
                    INSERT INTO recognition_events 
                    (person_id, camera_id, confidence, bbox)
                    VALUES (?, ?, ?, ?)
                ''', (person_id, camera_id, confidence, bbox_str))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging recognition event: {e}")


class RTSPCameraStream:
    """RTSP camera stream handler"""
    
    def __init__(self, camera_id: str, camera_config: Dict):
        """Initialize RTSP camera stream
        
        Args:
            camera_id: Camera identifier
            camera_config: Camera configuration
        """
        self.camera_id = camera_id
        self.config = camera_config
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=3)
        self.is_running = False
        self.thread = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.connection_attempts = 0
        
    def connect(self) -> bool:
        """Connect to RTSP stream
        
        Returns:
            True if successful, False otherwise
        """
        try:
            rtsp_url = self.config['rtsp_url']
            logger.info(f"Connecting to {self.camera_id}: {rtsp_url}")
            
            # Create video capture with optimized settings
            self.capture = cv2.VideoCapture(rtsp_url)
            
            # Configure capture settings
            connection_config = self.config.get('connection', {})
            if connection_config.get('tcp_transport', True):
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            
            stream_config = self.config['stream']
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, stream_config['width'])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, stream_config['height'])
            self.capture.set(cv2.CAP_PROP_FPS, stream_config['fps'])
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, connection_config.get('buffer_size', 3))
            
            # Test connection
            ret, frame = self.capture.read()
            if not ret or frame is None:
                logger.error(f"Failed to read from {self.camera_id}")
                return False
            
            logger.info(f"Successfully connected to {self.camera_id}")
            self.connection_attempts = 0
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {self.camera_id}: {e}")
            return False
    
    def start(self):
        """Start the camera stream thread"""
        if self.capture is None:
            if not self.connect():
                return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started stream thread for {self.camera_id}")
        return True
    
    def stop(self):
        """Stop the camera stream"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.capture:
            self.capture.release()
        self.capture = None
        logger.info(f"Stopped stream for {self.camera_id}")
    
    def _stream_loop(self):
        """Main streaming loop"""
        reconnect_interval = self.config.get('connection', {}).get('reconnect_interval', 5)
        max_reconnect_attempts = self.config.get('connection', {}).get('max_reconnect_attempts', 10)
        
        while self.is_running:
            try:
                ret, frame = self.capture.read()
                
                if not ret or frame is None:
                    logger.warning(f"Failed to read frame from {self.camera_id}")
                    
                    # Attempt reconnection
                    if self.connection_attempts < max_reconnect_attempts:
                        self.connection_attempts += 1
                        logger.info(f"Attempting to reconnect {self.camera_id} (attempt {self.connection_attempts})")
                        
                        self.capture.release()
                        time.sleep(reconnect_interval)
                        
                        if self.connect():
                            continue
                    else:
                        logger.error(f"Max reconnection attempts reached for {self.camera_id}")
                        break
                    
                    continue
                
                # Reset connection attempts on successful read
                self.connection_attempts = 0
                
                # Scale frame if configured
                stream_config = self.config['stream']
                if ('scale_to_width' in stream_config and 
                    'scale_to_height' in stream_config):
                    target_width = stream_config['scale_to_width']
                    target_height = stream_config['scale_to_height']
                    
                    if stream_config.get('maintain_aspect_ratio', True):
                        # Calculate aspect ratio preserving dimensions
                        h, w = frame.shape[:2]
                        aspect = w / h
                        
                        if aspect > target_width / target_height:
                            new_width = target_width
                            new_height = int(target_width / aspect)
                        else:
                            new_height = target_height
                            new_width = int(target_height * aspect)
                        
                        frame = cv2.resize(frame, (new_width, new_height))
                    else:
                        frame = cv2.resize(frame, (target_width, target_height))
                
                # Apply quality enhancements if configured
                quality_config = self.config.get('quality', {})
                if quality_config.get('brightness_adjustment') != 0:
                    frame = cv2.convertScaleAbs(frame, alpha=1, 
                                              beta=quality_config['brightness_adjustment'])
                
                if quality_config.get('contrast_adjustment') != 1.0:
                    frame = cv2.convertScaleAbs(frame, 
                                              alpha=quality_config['contrast_adjustment'], beta=0)
                
                if quality_config.get('apply_clahe', False):
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    l = clahe.apply(l)
                    frame = cv2.merge([l, a, b])
                    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
                
                # Add frame to queue (drop old frames if queue is full)
                if not self.frame_queue.full():
                    self.frame_queue.put({
                        'frame': frame,
                        'timestamp': time.time(),
                        'frame_number': self.frame_count
                    })
                else:
                    # Remove old frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put({
                        'frame': frame,
                        'timestamp': time.time(),
                        'frame_number': self.frame_count
                    })
                
                self.frame_count += 1
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in stream loop for {self.camera_id}: {e}")
                time.sleep(1)
    
    def get_latest_frame(self) -> Optional[Dict]:
        """Get the latest frame from the queue
        
        Returns:
            Frame data dictionary or None if no frame available
        """
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None


class FaceRecognizer:
    """Face recognition using DeGirum models"""
    
    def __init__(self, config: Dict):
        """Initialize face recognizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.detection_model = None
        self.recognition_model = None
        self.database = FaceRecognitionDatabase()
        
        # Initialize DeGirum models
        self._init_models()
    
    def _init_models(self):
        """Initialize DeGirum models"""
        try:
            # Initialize DeGirum connection
            if self.config.get('degirum', {}).get('cloud_token'):
                zoo = dg.zoo.connect(cloud_token=self.config['degirum']['cloud_token'])
            else:
                zoo = dg.zoo
            
            # Load face detection model
            detection_config = self.config['degirum']['models']['face_detection']
            self.detection_model = zoo.load_model(
                model_name=detection_config['model_name'],
                inference_width=detection_config['inference_width'],
                inference_height=detection_config['inference_height']
            )
            
            # Load face recognition model
            recognition_config = self.config['degirum']['models']['face_recognition']
            self.recognition_model = zoo.load_model(
                model_name=recognition_config['model_name'],
                inference_width=recognition_config['inference_width'],
                inference_height=recognition_config['inference_height']
            )
            
            logger.info("Successfully initialized DeGirum models")
            
        except Exception as e:
            logger.error(f"Error initializing DeGirum models: {e}")
            raise
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected faces
        """
        try:
            results = self.detection_model.predict(frame)
            
            faces = []
            confidence_threshold = self.config['degirum']['models']['face_detection']['confidence_threshold']
            
            for detection in results:
                if hasattr(detection, 'bbox') and hasattr(detection, 'confidence'):
                    x1, y1, x2, y2 = detection.bbox
                    confidence = detection.confidence
                    
                    if confidence >= confidence_threshold:
                        faces.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'landmarks': getattr(detection, 'landmarks', None)
                        })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Face embedding or None
        """
        try:
            # Preprocess face image
            face_resized = cv2.resize(face_image, (112, 112))
            
            # Run recognition model
            results = self.recognition_model.predict(face_resized)
            
            # Extract embedding
            if hasattr(results, 'embedding'):
                return np.array(results.embedding)
            elif len(results) > 0 and hasattr(results[0], 'embedding'):
                return np.array(results[0].embedding)
            elif isinstance(results, np.ndarray):
                return results.flatten()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize face using database
        
        Args:
            face_embedding: Face embedding to match
            
        Returns:
            Tuple of (person_id, confidence) or (None, 0.0)
        """
        try:
            known_faces = self.database.get_all_faces()
            
            if not known_faces:
                return None, 0.0
            
            best_match_id = None
            best_confidence = 0.0
            
            for person_id, face_data in known_faces.items():
                known_embedding = face_data['embedding']
                
                # Calculate cosine similarity
                dot_product = np.dot(face_embedding, known_embedding)
                norm_a = np.linalg.norm(face_embedding)
                norm_b = np.linalg.norm(known_embedding)
                
                if norm_a == 0 or norm_b == 0:
                    continue
                
                similarity = dot_product / (norm_a * norm_b)
                
                # Convert similarity to confidence (0-1 range)
                confidence = (similarity + 1) / 2
                
                # Check if this is the best match and above threshold
                threshold = face_data.get('confidence_threshold', 0.7)
                if confidence > threshold and confidence > best_confidence:
                    best_confidence = confidence
                    best_match_id = person_id
            
            return best_match_id, best_confidence
            
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return None, 0.0
    
    def process_frame(self, frame: np.ndarray, camera_id: str) -> Tuple[np.ndarray, List[Dict]]:
        """Process frame for face recognition
        
        Args:
            frame: Input frame
            camera_id: Camera identifier
            
        Returns:
            Tuple of (annotated_frame, recognition_results)
        """
        # Detect faces
        faces = self.detect_faces(frame)
        
        annotated_frame = frame.copy()
        recognition_results = []
        
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            detection_confidence = face['confidence']
            
            # Extract face region
            face_image = frame[y1:y2, x1:x2]
            if face_image.size == 0:
                continue
            
            # Extract embedding
            embedding = self.extract_embedding(face_image)
            if embedding is None:
                continue
            
            # Recognize face
            person_id, recognition_confidence = self.recognize_face(embedding)
            
            # Prepare result
            result = {
                'bbox': [x1, y1, x2, y2],
                'detection_confidence': detection_confidence,
                'recognition_confidence': recognition_confidence,
                'person_id': person_id,
                'person_name': None,
                'camera_id': camera_id,
                'timestamp': time.time()
            }
            
            # Draw bounding box and label
            if person_id:
                # Known person
                face_data = self.database.get_all_faces().get(person_id, {})
                person_name = face_data.get('name', 'Unknown')
                result['person_name'] = person_name
                
                # Green box for recognized faces
                color = (0, 255, 0)
                label = f"{person_name} ({recognition_confidence:.2f})"
                
                # Log recognition event
                self.database.log_recognition_event(person_id, camera_id, 
                                                  recognition_confidence, [x1, y1, x2, y2])
                self.database.update_last_seen(person_id)
                
            else:
                # Unknown person
                color = (0, 0, 255)
                label = f"Unknown ({detection_confidence:.2f})"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1-25), (x1+label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            recognition_results.append(result)
        
        return annotated_frame, recognition_results


class DualCameraRecognitionSystem:
    """Main dual camera recognition system"""
    
    def __init__(self, config_path: str):
        """Initialize the recognition system
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.recognizer = FaceRecognizer(self.config)
        self.camera_streams = {}
        self.is_running = False
        
        # Create directories
        Path("logs").mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(parents=True, exist_ok=True)
        
        # Initialize camera streams
        self._init_camera_streams()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _init_camera_streams(self):
        """Initialize camera streams"""
        for camera_id, camera_config in self.config.get('cameras', {}).items():
            if camera_config.get('enabled', True):
                self.camera_streams[camera_id] = RTSPCameraStream(camera_id, camera_config)
                logger.info(f"Initialized stream for {camera_id}")
    
    def start(self):
        """Start the recognition system"""
        logger.info("Starting dual camera recognition system")
        
        # Start camera streams
        for camera_id, stream in self.camera_streams.items():
            if stream.start():
                logger.info(f"Started camera stream: {camera_id}")
            else:
                logger.error(f"Failed to start camera stream: {camera_id}")
        
        self.is_running = True
        
        # Wait for streams to stabilize
        time.sleep(2)
    
    def stop(self):
        """Stop the recognition system"""
        logger.info("Stopping dual camera recognition system")
        self.is_running = False
        
        # Stop camera streams
        for stream in self.camera_streams.values():
            stream.stop()
    
    def run_recognition(self, display: bool = True, save_results: bool = False):
        """Run the main recognition loop
        
        Args:
            display: Whether to display video output
            save_results: Whether to save recognition results
        """
        try:
            recognition_log = []
            
            while self.is_running:
                all_results = []
                display_frames = {}
                
                # Process each camera
                for camera_id, stream in self.camera_streams.items():
                    frame_data = stream.get_latest_frame()
                    
                    if frame_data is None:
                        continue
                    
                    frame = frame_data['frame']
                    
                    # Process frame for recognition
                    annotated_frame, results = self.recognizer.process_frame(frame, camera_id)
                    
                    # Store results
                    all_results.extend(results)
                    display_frames[camera_id] = annotated_frame
                    
                    # Add camera info to frame
                    camera_name = self.config['cameras'][camera_id]['name']
                    cv2.putText(annotated_frame, camera_name, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Add timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(annotated_frame, timestamp, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Display frames if enabled
                if display and display_frames:
                    for camera_id, frame in display_frames.items():
                        window_name = f"Face Recognition - {self.config['cameras'][camera_id]['name']}"
                        cv2.imshow(window_name, frame)
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        # Force refresh database cache
                        self.recognizer.database._update_faces_cache()
                        logger.info("Refreshed face database cache")
                
                # Save results if enabled
                if save_results and all_results:
                    recognition_log.extend(all_results)
                
                # Print recognition summary
                if all_results:
                    known_persons = set()
                    unknown_count = 0
                    
                    for result in all_results:
                        if result['person_id']:
                            known_persons.add(result['person_name'])
                        else:
                            unknown_count += 1
                    
                    if known_persons or unknown_count > 0:
                        summary = f"Detected: {len(known_persons)} known"
                        if unknown_count > 0:
                            summary += f", {unknown_count} unknown"
                        logger.info(summary)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
            
            # Save recognition log if enabled
            if save_results and recognition_log:
                log_file = f"data/recognition_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(log_file, 'w') as f:
                    json.dump(recognition_log, f, indent=2, default=str)
                logger.info(f"Saved recognition log to {log_file}")
                
        except KeyboardInterrupt:
            logger.info("Recognition interrupted by user")
        except Exception as e:
            logger.error(f"Error in recognition loop: {e}")
        finally:
            if display:
                cv2.destroyAllWindows()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Dual Camera Face Recognition System")
    parser.add_argument("--config", default="config/dual_rtsp.yaml",
                       help="Configuration file path")
    parser.add_argument("--no-display", action="store_true",
                       help="Run without video display")
    parser.add_argument("--save-results", action="store_true",
                       help="Save recognition results to file")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create recognition system
    system = DualCameraRecognitionSystem(args.config)
    
    try:
        # Start system
        system.start()
        
        # Run recognition
        system.run_recognition(
            display=not args.no_display,
            save_results=args.save_results
        )
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        # Stop system
        system.stop()


if __name__ == "__main__":
    main()
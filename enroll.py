#!/usr/bin/env python3
"""
Face Enrollment Script for Dual RTSP Camera Facial Recognition System
Author: AI Assistant
Date: 2025
"""

import argparse
import cv2
import numpy as np
import os
import sys
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

from config import Config
from modules.detection import FaceDetector
from modules.recognition import FaceRecognizer
from modules.logging import setup_logger


class FaceEnrollment:
    """Face enrollment system for registering new faces"""
    
    def __init__(self, config_file="config.yaml"):
        """Initialize the enrollment system"""
        self.config = Config(config_file)
        self.logger = setup_logger("enrollment", self.config.logging)
        
        # Initialize components
        self.detector = FaceDetector(self.config)
        self.recognizer = FaceRecognizer(self.config)
        
        # Database setup
        self.db_path = self.config.database['path']
        self.embeddings_path = self.config.database['embeddings_path']
        
        # Create directories
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.embeddings_path, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info("Face enrollment system initialized")
    
    def _initialize_database(self):
        """Initialize the face database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    face_count INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    embedding_file TEXT NOT NULL,
                    quality_score REAL,
                    source_image TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def enroll_from_image(self, person_name: str, image_path: str) -> bool:
        """Enroll a person from a single image"""
        try:
            # Load image
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return False
            
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return False
            
            self.logger.info(f"Enrolling {person_name} from image: {image_path}")
            
            # Detect faces in image
            detections = self.detector.detect_faces(image)
            
            if not detections:
                self.logger.error("No faces detected in the image")
                return False
            
            if len(detections) > 1:
                self.logger.warning(f"Multiple faces detected ({len(detections)}), using the largest one")
                # Select the largest face
                detections = [max(detections, key=lambda d: d['bbox']['width'] * d['bbox']['height'])]
            
            # Extract and process face
            detection = detections[0]
            face_roi = self._extract_face_roi(image, detection)
            
            if face_roi is None:
                self.logger.error("Failed to extract face from image")
                return False
            
            # Calculate quality score
            quality_score = self._calculate_quality(face_roi)
            
            # Get face embedding
            embedding = self.recognizer.get_embedding(face_roi)
            if embedding is None:
                self.logger.error("Failed to generate face embedding")
                return False
            
            # Save to database
            success = self._save_face_to_database(person_name, embedding, quality_score, image_path)
            
            if success:
                self.logger.info(f"Successfully enrolled {person_name}")
                return True
            else:
                self.logger.error(f"Failed to save {person_name} to database")
                return False
                
        except Exception as e:
            self.logger.error(f"Error enrolling from image: {e}")
            return False
    
    def enroll_from_camera(self, person_name: str, camera_source: str = "0") -> bool:
        """Enroll a person using camera capture"""
        try:
            # Initialize camera
            if camera_source.isdigit():
                cap = cv2.VideoCapture(int(camera_source))
            else:
                cap = cv2.VideoCapture(camera_source)  # RTSP URL
            
            if not cap.isOpened():
                self.logger.error(f"Failed to open camera: {camera_source}")
                return False
            
            self.logger.info(f"Starting camera enrollment for {person_name}")
            print("Instructions:")
            print("- Look directly at the camera")
            print("- Press SPACE to capture a face")
            print("- Press 'q' to quit")
            
            captured_faces = []
            max_faces = self.config.database.get('max_faces_per_person', 5)
            
            while len(captured_faces) < max_faces:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    break
                
                # Detect faces
                detections = self.detector.detect_faces(frame)
                
                # Draw detections
                display_frame = frame.copy()
                best_detection = None
                
                for detection in detections:
                    bbox = detection['bbox']
                    h, w = frame.shape[:2]
                    
                    x1 = int(bbox['x'] * w)
                    y1 = int(bbox['y'] * h)
                    x2 = int((bbox['x'] + bbox['width']) * w)
                    y2 = int((bbox['y'] + bbox['height']) * h)
                    
                    # Choose the largest face as the best
                    if best_detection is None or (bbox['width'] * bbox['height'] > 
                                                 best_detection['bbox']['width'] * best_detection['bbox']['height']):
                        best_detection = detection
                        color = (0, 255, 0)  # Green for best face
                    else:
                        color = (0, 255, 255)  # Yellow for other faces
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"Conf: {detection['confidence']:.2f}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Add instructions
                cv2.putText(display_frame, f"Enrolling: {person_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Captured: {len(captured_faces)}/{max_faces}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "SPACE: Capture | Q: Quit", (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Face Enrollment", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and best_detection:  # Space to capture
                    face_roi = self._extract_face_roi(frame, best_detection)
                    if face_roi is not None:
                        quality_score = self._calculate_quality(face_roi)
                        if quality_score > 0.5:  # Minimum quality threshold
                            captured_faces.append((face_roi, quality_score))
                            print(f"Captured face {len(captured_faces)}/{max_faces} (Quality: {quality_score:.2f})")
                        else:
                            print(f"Face quality too low: {quality_score:.2f}")
                elif key == ord('q'):  # Quit
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            if not captured_faces:
                self.logger.error("No faces captured")
                return False
            
            # Process captured faces
            success_count = 0
            for i, (face_roi, quality_score) in enumerate(captured_faces):
                embedding = self.recognizer.get_embedding(face_roi)
                if embedding is not None:
                    # Save face image
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    face_filename = f"{person_name}_{timestamp}_{i}.jpg"
                    face_path = os.path.join(self.embeddings_path, face_filename)
                    cv2.imwrite(face_path, face_roi)
                    
                    # Save to database
                    if self._save_face_to_database(person_name, embedding, quality_score, face_path):
                        success_count += 1
            
            self.logger.info(f"Successfully enrolled {success_count}/{len(captured_faces)} faces for {person_name}")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error enrolling from camera: {e}")
            return False
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
    
    def _extract_face_roi(self, image: np.ndarray, detection: dict) -> Optional[np.ndarray]:
        """Extract face region of interest"""
        try:
            bbox = detection['bbox']
            h, w = image.shape[:2]
            
            # Convert normalized coordinates to pixels
            x1 = int(bbox['x'] * w)
            y1 = int(bbox['y'] * h)
            x2 = int((bbox['x'] + bbox['width']) * w)
            y2 = int((bbox['y'] + bbox['height']) * h)
            
            # Add padding
            padding = self.config.processing.get('face_padding', 0.2)
            pad_x = int((x2 - x1) * padding)
            pad_y = int((y2 - y1) * padding)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            face_roi = image[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
            
            # Resize to standard size
            input_size = self.config.degirum['face_recognition']['input_size']
            face_roi = cv2.resize(face_roi, tuple(input_size))
            
            return face_roi
            
        except Exception as e:
            self.logger.error(f"Error extracting face ROI: {e}")
            return None
    
    def _calculate_quality(self, face_image: np.ndarray) -> float:
        """Calculate face image quality score"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = gray.std()
            
            # Normalize scores
            blur_score = min(laplacian_var / 100.0, 1.0)  # Normalize to 0-1
            brightness_score = 1.0 - abs(brightness - 128) / 128.0  # Optimal around 128
            contrast_score = min(contrast / 64.0, 1.0)  # Normalize to 0-1
            
            # Combined quality score
            quality = (blur_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
            
            return quality
            
        except Exception as e:
            self.logger.error(f"Error calculating quality: {e}")
            return 0.0
    
    def _save_face_to_database(self, person_name: str, embedding: np.ndarray, 
                              quality_score: float, source_image: str) -> bool:
        """Save face embedding to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get or create person
            cursor.execute("SELECT id FROM persons WHERE name = ?", (person_name,))
            result = cursor.fetchone()
            
            if result:
                person_id = result[0]
                # Update person record
                cursor.execute("""
                    UPDATE persons 
                    SET updated_at = CURRENT_TIMESTAMP, 
                        face_count = face_count + 1 
                    WHERE id = ?
                """, (person_id,))
            else:
                # Create new person
                cursor.execute("""
                    INSERT INTO persons (name, face_count) 
                    VALUES (?, 1)
                """, (person_name,))
                person_id = cursor.lastrowid
            
            # Save embedding to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            embedding_filename = f"embedding_{person_id}_{timestamp}.pkl"
            embedding_path = os.path.join(self.embeddings_path, embedding_filename)
            
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding, f)
            
            # Save embedding record
            cursor.execute("""
                INSERT INTO face_embeddings (person_id, embedding_file, quality_score, source_image)
                VALUES (?, ?, ?, ?)
            """, (person_id, embedding_filename, quality_score, source_image))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")
            return False
    
    def list_enrolled_persons(self) -> List[Tuple[str, int, str]]:
        """List all enrolled persons"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, face_count, created_at 
                FROM persons 
                ORDER BY name
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error listing persons: {e}")
            return []
    
    def remove_person(self, person_name: str) -> bool:
        """Remove a person from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get person ID
            cursor.execute("SELECT id FROM persons WHERE name = ?", (person_name,))
            result = cursor.fetchone()
            
            if not result:
                self.logger.error(f"Person not found: {person_name}")
                return False
            
            person_id = result[0]
            
            # Get embedding files to delete
            cursor.execute("SELECT embedding_file FROM face_embeddings WHERE person_id = ?", (person_id,))
            embedding_files = cursor.fetchall()
            
            # Delete embedding files
            for (embedding_file,) in embedding_files:
                embedding_path = os.path.join(self.embeddings_path, embedding_file)
                if os.path.exists(embedding_path):
                    os.remove(embedding_path)
            
            # Delete from database
            cursor.execute("DELETE FROM face_embeddings WHERE person_id = ?", (person_id,))
            cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Successfully removed {person_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing person: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Face Enrollment System")
    parser.add_argument('--name', type=str, required=True, help='Person name to enroll')
    parser.add_argument('--image', type=str, help='Path to image file for enrollment')
    parser.add_argument('--camera', type=str, help='Camera source (0 for webcam, RTSP URL for IP camera)')
    parser.add_argument('--list', action='store_true', help='List all enrolled persons')
    parser.add_argument('--remove', type=str, help='Remove person from database')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    try:
        enrollment = FaceEnrollment(args.config)
        
        if args.list:
            # List enrolled persons
            persons = enrollment.list_enrolled_persons()
            if persons:
                print("\nEnrolled Persons:")
                print("-" * 50)
                for name, face_count, created_at in persons:
                    print(f"Name: {name}")
                    print(f"Face Count: {face_count}")
                    print(f"Created: {created_at}")
                    print("-" * 50)
            else:
                print("No persons enrolled yet.")
        
        elif args.remove:
            # Remove person
            if enrollment.remove_person(args.remove):
                print(f"Successfully removed {args.remove}")
            else:
                print(f"Failed to remove {args.remove}")
        
        elif args.image:
            # Enroll from image
            if enrollment.enroll_from_image(args.name, args.image):
                print(f"Successfully enrolled {args.name} from image")
            else:
                print(f"Failed to enroll {args.name}")
        
        elif args.camera:
            # Enroll from camera
            if enrollment.enroll_from_camera(args.name, args.camera):
                print(f"Successfully enrolled {args.name} from camera")
            else:
                print(f"Failed to enroll {args.name}")
        
        else:
            # Default: enroll from webcam
            if enrollment.enroll_from_camera(args.name, "0"):
                print(f"Successfully enrolled {args.name}")
            else:
                print(f"Failed to enroll {args.name}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
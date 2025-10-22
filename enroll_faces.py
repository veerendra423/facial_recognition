#!/usr/bin/env python3
"""
Face Enrollment System for Dual RTSP Camera Facial Recognition

This script allows users to enroll/register new faces into the system database
using either live camera feed or image files. It uses DeGirum SDK for high-accuracy
face detection and feature extraction.

Author: AI Assistant
Date: 2025
"""

import argparse
import logging
import time
import cv2
import numpy as np
import yaml
import sqlite3
import pickle
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import uuid

try:
    import degirum as dg
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please install DeGirum SDK: pip install degirum")
    exit(1)

from src.rtsp_manager import RTSPStreamManager
from src.face_database import FaceDatabase
from src.degirum_inference import DeGirumInferenceEngine


class FaceEnrollmentSystem:
    """Face enrollment system for registering new faces"""
    
    def __init__(self, config_path: str = "config/dual_rtsp.yaml"):
        """Initialize the face enrollment system
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.db = FaceDatabase(self.config['database']['path'])
        self.inference_engine = DeGirumInferenceEngine(self.config)
        self.stream_manager = None
        
        # Enrollment settings
        self.enrollment_config = self.config['face_recognition']['enrollment']
        self.min_face_size = self.enrollment_config['min_face_size']
        self.quality_threshold = self.enrollment_config['quality_threshold']
        self.num_samples = self.enrollment_config['num_samples_required']
        
        self.logger.info("Face enrollment system initialized")
    
    def enroll_from_camera(self, person_name: str, camera_index: int = 0) -> bool:
        """Enroll a person using live camera feed
        
        Args:
            person_name: Name of the person to enroll
            camera_index: Which camera to use (0 or 1)
            
        Returns:
            bool: True if enrollment successful, False otherwise
        """
        try:
            # Initialize stream manager for selected camera
            camera_config = {}
            if camera_index == 0:
                camera_config['camera_0'] = self.config['cameras']['camera_0']
            else:
                camera_config['camera_1'] = self.config['cameras']['camera_1']
            
            self.stream_manager = RTSPStreamManager(camera_config)
            self.stream_manager.start_streams()
            
            collected_samples = []
            samples_needed = self.num_samples
            
            print(f"\nEnrolling {person_name}")
            print(f"Please look at camera {camera_index}")
            print(f"Need to collect {samples_needed} good quality face samples")
            print("Press 'c' to capture sample, 'q' to quit, 's' to skip current frame")
            
            while len(collected_samples) < samples_needed:
                # Get frame from camera
                if camera_index == 0:
                    frame = self.stream_manager.get_frame('camera_0')
                else:
                    frame = self.stream_manager.get_frame('camera_1')
                
                if frame is None:
                    continue
                
                # Detect faces in frame
                detections = self.inference_engine.detect_faces(frame)
                
                # Draw face boxes and quality scores
                display_frame = frame.copy()
                best_face = None
                best_quality = 0
                
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    confidence = detection['confidence']
                    
                    # Calculate face quality
                    face_roi = frame[y1:y2, x1:x2]
                    quality_score = self._calculate_face_quality(face_roi)
                    
                    # Draw bounding box
                    color = (0, 255, 0) if quality_score > self.quality_threshold else (0, 165, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw quality score
                    cv2.putText(display_frame, f"Quality: {quality_score:.2f}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Track best face
                    if quality_score > best_quality and quality_score > self.quality_threshold:
                        best_face = detection
                        best_quality = quality_score
                
                # Display instructions
                cv2.putText(display_frame, f"Samples: {len(collected_samples)}/{samples_needed}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if best_face:
                    cv2.putText(display_frame, "Good quality face detected!", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(f"Face Enrollment - Camera {camera_index}", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c') and best_face:
                    # Capture the best face sample
                    x1, y1, x2, y2 = best_face['bbox']
                    face_roi = frame[y1:y2, x1:x2]
                    
                    # Extract face features
                    features = self.inference_engine.extract_face_features(face_roi)
                    if features is not None:
                        collected_samples.append({
                            'image': face_roi,
                            'features': features,
                            'quality': best_quality,
                            'bbox': best_face['bbox']
                        })
                        print(f"Sample {len(collected_samples)}/{samples_needed} captured (Quality: {best_quality:.2f})")
                elif key == ord('c') and not best_face:
                    print("No good quality face detected. Please try again.")
            
            cv2.destroyAllWindows()
            self.stream_manager.stop_streams()
            
            if len(collected_samples) >= samples_needed:
                # Save to database
                success = self._save_enrolled_face(person_name, collected_samples)
                if success:
                    print(f"Successfully enrolled {person_name} with {len(collected_samples)} samples")
                    return True
                else:
                    print(f"Failed to save enrollment for {person_name}")
                    return False
            else:
                print(f"Enrollment cancelled. Only collected {len(collected_samples)}/{samples_needed} samples")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during camera enrollment: {e}")
            if self.stream_manager:
                self.stream_manager.stop_streams()
            cv2.destroyAllWindows()
            return False
    
    def enroll_from_images(self, person_name: str, image_paths: List[str]) -> bool:
        """Enroll a person using image files
        
        Args:
            person_name: Name of the person to enroll
            image_paths: List of image file paths
            
        Returns:
            bool: True if enrollment successful, False otherwise
        """
        try:
            collected_samples = []
            
            print(f"\nEnrolling {person_name} from {len(image_paths)} images")
            
            for i, image_path in enumerate(image_paths):
                print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image {image_path}")
                    continue
                
                # Detect faces in image
                detections = self.inference_engine.detect_faces(image)
                
                if not detections:
                    print(f"Warning: No faces detected in {image_path}")
                    continue
                
                # Process each detected face
                for j, detection in enumerate(detections):
                    x1, y1, x2, y2 = detection['bbox']
                    face_roi = image[y1:y2, x1:x2]
                    
                    # Calculate face quality
                    quality_score = self._calculate_face_quality(face_roi)
                    
                    if quality_score > self.quality_threshold:
                        # Extract face features
                        features = self.inference_engine.extract_face_features(face_roi)
                        if features is not None:
                            collected_samples.append({
                                'image': face_roi,
                                'features': features,
                                'quality': quality_score,
                                'bbox': detection['bbox'],
                                'source_image': image_path
                            })
                            print(f"  Face {j+1}: Quality {quality_score:.2f} - Added")
                        else:
                            print(f"  Face {j+1}: Feature extraction failed")
                    else:
                        print(f"  Face {j+1}: Quality {quality_score:.2f} - Too low (threshold: {self.quality_threshold})")
            
            if len(collected_samples) >= self.num_samples:
                # Save to database
                success = self._save_enrolled_face(person_name, collected_samples)
                if success:
                    print(f"Successfully enrolled {person_name} with {len(collected_samples)} samples")
                    return True
                else:
                    print(f"Failed to save enrollment for {person_name}")
                    return False
            else:
                print(f"Insufficient good quality samples. Found {len(collected_samples)}, need {self.num_samples}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during image enrollment: {e}")
            return False
    
    def _calculate_face_quality(self, face_image: np.ndarray) -> float:
        """Calculate face quality score based on various metrics
        
        Args:
            face_image: Face region image
            
        Returns:
            float: Quality score between 0 and 1
        """
        if face_image.size == 0:
            return 0.0
        
        # Check minimum size
        h, w = face_image.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            return 0.0
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # Calculate brightness (avoid too dark or too bright)
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5
        
        # Calculate contrast
        contrast = gray.std()
        contrast_score = min(contrast / 64.0, 1.0)  # Normalize
        
        # Check for blur (using gradient magnitude)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_score = min(np.mean(gradient_magnitude) / 50.0, 1.0)
        
        # Combine scores with weights
        quality_score = (
            0.3 * sharpness_score +
            0.2 * brightness_score +
            0.2 * contrast_score +
            0.3 * edge_score
        )
        
        return quality_score
    
    def _save_enrolled_face(self, person_name: str, samples: List[Dict]) -> bool:
        """Save enrolled face samples to database
        
        Args:
            person_name: Name of the person
            samples: List of face samples with features
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Calculate average features from all samples
            feature_vectors = [sample['features'] for sample in samples]
            avg_features = np.mean(feature_vectors, axis=0)
            
            # Select best quality sample for thumbnail
            best_sample = max(samples, key=lambda x: x['quality'])
            thumbnail_image = best_sample['image']
            
            # Save to database
            face_id = self.db.add_person(
                name=person_name,
                features=avg_features,
                thumbnail_image=thumbnail_image,
                metadata={
                    'num_samples': len(samples),
                    'avg_quality': np.mean([s['quality'] for s in samples]),
                    'enrollment_date': datetime.now().isoformat(),
                    'samples': [
                        {
                            'quality': s['quality'],
                            'bbox': s['bbox'],
                            'source_image': s.get('source_image', 'camera')
                        } for s in samples
                    ]
                }
            )
            
            if face_id:
                self.logger.info(f"Enrolled {person_name} with ID {face_id}")
                return True
            else:
                self.logger.error(f"Failed to save enrollment for {person_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving enrolled face: {e}")
            return False
    
    def list_enrolled_faces(self) -> List[Dict]:
        """List all enrolled faces
        
        Returns:
            List of enrolled face information
        """
        return self.db.get_all_persons()
    
    def delete_enrolled_face(self, person_id: int) -> bool:
        """Delete an enrolled face
        
        Args:
            person_id: ID of the person to delete
            
        Returns:
            bool: True if deleted successfully
        """
        return self.db.delete_person(person_id)


def main():
    """Main function for face enrollment"""
    parser = argparse.ArgumentParser(description="Face Enrollment System")
    parser.add_argument('--config', type=str, default='config/dual_rtsp.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['camera', 'images', 'list', 'delete'],
                       required=True, help='Enrollment mode')
    parser.add_argument('--name', type=str, help='Person name to enroll')
    parser.add_argument('--camera', type=int, choices=[0, 1], default=0,
                       help='Camera index for live enrollment')
    parser.add_argument('--images', type=str, nargs='+',
                       help='Image file paths for enrollment')
    parser.add_argument('--person-id', type=int,
                       help='Person ID for deletion')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize enrollment system
    enrollment_system = FaceEnrollmentSystem(args.config)
    
    try:
        if args.mode == 'camera':
            if not args.name:
                print("Error: --name is required for camera enrollment")
                return
            
            success = enrollment_system.enroll_from_camera(args.name, args.camera)
            if success:
                print("Enrollment completed successfully!")
            else:
                print("Enrollment failed!")
        
        elif args.mode == 'images':
            if not args.name or not args.images:
                print("Error: --name and --images are required for image enrollment")
                return
            
            success = enrollment_system.enroll_from_images(args.name, args.images)
            if success:
                print("Enrollment completed successfully!")
            else:
                print("Enrollment failed!")
        
        elif args.mode == 'list':
            faces = enrollment_system.list_enrolled_faces()
            if faces:
                print(f"\nEnrolled faces ({len(faces)}):")
                print("-" * 60)
                for face in faces:
                    metadata = face.get('metadata', {})
                    print(f"ID: {face['id']}")
                    print(f"Name: {face['name']}")
                    print(f"Enrollment Date: {metadata.get('enrollment_date', 'Unknown')}")
                    print(f"Number of Samples: {metadata.get('num_samples', 'Unknown')}")
                    print(f"Average Quality: {metadata.get('avg_quality', 'Unknown'):.2f}")
                    print("-" * 60)
            else:
                print("No faces enrolled yet.")
        
        elif args.mode == 'delete':
            if not args.person_id:
                print("Error: --person-id is required for deletion")
                return
            
            success = enrollment_system.delete_enrolled_face(args.person_id)
            if success:
                print(f"Person ID {args.person_id} deleted successfully!")
            else:
                print(f"Failed to delete person ID {args.person_id}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
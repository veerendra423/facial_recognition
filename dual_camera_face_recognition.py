#!/usr/bin/env python3
"""
Dual Camera Facial Recognition System with Raspberry Pi 5 and Hailo AI

This script implements a comprehensive facial recognition system using two cameras
on Raspberry Pi 5 with Hailo-8L AI accelerator for real-time face detection and recognition.

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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

try:
    from picamera2 import Picamera2, Preview
    import hailo
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure you're running on Raspberry Pi with required packages installed.")
    exit(1)

from src.camera_manager import DualCameraManager
from src.hailo_inference import HailoInferenceEngine
from src.face_database import FaceDatabase
from src.display_manager import DisplayManager
from src.config_manager import ConfigManager
from src.utils import setup_logging, create_directories


@dataclass
class DetectionResult:
    """Container for face detection results"""
    camera_id: int
    frame: np.ndarray
    faces: List[Dict[str, Any]]
    timestamp: datetime
    fps: float


class DualCameraFaceRecognition:
    """Main class for dual camera facial recognition system"""
    
    def __init__(self, config_path: str):
        """Initialize the dual camera face recognition system"""
        self.config = ConfigManager(config_path)
        self.logger = setup_logging(self.config.get('logging'))
        
        # Create necessary directories
        create_directories([
            'data', 'logs', 'debug',
            self.config.get('database.embeddings_path'),
            self.config.get('storage.storage_path')
        ])
        
        # Initialize components
        self.camera_manager = None
        self.hailo_engine = None
        self.face_database = None
        self.display_manager = None
        
        # Threading and synchronization
        self.running = False
        self.frame_queues = {0: queue.Queue(maxsize=5), 1: queue.Queue(maxsize=5)}
        self.result_queue = queue.Queue(maxsize=10)
        self.threads = []
        
        # Performance tracking
        self.fps_counters = {0: 0, 1: 0}
        self.last_fps_time = time.time()
        
        self.logger.info("DualCameraFaceRecognition initialized")
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            # Initialize camera manager
            self.camera_manager = DualCameraManager(self.config.get('cameras'))
            if not self.camera_manager.initialize():
                self.logger.error("Failed to initialize camera manager")
                return False
            
            # Initialize Hailo inference engine
            self.hailo_engine = HailoInferenceEngine(self.config.get('hailo'))
            if not self.hailo_engine.initialize():
                self.logger.error("Failed to initialize Hailo inference engine")
                return False
            
            # Initialize face database
            self.face_database = FaceDatabase(self.config.get('database'))
            if not self.face_database.initialize():
                self.logger.error("Failed to initialize face database")
                return False
            
            # Initialize display manager
            if self.config.get('display.enabled'):
                self.display_manager = DisplayManager(self.config.get('display'))
                if not self.display_manager.initialize():
                    self.logger.warning("Failed to initialize display manager")
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
    
    def camera_capture_thread(self, camera_id: int):
        """Thread function for capturing frames from a specific camera"""
        self.logger.info(f"Starting capture thread for camera {camera_id}")
        
        while self.running:
            try:
                frame = self.camera_manager.capture_frame(camera_id)
                if frame is not None:
                    # Add frame to queue (non-blocking)
                    try:
                        self.frame_queues[camera_id].put_nowait(frame)
                        self.fps_counters[camera_id] += 1
                    except queue.Full:
                        # Skip frame if queue is full
                        pass
                else:
                    time.sleep(0.01)  # Brief sleep if no frame available
                    
            except Exception as e:
                self.logger.error(f"Error in camera {camera_id} capture thread: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"Camera {camera_id} capture thread stopped")
    
    def face_processing_thread(self, camera_id: int):
        """Thread function for processing faces from a specific camera"""
        self.logger.info(f"Starting face processing thread for camera {camera_id}")
        
        while self.running:
            try:
                # Get frame from queue
                try:
                    frame = self.frame_queues[camera_id].get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process frame for face detection and recognition
                result = self.process_frame(camera_id, frame)
                if result:
                    # Add result to output queue
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        # Skip result if queue is full
                        pass
                
            except Exception as e:
                self.logger.error(f"Error in camera {camera_id} processing thread: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"Camera {camera_id} processing thread stopped")
    
    def process_frame(self, camera_id: int, frame: np.ndarray) -> Optional[DetectionResult]:
        """Process a single frame for face detection and recognition"""
        try:
            # Face detection using Hailo
            detections = self.hailo_engine.detect_faces(frame)
            
            # Process each detected face
            processed_faces = []
            for detection in detections:
                # Extract face region
                bbox = detection['bbox']
                confidence = detection['confidence']
                
                if confidence < self.config.get('hailo.models.face_detection.confidence_threshold'):
                    continue
                
                # Extract face for recognition
                face_image = self.extract_face_region(frame, bbox)
                if face_image is None:
                    continue
                
                # Get face embedding
                embedding = self.hailo_engine.get_face_embedding(face_image)
                if embedding is None:
                    continue
                
                # Match against database
                person_id, similarity = self.face_database.match_face(embedding)
                
                face_info = {
                    'bbox': bbox,
                    'confidence': confidence,
                    'embedding': embedding,
                    'person_id': person_id,
                    'similarity': similarity,
                    'timestamp': datetime.now()
                }
                processed_faces.append(face_info)
            
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps = self.fps_counters[camera_id] / (current_time - self.last_fps_time)
                self.fps_counters[camera_id] = 0
                self.last_fps_time = current_time
            else:
                fps = 0
            
            return DetectionResult(
                camera_id=camera_id,
                frame=frame,
                faces=processed_faces,
                timestamp=datetime.now(),
                fps=fps
            )
            
        except Exception as e:
            self.logger.error(f"Error processing frame from camera {camera_id}: {e}")
            return None
    
    def extract_face_region(self, frame: np.ndarray, bbox: Dict[str, float]) -> Optional[np.ndarray]:
        """Extract face region from frame using bounding box"""
        try:
            h, w = frame.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int(bbox['xmin'] * w)
            y1 = int(bbox['ymin'] * h)
            x2 = int(bbox['xmax'] * w)
            y2 = int(bbox['ymax'] * h)
            
            # Add padding
            padding = self.config.get('processing.face_padding', 0.2)
            width = x2 - x1
            height = y2 - y1
            
            x1 = max(0, x1 - int(width * padding))
            y1 = max(0, y1 - int(height * padding))
            x2 = min(w, x2 + int(width * padding))
            y2 = min(h, y2 + int(height * padding))
            
            # Extract face region
            face_image = frame[y1:y2, x1:x2]
            
            if face_image.size == 0:
                return None
            
            # Resize to standard size for recognition
            face_image = cv2.resize(face_image, (112, 112))
            return face_image
            
        except Exception as e:
            self.logger.error(f"Error extracting face region: {e}")
            return None
    
    def display_thread(self):
        """Thread function for displaying results"""
        if not self.display_manager:
            return
            
        self.logger.info("Starting display thread")
        
        while self.running:
            try:
                # Get results from queue
                results = []
                try:
                    # Collect results for both cameras
                    while len(results) < 2:
                        result = self.result_queue.get(timeout=0.1)
                        results.append(result)
                except queue.Empty:
                    if results:
                        # Display available results
                        self.display_manager.update_display(results)
                    continue
                
                # Update display with results
                self.display_manager.update_display(results)
                
            except Exception as e:
                self.logger.error(f"Error in display thread: {e}")
                time.sleep(0.1)
        
        self.logger.info("Display thread stopped")
    
    def start(self):
        """Start the dual camera face recognition system"""
        self.logger.info("Starting dual camera face recognition system")
        
        if not self.initialize_components():
            self.logger.error("Failed to initialize components")
            return False
        
        self.running = True
        
        # Start camera capture threads
        for camera_id in [0, 1]:
            if self.config.get(f'cameras.camera_{camera_id}.enabled'):
                capture_thread = threading.Thread(
                    target=self.camera_capture_thread,
                    args=(camera_id,),
                    name=f"Camera{camera_id}Capture"
                )
                capture_thread.start()
                self.threads.append(capture_thread)
                
                # Start processing thread for this camera
                process_thread = threading.Thread(
                    target=self.face_processing_thread,
                    args=(camera_id,),
                    name=f"Camera{camera_id}Process"
                )
                process_thread.start()
                self.threads.append(process_thread)
        
        # Start display thread
        if self.display_manager:
            display_thread = threading.Thread(
                target=self.display_thread,
                name="Display"
            )
            display_thread.start()
            self.threads.append(display_thread)
        
        self.logger.info("All threads started successfully")
        return True
    
    def stop(self):
        """Stop the dual camera face recognition system"""
        self.logger.info("Stopping dual camera face recognition system")
        
        self.running = False
        
        # Wait for all threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Cleanup components
        if self.camera_manager:
            self.camera_manager.cleanup()
        if self.hailo_engine:
            self.hailo_engine.cleanup()
        if self.face_database:
            self.face_database.cleanup()
        if self.display_manager:
            self.display_manager.cleanup()
        
        self.logger.info("System stopped successfully")
    
    def run(self):
        """Main run loop"""
        try:
            if not self.start():
                return False
            
            self.logger.info("System running. Press Ctrl+C to stop.")
            
            # Main loop
            while self.running:
                try:
                    time.sleep(1.0)
                    # Periodic tasks can be added here
                    
                except KeyboardInterrupt:
                    self.logger.info("Keyboard interrupt received")
                    break
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in main run loop: {e}")
            return False
        finally:
            self.stop()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Dual Camera Facial Recognition System")
    parser.add_argument(
        '--config',
        type=str,
        default='config/dual_camera.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Check if configuration file exists
    if not Path(args.config).exists():
        print(f"Configuration file not found: {args.config}")
        return 1
    
    try:
        # Create and run the system
        system = DualCameraFaceRecognition(args.config)
        success = system.run()
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error running system: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
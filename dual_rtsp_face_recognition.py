#!/usr/bin/env python3
"""
Dual RTSP Camera Facial Recognition System with Raspberry Pi 5 and DeGirum AI

This script implements a comprehensive facial recognition system using two RTSP cameras
on Raspberry Pi 5 with DeGirum AI SDK for real-time face detection and recognition
optimized for high accuracy.

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
import gc

try:
    import degirum as dg
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please install DeGirum SDK: pip install degirum")
    exit(1)

from src.rtsp_manager import RTSPStreamManager
from src.degirum_inference import DeGirumInferenceEngine
from src.face_database import FaceDatabase
from src.display_manager import DisplayManager
from src.config_manager import ConfigManager
from src.utils import setup_logging, create_directories, calculate_face_quality
from src.face_tracker import FaceTracker


@dataclass
class DetectionResult:
    """Container for face detection results"""
    camera_id: int
    camera_name: str
    frame: np.ndarray
    faces: List[Dict[str, Any]]
    timestamp: datetime
    fps: float
    quality_score: float


class DualRTSPFaceRecognition:
    """Main class for dual RTSP camera facial recognition system with DeGirum"""
    
    def __init__(self, config_path: str):
        """Initialize the dual RTSP camera face recognition system"""
        self.config = ConfigManager(config_path)
        self.logger = setup_logging(self.config.get('logging'))
        
        # Create necessary directories
        create_directories([
            'data', 'logs', 'debug',
            self.config.get('database.embeddings_path'),
            self.config.get('storage.storage_path')
        ])
        
        # Initialize components
        self.rtsp_manager = None
        self.degirum_engine = None
        self.face_database = None
        self.display_manager = None
        self.face_tracker = None
        
        # Threading and synchronization
        self.running = False
        self.frame_queues = {0: queue.Queue(maxsize=10), 1: queue.Queue(maxsize=10)}
        self.result_queue = queue.Queue(maxsize=20)
        self.threads = []
        
        # Performance tracking
        self.fps_counters = {0: 0, 1: 0}
        self.last_fps_time = {0: time.time(), 1: time.time()}
        self.frame_counts = {0: 0, 1: 0}
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'average_confidence': 0.0,
            'average_quality': 0.0
        }
        
        self.logger.info("DualRTSPFaceRecognition initialized")
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            # Initialize RTSP stream manager
            self.rtsp_manager = RTSPStreamManager(self.config.get('cameras'))
            if not self.rtsp_manager.initialize():
                self.logger.error("Failed to initialize RTSP manager")
                return False
            
            # Initialize DeGirum inference engine
            self.degirum_engine = DeGirumInferenceEngine(self.config.get('degirum'))
            if not self.degirum_engine.initialize():
                self.logger.error("Failed to initialize DeGirum inference engine")
                return False
            
            # Initialize face database
            self.face_database = FaceDatabase(self.config.get('database'))
            if not self.face_database.initialize():
                self.logger.error("Failed to initialize face database")
                return False
            
            # Initialize face tracker if enabled
            if self.config.get('security.face_tracking'):
                self.face_tracker = FaceTracker(self.config.get('security'))
                self.logger.info("Face tracker initialized")
            
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
        """Thread function for capturing frames from a specific RTSP camera"""
        camera_config = self.config.get(f'cameras.camera_{camera_id}')
        camera_name = camera_config.get('name', f'Camera {camera_id}')
        
        self.logger.info(f"Starting capture thread for {camera_name}")
        
        while self.running:
            try:
                frame = self.rtsp_manager.get_frame(camera_id)
                if frame is not None:
                    # Apply quality adjustments if configured
                    frame = self._apply_quality_adjustments(frame, camera_config.get('quality', {}))
                    
                    # Scale frame if needed
                    frame = self._scale_frame(frame, camera_config.get('stream', {}))
                    
                    # Add frame to queue (non-blocking)
                    try:
                        self.frame_queues[camera_id].put_nowait({
                            'frame': frame,
                            'timestamp': datetime.now(),
                            'camera_name': camera_name
                        })
                        self.fps_counters[camera_id] += 1
                    except queue.Full:
                        # Skip frame if queue is full
                        pass
                else:
                    time.sleep(0.01)  # Brief sleep if no frame available
                    
            except Exception as e:
                self.logger.error(f"Error in {camera_name} capture thread: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"{camera_name} capture thread stopped")
    
    def face_processing_thread(self, camera_id: int):
        """Thread function for processing faces from a specific RTSP camera"""
        camera_config = self.config.get(f'cameras.camera_{camera_id}')
        camera_name = camera_config.get('name', f'Camera {camera_id}')
        
        self.logger.info(f"Starting face processing thread for {camera_name}")
        
        while self.running:
            try:
                # Get frame from queue
                try:
                    frame_data = self.frame_queues[camera_id].get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process frame for face detection and recognition
                result = self.process_frame(camera_id, frame_data)
                if result:
                    # Add result to output queue
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        # Skip result if queue is full
                        pass
                
            except Exception as e:
                self.logger.error(f"Error in {camera_name} processing thread: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"{camera_name} processing thread stopped")
    
    def process_frame(self, camera_id: int, frame_data: Dict[str, Any]) -> Optional[DetectionResult]:
        """Process a single frame for face detection and recognition with high accuracy"""
        try:
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            camera_name = frame_data['camera_name']
            
            # Calculate frame quality
            quality_score = calculate_face_quality(frame)
            
            # Face detection using DeGirum
            detections = self.degirum_engine.detect_faces(frame)
            
            # Filter detections by confidence and size
            detection_config = self.config.get('processing.detection')
            min_confidence = detection_config.get('min_face_confidence', 0.6)
            
            filtered_detections = []
            for detection in detections:
                if detection.get('confidence', 0) >= min_confidence:
                    # Check face size
                    bbox = detection['bbox']
                    face_width = bbox['width'] * frame.shape[1]
                    face_height = bbox['height'] * frame.shape[0]
                    face_size = min(face_width, face_height)
                    
                    min_size = self.config.get('degirum.face_detection.min_face_size', 50)
                    max_size = self.config.get('degirum.face_detection.max_face_size', 500)
                    
                    if min_size <= face_size <= max_size:
                        filtered_detections.append(detection)
            
            # Process each detected face
            processed_faces = []
            for detection in filtered_detections[:detection_config.get('max_faces_per_frame', 15)]:
                face_result = self._process_single_face(frame, detection, timestamp)
                if face_result:
                    processed_faces.append(face_result)
            
            # Update face tracking if enabled
            if self.face_tracker:
                self.face_tracker.update_tracks(camera_id, processed_faces)
            
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_fps_time[camera_id] >= 1.0:
                fps = self.fps_counters[camera_id] / (current_time - self.last_fps_time[camera_id])
                self.fps_counters[camera_id] = 0
                self.last_fps_time[camera_id] = current_time
            else:
                fps = 0
            
            # Update statistics
            self.stats['total_frames_processed'] += 1
            self.stats['total_faces_detected'] += len(processed_faces)
            
            return DetectionResult(
                camera_id=camera_id,
                camera_name=camera_name,
                frame=frame,
                faces=processed_faces,
                timestamp=timestamp,
                fps=fps,
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Error processing frame from camera {camera_id}: {e}")
            return None
    
    def _process_single_face(self, frame: np.ndarray, detection: Dict[str, Any], timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Process a single detected face for recognition"""
        try:
            # Extract face region
            face_image = self._extract_face_region(frame, detection['bbox'])
            if face_image is None:
                return None
            
            # Quality check
            recognition_config = self.config.get('processing.recognition')
            if recognition_config.get('enable_quality_check', True):
                face_quality = calculate_face_quality(face_image)
                min_quality = recognition_config.get('min_quality_score', 0.6)
                if face_quality < min_quality:
                    self.logger.debug(f"Face quality too low: {face_quality:.3f} < {min_quality}")
                    return None
            
            # Face enhancement if enabled
            if recognition_config.get('enable_face_enhancement', True):
                face_image = self._enhance_face_image(face_image, recognition_config.get('enhancement_method'))
            
            # Get face embedding using DeGirum
            embedding = self.degirum_engine.get_face_embedding(face_image)
            if embedding is None:
                return None
            
            # Match against database
            person_id, similarity = self.face_database.match_face(
                embedding, 
                threshold=self.config.get('degirum.face_recognition.similarity_threshold', 0.65)
            )
            
            # Multi-frame verification if enabled
            if self.config.get('security.enable_multi_frame_verification', True):
                verification_result = self._verify_multi_frame(person_id, similarity, timestamp)
                if not verification_result:
                    person_id = None
                    similarity = 0.0
            
            # Update statistics
            if person_id:
                self.stats['total_faces_recognized'] += 1
                self.stats['average_confidence'] = (
                    self.stats['average_confidence'] * (self.stats['total_faces_recognized'] - 1) + 
                    detection['confidence']
                ) / self.stats['total_faces_recognized']
            
            face_result = {
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'embedding': embedding,
                'person_id': person_id,
                'similarity': similarity,
                'timestamp': timestamp,
                'quality_score': face_quality if 'face_quality' in locals() else 0.0,
                'landmarks': detection.get('landmarks', [])
            }
            
            # Save high-quality unknown faces if configured
            if (not person_id and 
                self.config.get('storage.save_unknown_faces', True) and
                face_quality > self.config.get('database.quality_threshold', 0.7)):
                self._save_unknown_face(face_image, face_result)
            
            return face_result
            
        except Exception as e:
            self.logger.error(f"Error processing single face: {e}")
            return None
    
    def _extract_face_region(self, frame: np.ndarray, bbox: Dict[str, float]) -> Optional[np.ndarray]:
        """Extract face region from frame using bounding box with proper alignment"""
        try:
            h, w = frame.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            x = int(bbox['x'] * w)
            y = int(bbox['y'] * h)
            width = int(bbox['width'] * w)
            height = int(bbox['height'] * h)
            
            # Calculate crop region with margin
            crop_margin = self.config.get('degirum.face_recognition.crop_margin', 0.3)
            margin_x = int(width * crop_margin)
            margin_y = int(height * crop_margin)
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + width + margin_x)
            y2 = min(h, y + height + margin_y)
            
            # Extract face region
            face_image = frame[y1:y2, x1:x2]
            
            if face_image.size == 0:
                return None
            
            # Resize to model input size
            input_size = self.config.get('degirum.face_recognition.input_size', [160, 160])
            face_image = cv2.resize(face_image, tuple(input_size))
            
            return face_image
            
        except Exception as e:
            self.logger.error(f"Error extracting face region: {e}")
            return None
    
    def _apply_quality_adjustments(self, frame: np.ndarray, quality_config: Dict[str, Any]) -> np.ndarray:
        """Apply quality adjustments to frame"""
        try:
            # Brightness adjustment
            brightness = quality_config.get('brightness_adjustment', 0)
            if brightness != 0:
                frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
            
            # Contrast adjustment
            contrast = quality_config.get('contrast_adjustment', 0)
            if contrast != 0:
                alpha = 1.0 + contrast / 100.0
                frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
            
            # Saturation adjustment
            saturation = quality_config.get('saturation_adjustment', 0)
            if saturation != 0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.0 + saturation / 100.0)
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error applying quality adjustments: {e}")
            return frame
    
    def _scale_frame(self, frame: np.ndarray, stream_config: Dict[str, Any]) -> np.ndarray:
        """Scale frame according to configuration"""
        try:
            target_width = stream_config.get('scale_to_width')
            target_height = stream_config.get('scale_to_height')
            maintain_aspect = stream_config.get('maintain_aspect_ratio', True)
            
            if target_width and target_height:
                h, w = frame.shape[:2]
                
                if maintain_aspect:
                    # Calculate scaling factor
                    scale_x = target_width / w
                    scale_y = target_height / h
                    scale = min(scale_x, scale_y)
                    
                    new_width = int(w * scale)
                    new_height = int(h * scale)
                else:
                    new_width = target_width
                    new_height = target_height
                
                frame = cv2.resize(frame, (new_width, new_height))
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error scaling frame: {e}")
            return frame
    
    def _enhance_face_image(self, face_image: np.ndarray, method: str) -> np.ndarray:
        """Enhance face image for better recognition accuracy"""
        try:
            if method == "histogram_equalization":
                # Convert to LAB color space and equalize L channel
                lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
                face_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
            elif method == "clahe":
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                face_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
            elif method == "gamma":
                # Apply gamma correction
                gamma = 1.2
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                       for i in np.arange(0, 256)]).astype("uint8")
                face_image = cv2.LUT(face_image, lookup_table)
            
            return face_image
            
        except Exception as e:
            self.logger.error(f"Error enhancing face image: {e}")
            return face_image
    
    def _verify_multi_frame(self, person_id: Optional[str], similarity: float, timestamp: datetime) -> bool:
        """Verify face recognition across multiple frames"""
        # Implementation would track recognition results across frames
        # For now, return True if similarity is above threshold
        verification_threshold = self.config.get('security.verification_threshold', 0.7)
        return similarity >= verification_threshold
    
    def _save_unknown_face(self, face_image: np.ndarray, face_result: Dict[str, Any]):
        """Save unknown face for later analysis"""
        try:
            storage_path = Path(self.config.get('storage.storage_path', 'data/captures'))
            unknown_path = storage_path / 'unknown_faces'
            unknown_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = face_result['timestamp'].strftime('%Y%m%d_%H%M%S_%f')
            filename = f"unknown_{timestamp}.jpg"
            filepath = unknown_path / filename
            
            # Save with high quality
            quality = self.config.get('storage.image_quality', 95)
            cv2.imwrite(str(filepath), face_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            self.logger.debug(f"Saved unknown face: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving unknown face: {e}")
    
    def display_thread(self):
        """Thread function for displaying results"""
        if not self.display_manager:
            return
            
        self.logger.info("Starting display thread")
        
        while self.running:
            try:
                # Collect results from both cameras
                results = {}
                timeout = 0.1
                
                # Try to get recent results for both cameras
                start_time = time.time()
                while len(results) < 2 and (time.time() - start_time) < timeout:
                    try:
                        result = self.result_queue.get(timeout=0.05)
                        results[result.camera_id] = result
                    except queue.Empty:
                        break
                
                if results:
                    # Update display with available results
                    self.display_manager.update_display(list(results.values()), self.stats)
                else:
                    time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in display thread: {e}")
                time.sleep(0.1)
        
        self.logger.info("Display thread stopped")
    
    def statistics_thread(self):
        """Thread for updating and logging statistics"""
        while self.running:
            try:
                time.sleep(10)  # Update every 10 seconds
                
                # Log statistics
                self.logger.info(
                    f"Stats - Frames: {self.stats['total_frames_processed']}, "
                    f"Faces detected: {self.stats['total_faces_detected']}, "
                    f"Faces recognized: {self.stats['total_faces_recognized']}, "
                    f"Avg confidence: {self.stats['average_confidence']:.3f}"
                )
                
                # Periodic garbage collection
                if self.stats['total_frames_processed'] % 1000 == 0:
                    gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error in statistics thread: {e}")
                time.sleep(1)
    
    def start(self) -> bool:
        """Start the dual RTSP camera face recognition system"""
        self.logger.info("Starting dual RTSP camera face recognition system")
        
        if not self.initialize_components():
            self.logger.error("Failed to initialize components")
            return False
        
        self.running = True
        
        # Start RTSP streams
        if not self.rtsp_manager.start_streams():
            self.logger.error("Failed to start RTSP streams")
            return False
        
        # Start processing threads for each enabled camera
        for camera_id in [0, 1]:
            camera_config = self.config.get(f'cameras.camera_{camera_id}')
            if camera_config and camera_config.get('enabled', False):
                # Start capture thread
                capture_thread = threading.Thread(
                    target=self.camera_capture_thread,
                    args=(camera_id,),
                    name=f"RTSPCapture{camera_id}",
                    daemon=True
                )
                capture_thread.start()
                self.threads.append(capture_thread)
                
                # Start processing thread
                process_thread = threading.Thread(
                    target=self.face_processing_thread,
                    args=(camera_id,),
                    name=f"FaceProcess{camera_id}",
                    daemon=True
                )
                process_thread.start()
                self.threads.append(process_thread)
        
        # Start display thread
        if self.display_manager:
            display_thread = threading.Thread(
                target=self.display_thread,
                name="Display",
                daemon=True
            )
            display_thread.start()
            self.threads.append(display_thread)
        
        # Start statistics thread
        stats_thread = threading.Thread(
            target=self.statistics_thread,
            name="Statistics",
            daemon=True
        )
        stats_thread.start()
        self.threads.append(stats_thread)
        
        self.logger.info("All threads started successfully")
        return True
    
    def stop(self):
        """Stop the dual RTSP camera face recognition system"""
        self.logger.info("Stopping dual RTSP camera face recognition system")
        
        self.running = False
        
        # Stop RTSP streams
        if self.rtsp_manager:
            self.rtsp_manager.stop_streams()
        
        # Wait for all threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Cleanup components
        if self.rtsp_manager:
            self.rtsp_manager.cleanup()
        if self.degirum_engine:
            self.degirum_engine.cleanup()
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
                    # Periodic maintenance tasks can be added here
                    
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
    parser = argparse.ArgumentParser(description="Dual RTSP Camera Facial Recognition System with DeGirum")
    parser.add_argument(
        '--config',
        type=str,
        default='config/dual_rtsp.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--test-streams',
        action='store_true',
        help='Test RTSP stream connections only'
    )
    
    args = parser.parse_args()
    
    # Check if configuration file exists
    if not Path(args.config).exists():
        print(f"Configuration file not found: {args.config}")
        return 1
    
    try:
        if args.test_streams:
            # Test RTSP streams only
            from src.rtsp_manager import RTSPStreamManager
            config = ConfigManager(args.config)
            rtsp_manager = RTSPStreamManager(config.get('cameras'))
            return 0 if rtsp_manager.test_connections() else 1
        
        # Create and run the system
        system = DualRTSPFaceRecognition(args.config)
        success = system.run()
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error running system: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Main entry point for Dual RTSP Camera Facial Recognition System
Author: AI Assistant
Date: 2025
"""

import argparse
import sys
import time
import signal
import threading
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

from config import Config
from modules.detection import FaceDetector
from modules.recognition import FaceRecognizer
from modules.tracking import FaceTracker
from modules.logging import setup_logger
import cv2
import numpy as np
from datetime import datetime


class DualCameraSystem:
    """Main system class for dual camera facial recognition"""
    
    def __init__(self, config_file="config.yaml"):
        """Initialize the dual camera system"""
        self.config = Config(config_file)
        self.logger = setup_logger("main", self.config.logging)
        
        # Initialize components
        self.detector = FaceDetector(self.config)
        self.recognizer = FaceRecognizer(self.config)
        self.tracker = FaceTracker(self.config)
        
        # System state
        self.running = False
        self.threads = []
        
        # Camera streams
        self.cameras = {}
        self.latest_frames = {0: None, 1: None}
        self.frame_locks = {0: threading.Lock(), 1: threading.Lock()}
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'faces_recognized': 0,
            'fps_camera_0': 0,
            'fps_camera_1': 0
        }
        
        self.logger.info("Dual camera system initialized")
    
    def initialize_cameras(self):
        """Initialize RTSP camera connections"""
        try:
            for camera_id in [0, 1]:
                camera_config = self.config.cameras[f'camera_{camera_id}']
                if camera_config['enabled']:
                    rtsp_url = camera_config['rtsp_url']
                    self.logger.info(f"Connecting to camera {camera_id}: {rtsp_url}")
                    
                    # Create VideoCapture object
                    cap = cv2.VideoCapture(rtsp_url)
                    
                    # Set buffer size to reduce latency
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test connection
                    ret, frame = cap.read()
                    if ret:
                        self.cameras[camera_id] = cap
                        self.logger.info(f"Camera {camera_id} connected successfully")
                    else:
                        self.logger.error(f"Failed to connect to camera {camera_id}")
                        return False
            
            return len(self.cameras) > 0
            
        except Exception as e:
            self.logger.error(f"Error initializing cameras: {e}")
            return False
    
    def camera_thread(self, camera_id):
        """Thread function for capturing frames from a camera"""
        cap = self.cameras[camera_id]
        camera_name = self.config.cameras[f'camera_{camera_id}']['name']
        
        self.logger.info(f"Starting capture thread for {camera_name}")
        
        fps_counter = 0
        last_fps_time = time.time()
        
        while self.running:
            try:
                ret, frame = cap.read()
                if ret:
                    # Store latest frame
                    with self.frame_locks[camera_id]:
                        self.latest_frames[camera_id] = frame.copy()
                    
                    fps_counter += 1
                    
                    # Calculate FPS
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        self.stats[f'fps_camera_{camera_id}'] = fps_counter
                        fps_counter = 0
                        last_fps_time = current_time
                
                else:
                    self.logger.warning(f"Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error in camera {camera_id} thread: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"Camera {camera_id} thread stopped")
    
    def process_frame(self, camera_id, frame):
        """Process a single frame for face detection and recognition"""
        try:
            # Detect faces
            detections = self.detector.detect_faces(frame)
            
            # Process each detection
            results = []
            for detection in detections:
                # Extract face
                face_roi = self.extract_face_roi(frame, detection)
                if face_roi is None:
                    continue
                
                # Recognize face
                person_id, confidence = self.recognizer.recognize_face(face_roi)
                
                # Update tracking
                track_id = self.tracker.update_track(camera_id, detection, person_id)
                
                result = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'person_id': person_id,
                    'recognition_confidence': confidence,
                    'track_id': track_id,
                    'timestamp': datetime.now()
                }
                results.append(result)
            
            # Update statistics
            self.stats['frames_processed'] += 1
            self.stats['faces_detected'] += len(detections)
            self.stats['faces_recognized'] += sum(1 for r in results if r['person_id'])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return []
    
    def extract_face_roi(self, frame, detection):
        """Extract face region of interest from frame"""
        try:
            bbox = detection['bbox']
            h, w = frame.shape[:2]
            
            # Convert normalized coordinates to pixels
            x1 = int(bbox['x'] * w)
            y1 = int(bbox['y'] * h)
            x2 = int((bbox['x'] + bbox['width']) * w)
            y2 = int((bbox['y'] + bbox['height']) * h)
            
            # Add padding
            padding = 0.2
            pad_x = int((x2 - x1) * padding)
            pad_y = int((y2 - y1) * padding)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
            
            # Resize to standard size
            face_roi = cv2.resize(face_roi, (160, 160))
            return face_roi
            
        except Exception as e:
            self.logger.error(f"Error extracting face ROI: {e}")
            return None
    
    def processing_thread(self):
        """Main processing thread"""
        self.logger.info("Starting processing thread")
        
        display_enabled = self.config.display.get('enabled', True)
        
        while self.running:
            try:
                frames_to_process = {}
                
                # Get latest frames from both cameras
                for camera_id in self.cameras.keys():
                    with self.frame_locks[camera_id]:
                        if self.latest_frames[camera_id] is not None:
                            frames_to_process[camera_id] = self.latest_frames[camera_id].copy()
                
                # Process frames
                all_results = {}
                for camera_id, frame in frames_to_process.items():
                    results = self.process_frame(camera_id, frame)
                    all_results[camera_id] = {
                        'frame': frame,
                        'results': results,
                        'camera_name': self.config.cameras[f'camera_{camera_id}']['name']
                    }
                
                # Display results if enabled
                if display_enabled and all_results:
                    self.display_results(all_results)
                
                time.sleep(0.03)  # ~30 FPS processing
                
            except Exception as e:
                self.logger.error(f"Error in processing thread: {e}")
                time.sleep(0.1)
        
        self.logger.info("Processing thread stopped")
    
    def display_results(self, all_results):
        """Display results from all cameras"""
        try:
            display_frames = []
            
            for camera_id in sorted(all_results.keys()):
                data = all_results[camera_id]
                frame = data['frame'].copy()
                results = data['results']
                camera_name = data['camera_name']
                
                # Draw detections
                for result in results:
                    bbox = result['bbox']
                    h, w = frame.shape[:2]
                    
                    x1 = int(bbox['x'] * w)
                    y1 = int(bbox['y'] * h)
                    x2 = int((bbox['x'] + bbox['width']) * w)
                    y2 = int((bbox['y'] + bbox['height']) * h)
                    
                    # Choose color based on recognition
                    if result['person_id']:
                        color = (0, 255, 0)  # Green for recognized
                        label = f"{result['person_id']} ({result['recognition_confidence']:.2f})"
                    else:
                        color = (0, 0, 255)  # Red for unknown
                        label = f"Unknown ({result['confidence']:.2f})"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add camera info
                info_text = f"{camera_name} - FPS: {self.stats[f'fps_camera_{camera_id}']}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                display_frames.append(frame)
            
            # Combine frames for display
            if len(display_frames) == 2:
                # Side by side
                combined = np.hstack(display_frames)
            elif len(display_frames) == 1:
                combined = display_frames[0]
            else:
                return
            
            # Resize for display if too large
            h, w = combined.shape[:2]
            if w > 1920:
                scale = 1920 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                combined = cv2.resize(combined, (new_w, new_h))
            
            # Add system stats
            stats_text = f"Frames: {self.stats['frames_processed']} | " \
                        f"Detected: {self.stats['faces_detected']} | " \
                        f"Recognized: {self.stats['faces_recognized']}"
            
            cv2.putText(combined, stats_text, (10, combined.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Dual Camera Facial Recognition", combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                self.stop()
            
        except Exception as e:
            self.logger.error(f"Error displaying results: {e}")
    
    def start(self):
        """Start the dual camera system"""
        self.logger.info("Starting dual camera system")
        
        # Initialize cameras
        if not self.initialize_cameras():
            self.logger.error("Failed to initialize cameras")
            return False
        
        self.running = True
        
        # Start camera threads
        for camera_id in self.cameras.keys():
            thread = threading.Thread(
                target=self.camera_thread,
                args=(camera_id,),
                name=f"Camera{camera_id}",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        # Start processing thread
        process_thread = threading.Thread(
            target=self.processing_thread,
            name="Processing",
            daemon=True
        )
        process_thread.start()
        self.threads.append(process_thread)
        
        self.logger.info("All threads started successfully")
        return True
    
    def stop(self):
        """Stop the dual camera system"""
        self.logger.info("Stopping dual camera system")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Release cameras
        for cap in self.cameras.values():
            cap.release()
        
        # Close display windows
        cv2.destroyAllWindows()
        
        self.logger.info("System stopped successfully")
    
    def run(self):
        """Main run loop"""
        try:
            if not self.start():
                return False
            
            self.logger.info("System running. Press 'q' or ESC to quit")
            
            # Keep main thread alive
            while self.running:
                time.sleep(1.0)
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            return False
        finally:
            self.stop()


def signal_handler(signum, frame):
    """Handle system signals"""
    print("\nShutdown signal received...")
    sys.exit(0)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Dual RTSP Camera Facial Recognition System")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display'
    )
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create and run system
        system = DualCameraSystem(args.config)
        
        # Override display setting if requested
        if args.no_display:
            system.config.display['enabled'] = False
        
        success = system.run()
        return 0 if success else 1
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
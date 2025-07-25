"""
Camera Manager for Dual Camera Setup on Raspberry Pi 5

This module handles the initialization and management of two cameras
connected to the Raspberry Pi 5's dual CSI ports.

Author: AI Assistant
Date: 2025
"""

import logging
import threading
import time
from typing import Dict, Optional, Any
import numpy as np
import cv2

try:
    from picamera2 import Picamera2, Preview
    from libcamera import controls
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure picamera2 is installed: sudo apt install python3-picamera2")


class DualCameraManager:
    """Manages dual camera setup for Raspberry Pi 5"""
    
    def __init__(self, camera_config: Dict[str, Any]):
        """Initialize dual camera manager
        
        Args:
            camera_config: Configuration dictionary for cameras
        """
        self.config = camera_config
        self.logger = logging.getLogger(__name__)
        
        # Camera instances
        self.cameras = {}
        self.camera_threads = {}
        self.running = False
        
        # Frame storage
        self.latest_frames = {0: None, 1: None}
        self.frame_locks = {0: threading.Lock(), 1: threading.Lock()}
        
        self.logger.info("DualCameraManager initialized")
    
    def initialize(self) -> bool:
        """Initialize both cameras
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check available cameras
            available_cameras = Picamera2.global_camera_info()
            self.logger.info(f"Available cameras: {len(available_cameras)}")
            
            for i, camera_info in enumerate(available_cameras):
                self.logger.info(f"Camera {i}: {camera_info}")
            
            if len(available_cameras) < 2:
                self.logger.warning(f"Only {len(available_cameras)} camera(s) detected")
            
            # Initialize cameras
            for camera_id in [0, 1]:
                camera_key = f"camera_{camera_id}"
                if camera_key in self.config and self.config[camera_key].get('enabled', False):
                    if self._initialize_camera(camera_id):
                        self.logger.info(f"Camera {camera_id} initialized successfully")
                    else:
                        self.logger.error(f"Failed to initialize camera {camera_id}")
                        return False
            
            if not self.cameras:
                self.logger.error("No cameras were initialized")
                return False
            
            self.logger.info("All cameras initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing cameras: {e}")
            return False
    
    def _initialize_camera(self, camera_id: int) -> bool:
        """Initialize a specific camera
        
        Args:
            camera_id: Camera index (0 or 1)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            camera_key = f"camera_{camera_id}"
            config = self.config[camera_key]
            
            # Create Picamera2 instance
            picam = Picamera2(camera_id)
            
            # Get camera configuration
            resolution = config.get('resolution', {'width': 1280, 'height': 720})
            framerate = config.get('framerate', 30)
            format_type = config.get('format', 'NV12')
            
            # Configure camera
            camera_config = picam.create_still_configuration(
                main={
                    "size": (resolution['width'], resolution['height']),
                    "format": format_type
                },
                buffer_count=4
            )
            picam.configure(camera_config)
            
            # Set camera controls
            controls_dict = {}
            
            # Auto exposure and white balance
            controls_dict[controls.AeEnable] = True
            controls_dict[controls.AwbEnable] = True
            
            # Frame rate
            controls_dict[controls.FrameRate] = framerate
            
            picam.set_controls(controls_dict)
            
            # Start camera
            picam.start()
            
            # Wait for camera to stabilize
            time.sleep(2)
            
            # Store camera instance
            self.cameras[camera_id] = picam
            
            self.logger.info(f"Camera {camera_id} configured: {resolution['width']}x{resolution['height']} @ {framerate}fps")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera {camera_id}: {e}")
            return False
    
    def start_capture(self) -> bool:
        """Start continuous capture from all cameras
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.running = True
            
            # Start capture thread for each camera
            for camera_id in self.cameras.keys():
                thread = threading.Thread(
                    target=self._capture_loop,
                    args=(camera_id,),
                    name=f"CameraCapture{camera_id}"
                )
                thread.daemon = True
                thread.start()
                self.camera_threads[camera_id] = thread
            
            self.logger.info("Camera capture started for all cameras")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting capture: {e}")
            return False
    
    def _capture_loop(self, camera_id: int):
        """Continuous capture loop for a specific camera
        
        Args:
            camera_id: Camera index
        """
        self.logger.info(f"Starting capture loop for camera {camera_id}")
        
        picam = self.cameras[camera_id]
        
        while self.running:
            try:
                # Capture frame
                frame = picam.capture_array()
                
                if frame is not None:
                    # Convert to BGR if necessary
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        # Already RGB, convert to BGR for OpenCV
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    elif len(frame.shape) == 3 and frame.shape[2] == 4:
                        # RGBA, convert to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif len(frame.shape) == 2:
                        # Grayscale, convert to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    # Store latest frame
                    with self.frame_locks[camera_id]:
                        self.latest_frames[camera_id] = frame.copy()
                
            except Exception as e:
                self.logger.error(f"Error in capture loop for camera {camera_id}: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"Capture loop stopped for camera {camera_id}")
    
    def capture_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """Capture a single frame from specified camera
        
        Args:
            camera_id: Camera index (0 or 1)
            
        Returns:
            np.ndarray: Frame data or None if error
        """
        try:
            if camera_id not in self.cameras:
                self.logger.error(f"Camera {camera_id} not initialized")
                return None
            
            # Return latest captured frame
            with self.frame_locks[camera_id]:
                if self.latest_frames[camera_id] is not None:
                    return self.latest_frames[camera_id].copy()
                else:
                    return None
            
        except Exception as e:
            self.logger.error(f"Error capturing frame from camera {camera_id}: {e}")
            return None
    
    def capture_frames_sync(self) -> Dict[int, Optional[np.ndarray]]:
        """Capture frames from all cameras synchronously
        
        Returns:
            Dict[int, np.ndarray]: Dictionary of camera_id -> frame
        """
        frames = {}
        
        for camera_id in self.cameras.keys():
            frames[camera_id] = self.capture_frame(camera_id)
        
        return frames
    
    def get_camera_info(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific camera
        
        Args:
            camera_id: Camera index
            
        Returns:
            Dict[str, Any]: Camera information or None if error
        """
        try:
            if camera_id not in self.cameras:
                return None
            
            picam = self.cameras[camera_id]
            camera_key = f"camera_{camera_id}"
            config = self.config[camera_key]
            
            info = {
                'camera_id': camera_id,
                'name': config.get('name', f'Camera {camera_id}'),
                'resolution': config.get('resolution'),
                'framerate': config.get('framerate'),
                'format': config.get('format'),
                'enabled': config.get('enabled', False),
                'properties': picam.camera_properties
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting camera {camera_id} info: {e}")
            return None
    
    def get_all_camera_info(self) -> Dict[int, Dict[str, Any]]:
        """Get information about all cameras
        
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary of camera_id -> info
        """
        info = {}
        
        for camera_id in self.cameras.keys():
            camera_info = self.get_camera_info(camera_id)
            if camera_info:
                info[camera_id] = camera_info
        
        return info
    
    def set_camera_control(self, camera_id: int, control_name: str, value: Any) -> bool:
        """Set a control parameter for a specific camera
        
        Args:
            camera_id: Camera index
            control_name: Name of the control
            value: Value to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if camera_id not in self.cameras:
                self.logger.error(f"Camera {camera_id} not initialized")
                return False
            
            picam = self.cameras[camera_id]
            
            # Map control names to libcamera controls
            control_map = {
                'brightness': controls.Brightness,
                'contrast': controls.Contrast,
                'saturation': controls.Saturation,
                'sharpness': controls.Sharpness,
                'exposure_time': controls.ExposureTime,
                'analogue_gain': controls.AnalogueGain,
                'digital_gain': controls.DigitalGain,
                'awb_enable': controls.AwbEnable,
                'ae_enable': controls.AeEnable
            }
            
            if control_name in control_map:
                picam.set_controls({control_map[control_name]: value})
                self.logger.info(f"Set {control_name} = {value} for camera {camera_id}")
                return True
            else:
                self.logger.error(f"Unknown control: {control_name}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error setting camera {camera_id} control {control_name}: {e}")
            return False
    
    def stop_capture(self):
        """Stop capture from all cameras"""
        self.running = False
        
        # Wait for capture threads to finish
        for camera_id, thread in self.camera_threads.items():
            thread.join(timeout=2.0)
            self.logger.info(f"Capture thread for camera {camera_id} stopped")
        
        self.camera_threads.clear()
    
    def cleanup(self):
        """Cleanup all camera resources"""
        self.logger.info("Cleaning up camera resources")
        
        # Stop capture if running
        if self.running:
            self.stop_capture()
        
        # Close all cameras
        for camera_id, picam in self.cameras.items():
            try:
                picam.stop()
                picam.close()
                self.logger.info(f"Camera {camera_id} closed")
            except Exception as e:
                self.logger.error(f"Error closing camera {camera_id}: {e}")
        
        self.cameras.clear()
        self.latest_frames = {0: None, 1: None}
        
        self.logger.info("Camera cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
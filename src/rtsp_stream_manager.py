"""
RTSP Stream Manager for Network Cameras

This module handles RTSP video streams from network cameras with automatic
reconnection, buffering, and quality optimization.

Author: AI Assistant
Date: 2025
"""

import cv2
import time
import threading
import queue
import logging
from typing import Dict, Optional, Any, Tuple
import numpy as np


logger = logging.getLogger(__name__)


class RTSPStreamManager:
    """Enhanced RTSP stream manager with robust error handling"""
    
    def __init__(self, camera_id: str, rtsp_config: Dict[str, Any]):
        """Initialize RTSP stream manager
        
        Args:
            camera_id: Unique camera identifier
            rtsp_config: RTSP camera configuration
        """
        self.camera_id = camera_id
        self.config = rtsp_config
        self.rtsp_url = rtsp_config['rtsp_url']
        self.name = rtsp_config.get('name', camera_id)
        
        # Stream objects
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=self.config['connection']['buffer_size'])
        self.stats_queue = queue.Queue(maxsize=100)
        
        # Threading
        self.stream_thread = None
        self.stats_thread = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Connection management
        self.connection_attempts = 0
        self.last_frame_time = 0
        self.total_frames = 0
        self.dropped_frames = 0
        
        # Quality metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Frame processing
        self.target_width = self.config['stream'].get('scale_to_width', 1280)
        self.target_height = self.config['stream'].get('scale_to_height', 720)
        self.maintain_aspect = self.config['stream'].get('maintain_aspect_ratio', True)
        
        logger.info(f"Initialized RTSP manager for {self.name} ({self.camera_id})")
    
    def connect(self) -> bool:
        """Establish connection to RTSP stream
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to {self.name}: {self.rtsp_url}")
            
            # Release existing capture if any
            if self.capture:
                self.capture.release()
            
            # Create new capture with RTSP URL
            self.capture = cv2.VideoCapture(self.rtsp_url)
            
            # Configure capture properties
            connection_config = self.config['connection']
            
            # Use TCP transport for reliability if configured
            if connection_config.get('tcp_transport', True):
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            
            # Set buffer size
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, connection_config['buffer_size'])
            
            # Set stream properties
            stream_config = self.config['stream']
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, stream_config['width'])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, stream_config['height'])
            self.capture.set(cv2.CAP_PROP_FPS, stream_config['fps'])
            
            # Additional OpenCV settings for RTSP
            self.capture.set(cv2.CAP_PROP_CONVERT_RGB, False)
            
            # Test connection by reading a frame
            ret, test_frame = self.capture.read()
            if not ret or test_frame is None:
                logger.error(f"Failed to read test frame from {self.name}")
                return False
            
            # Verify frame dimensions
            h, w = test_frame.shape[:2]
            logger.info(f"Connected to {self.name}: {w}x{h} @ {stream_config['fps']}fps")
            
            self.connection_attempts = 0
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {self.name}: {e}")
            return False
    
    def start(self) -> bool:
        """Start the RTSP stream threads
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning(f"Stream {self.name} is already running")
            return True
        
        # Connect to stream
        if not self.connect():
            logger.error(f"Failed to connect to {self.name}")
            return False
        
        # Start streaming
        self.is_running = True
        
        # Start stream reading thread
        self.stream_thread = threading.Thread(
            target=self._stream_reader_loop,
            name=f"RTSP-{self.camera_id}",
            daemon=True
        )
        self.stream_thread.start()
        
        # Start statistics thread
        self.stats_thread = threading.Thread(
            target=self._stats_loop,
            name=f"Stats-{self.camera_id}",
            daemon=True
        )
        self.stats_thread.start()
        
        logger.info(f"Started RTSP stream for {self.name}")
        return True
    
    def stop(self):
        """Stop the RTSP stream"""
        logger.info(f"Stopping RTSP stream for {self.name}")
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=3)
        
        if self.stats_thread and self.stats_thread.is_alive():
            self.stats_thread.join(timeout=1)
        
        # Release capture
        with self.lock:
            if self.capture:
                self.capture.release()
                self.capture = None
        
        # Clear queues
        self._clear_queue(self.frame_queue)
        self._clear_queue(self.stats_queue)
        
        logger.info(f"Stopped RTSP stream for {self.name}")
    
    def _clear_queue(self, q: queue.Queue):
        """Clear all items from a queue"""
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
    
    def _stream_reader_loop(self):
        """Main stream reading loop"""
        reconnect_interval = self.config['connection']['reconnect_interval']
        max_reconnect_attempts = self.config['connection']['max_reconnect_attempts']
        timeout = self.config['connection']['timeout']
        
        consecutive_failures = 0
        last_successful_read = time.time()
        
        while self.is_running:
            try:
                with self.lock:
                    if not self.capture:
                        break
                    
                    # Read frame with timeout handling
                    ret, frame = self.capture.read()
                
                current_time = time.time()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.debug(f"Failed to read frame from {self.name} (failure #{consecutive_failures})")
                    
                    # Check if we've been failing for too long
                    if current_time - last_successful_read > timeout:
                        logger.warning(f"Stream timeout for {self.name}, attempting reconnection")
                        
                        if self.connection_attempts < max_reconnect_attempts:
                            self.connection_attempts += 1
                            logger.info(f"Reconnecting to {self.name} (attempt {self.connection_attempts})")
                            
                            with self.lock:
                                if self.capture:
                                    self.capture.release()
                                    self.capture = None
                            
                            time.sleep(reconnect_interval)
                            
                            if self.connect():
                                consecutive_failures = 0
                                last_successful_read = current_time
                                continue
                        else:
                            logger.error(f"Max reconnection attempts reached for {self.name}")
                            break
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failure counters on successful read
                consecutive_failures = 0
                last_successful_read = current_time
                self.connection_attempts = 0
                self.total_frames += 1
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Add frame to queue with timestamp
                frame_data = {
                    'frame': processed_frame,
                    'original_frame': frame,
                    'timestamp': current_time,
                    'frame_number': self.total_frames,
                    'camera_id': self.camera_id
                }
                
                # Add to queue, dropping oldest frame if full
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                        self.dropped_frames += 1
                    
                    self.frame_queue.put(frame_data, timeout=0.1)
                    self.last_frame_time = current_time
                    
                except queue.Full:
                    self.dropped_frames += 1
                    logger.debug(f"Dropped frame for {self.name}")
                
                # Update FPS counter
                self.fps_counter += 1
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in stream reader loop for {self.name}: {e}")
                time.sleep(1)
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process and enhance frame quality
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        try:
            processed_frame = frame.copy()
            
            # Resize frame if configured
            if self.target_width and self.target_height:
                processed_frame = self._resize_frame(processed_frame)
            
            # Apply quality enhancements
            quality_config = self.config.get('quality', {})
            
            # Brightness adjustment
            brightness = quality_config.get('brightness_adjustment', 0)
            if brightness != 0:
                processed_frame = cv2.convertScaleAbs(processed_frame, alpha=1, beta=brightness)
            
            # Contrast adjustment
            contrast = quality_config.get('contrast_adjustment', 1.0)
            if contrast != 1.0:
                processed_frame = cv2.convertScaleAbs(processed_frame, alpha=contrast, beta=0)
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if quality_config.get('apply_clahe', False):
                processed_frame = self._apply_clahe(processed_frame)
            
            # Noise reduction
            if quality_config.get('denoise', False):
                processed_frame = cv2.fastNlMeansDenoisingColored(processed_frame)
            
            # Sharpening
            if quality_config.get('sharpen', False):
                processed_frame = self._apply_sharpening(processed_frame)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame for {self.name}: {e}")
            return frame
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame with aspect ratio preservation
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        if not self.maintain_aspect:
            return cv2.resize(frame, (self.target_width, self.target_height))
        
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        target_aspect = self.target_width / self.target_height
        
        if aspect_ratio > target_aspect:
            # Width is limiting factor
            new_width = self.target_width
            new_height = int(self.target_width / aspect_ratio)
        else:
            # Height is limiting factor
            new_height = self.target_height
            new_width = int(self.target_height * aspect_ratio)
        
        return cv2.resize(frame, (new_width, new_height))
    
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE for better contrast
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    def _apply_sharpening(self, frame: np.ndarray) -> np.ndarray:
        """Apply sharpening filter
        
        Args:
            frame: Input frame
            
        Returns:
            Sharpened frame
        """
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(frame, -1, kernel)
    
    def _stats_loop(self):
        """Statistics collection loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Calculate FPS
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                # Collect statistics
                stats = {
                    'timestamp': current_time,
                    'camera_id': self.camera_id,
                    'fps': self.current_fps,
                    'total_frames': self.total_frames,
                    'dropped_frames': self.dropped_frames,
                    'queue_size': self.frame_queue.qsize(),
                    'connection_attempts': self.connection_attempts,
                    'last_frame_age': current_time - self.last_frame_time if self.last_frame_time > 0 else 0
                }
                
                # Add to stats queue
                try:
                    if self.stats_queue.full():
                        self.stats_queue.get_nowait()
                    self.stats_queue.put(stats)
                except queue.Full:
                    pass
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in stats loop for {self.name}: {e}")
                time.sleep(1.0)
    
    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Get the latest frame from the stream
        
        Returns:
            Frame data dictionary or None if no frame available
        """
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_frame_blocking(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get frame with blocking wait
        
        Args:
            timeout: Maximum time to wait for frame
            
        Returns:
            Frame data dictionary or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get latest stream statistics
        
        Returns:
            Statistics dictionary or None
        """
        try:
            return self.stats_queue.get_nowait()
        except queue.Empty:
            return None
    
    def is_connected(self) -> bool:
        """Check if stream is connected and receiving frames
        
        Returns:
            True if connected and receiving frames
        """
        if not self.is_running:
            return False
        
        current_time = time.time()
        frame_age = current_time - self.last_frame_time if self.last_frame_time > 0 else float('inf')
        
        # Consider connected if we received a frame in the last 5 seconds
        return frame_age < 5.0
    
    def get_stream_info(self) -> Dict[str, Any]:
        """Get comprehensive stream information
        
        Returns:
            Stream information dictionary
        """
        return {
            'camera_id': self.camera_id,
            'name': self.name,
            'rtsp_url': self.rtsp_url,
            'is_running': self.is_running,
            'is_connected': self.is_connected(),
            'current_fps': self.current_fps,
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'drop_rate': self.dropped_frames / max(self.total_frames, 1),
            'connection_attempts': self.connection_attempts,
            'queue_size': self.frame_queue.qsize(),
            'queue_capacity': self.frame_queue.maxsize,
            'last_frame_age': time.time() - self.last_frame_time if self.last_frame_time > 0 else None
        }


class MultiRTSPManager:
    """Manager for multiple RTSP streams"""
    
    def __init__(self, cameras_config: Dict[str, Dict]):
        """Initialize multi-RTSP manager
        
        Args:
            cameras_config: Dictionary of camera configurations
        """
        self.cameras_config = cameras_config
        self.streams = {}
        self.is_running = False
        
        # Initialize stream managers
        for camera_id, config in cameras_config.items():
            if config.get('enabled', True):
                self.streams[camera_id] = RTSPStreamManager(camera_id, config)
                logger.info(f"Initialized stream manager for {camera_id}")
    
    def start_all(self) -> Dict[str, bool]:
        """Start all configured streams
        
        Returns:
            Dictionary of camera_id -> success status
        """
        results = {}
        
        for camera_id, stream in self.streams.items():
            success = stream.start()
            results[camera_id] = success
            if success:
                logger.info(f"Successfully started stream: {camera_id}")
            else:
                logger.error(f"Failed to start stream: {camera_id}")
        
        self.is_running = any(results.values())
        return results
    
    def stop_all(self):
        """Stop all streams"""
        logger.info("Stopping all RTSP streams")
        
        for stream in self.streams.values():
            stream.stop()
        
        self.is_running = False
        logger.info("All RTSP streams stopped")
    
    def get_all_latest_frames(self) -> Dict[str, Optional[Dict]]:
        """Get latest frames from all streams
        
        Returns:
            Dictionary of camera_id -> frame_data
        """
        frames = {}
        for camera_id, stream in self.streams.items():
            frames[camera_id] = stream.get_latest_frame()
        return frames
    
    def get_all_stats(self) -> Dict[str, Optional[Dict]]:
        """Get statistics from all streams
        
        Returns:
            Dictionary of camera_id -> stats
        """
        stats = {}
        for camera_id, stream in self.streams.items():
            stats[camera_id] = stream.get_stats()
        return stats
    
    def get_streams_info(self) -> Dict[str, Dict]:
        """Get information about all streams
        
        Returns:
            Dictionary of camera_id -> stream_info
        """
        info = {}
        for camera_id, stream in self.streams.items():
            info[camera_id] = stream.get_stream_info()
        return info
    
    def restart_stream(self, camera_id: str) -> bool:
        """Restart a specific stream
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            True if restart successful
        """
        if camera_id not in self.streams:
            logger.error(f"Stream not found: {camera_id}")
            return False
        
        logger.info(f"Restarting stream: {camera_id}")
        
        stream = self.streams[camera_id]
        stream.stop()
        time.sleep(1)
        return stream.start()
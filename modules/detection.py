"""
Face Detection Module using DeGirum SDK
Author: AI Assistant
Date: 2025
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional

try:
    import degirum as dg
except ImportError:
    print("DeGirum SDK not installed. Please install: pip install degirum")
    dg = None


class FaceDetector:
    """Face detection using DeGirum AI models"""
    
    def __init__(self, config):
        """Initialize face detector"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # DeGirum configuration
        self.device_config = config.get_degirum_device_config()
        self.detection_config = config.get_detection_config()
        
        # Model parameters
        self.model_name = self.detection_config.get('model_name')
        self.confidence_threshold = self.detection_config.get('confidence_threshold', 0.6)
        self.nms_threshold = self.detection_config.get('nms_threshold', 0.4)
        self.input_size = self.detection_config.get('input_size', [300, 300])
        
        # Initialize model
        self.model = None
        self._initialize_model()
        
        self.logger.info("Face detector initialized")
    
    def _initialize_model(self):
        """Initialize DeGirum face detection model"""
        try:
            if dg is None:
                raise ImportError("DeGirum SDK not available")
            
            # Connect to DeGirum device
            device_type = self.device_config.get('type', 'auto')
            device_id = self.device_config.get('device_id', 0)
            
            if device_type == 'auto':
                # Auto-detect best available device
                zoo = dg.connect_model_zoo()
            elif device_type == 'cpu':
                zoo = dg.connect_model_zoo(dg.hw.cpu)
            elif device_type == 'orca':
                zoo = dg.connect_model_zoo(f"orca{device_id}")
            else:
                # Default to auto
                zoo = dg.connect_model_zoo()
            
            # Load face detection model
            if self.model_name:
                self.model = zoo.load_model(self.model_name)
                self.logger.info(f"Loaded DeGirum model: {self.model_name}")
            else:
                # Use default face detection model
                available_models = [
                    "mobilenet_v2_ssd_coco--300x300_quant_n2x_orca1_1",
                    "yolo_v5s_face_detection--640x640_quant_n2x_orca1_1",
                    "retinaface_mobilenet_v1--1024x1024_quant_n2x_orca1_1"
                ]
                
                for model_name in available_models:
                    try:
                        self.model = zoo.load_model(model_name)
                        self.model_name = model_name
                        self.logger.info(f"Loaded default model: {model_name}")
                        break
                    except Exception as e:
                        self.logger.debug(f"Failed to load {model_name}: {e}")
                        continue
                
                if self.model is None:
                    raise Exception("No suitable face detection model found")
            
            # Set model parameters
            if hasattr(self.model, 'confidence_threshold'):
                self.model.confidence_threshold = self.confidence_threshold
            if hasattr(self.model, 'nms_threshold'):
                self.model.nms_threshold = self.nms_threshold
            
            self.logger.info("DeGirum face detection model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DeGirum model: {e}")
            # Fallback to OpenCV Haar Cascades
            self._initialize_opencv_fallback()
    
    def _initialize_opencv_fallback(self):
        """Initialize OpenCV Haar Cascade as fallback"""
        try:
            self.logger.warning("Using OpenCV Haar Cascade as fallback")
            self.opencv_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_opencv_fallback = True
            self.logger.info("OpenCV fallback initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenCV fallback: {e}")
            self.use_opencv_fallback = False
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in an image"""
        try:
            if self.model is not None:
                return self._detect_with_degirum(image)
            elif hasattr(self, 'use_opencv_fallback') and self.use_opencv_fallback:
                return self._detect_with_opencv(image)
            else:
                self.logger.error("No face detection method available")
                return []
            
        except Exception as e:
            self.logger.error(f"Error in face detection: {e}")
            return []
    
    def _detect_with_degirum(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using DeGirum model"""
        try:
            # Preprocess image for DeGirum
            processed_image = self._preprocess_image(image)
            
            # Run inference
            results = self.model(processed_image)
            
            # Parse results
            detections = []
            
            if hasattr(results, 'results'):
                # Handle different result formats
                for result in results.results:
                    if hasattr(result, 'bbox') and hasattr(result, 'score'):
                        # Standard detection format
                        bbox = result.bbox
                        confidence = result.score
                        
                        # Convert to normalized coordinates
                        h, w = image.shape[:2]
                        detection = {
                            'bbox': {
                                'x': bbox[0] / w,
                                'y': bbox[1] / h,
                                'width': (bbox[2] - bbox[0]) / w,
                                'height': (bbox[3] - bbox[1]) / h
                            },
                            'confidence': confidence,
                            'landmarks': getattr(result, 'landmarks', [])
                        }
                        
                        if confidence >= self.confidence_threshold:
                            detections.append(detection)
            
            elif isinstance(results, list):
                # Direct list of detections
                for result in results:
                    if isinstance(result, dict) and 'bbox' in result:
                        bbox = result['bbox']
                        confidence = result.get('confidence', result.get('score', 0))
                        
                        # Ensure normalized coordinates
                        if max(bbox) > 1.0:
                            # Convert from pixel to normalized coordinates
                            h, w = image.shape[:2]
                            bbox = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
                        
                        detection = {
                            'bbox': {
                                'x': bbox[0],
                                'y': bbox[1], 
                                'width': bbox[2] - bbox[0],
                                'height': bbox[3] - bbox[1]
                            },
                            'confidence': confidence,
                            'landmarks': result.get('landmarks', [])
                        }
                        
                        if confidence >= self.confidence_threshold:
                            detections.append(detection)
            
            # Apply Non-Maximum Suppression if needed
            if len(detections) > 1:
                detections = self._apply_nms(detections)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in DeGirum detection: {e}")
            return []
    
    def _detect_with_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV Haar Cascade"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.opencv_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Convert to normalized format
            detections = []
            h, w = image.shape[:2]
            
            for (x, y, width, height) in faces:
                detection = {
                    'bbox': {
                        'x': x / w,
                        'y': y / h,
                        'width': width / w,
                        'height': height / h
                    },
                    'confidence': 0.8,  # Default confidence for OpenCV
                    'landmarks': []
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in OpenCV detection: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for DeGirum model"""
        try:
            # Resize to model input size if specified
            if self.input_size and len(self.input_size) == 2:
                processed = cv2.resize(image, tuple(self.input_size))
            else:
                processed = image.copy()
            
            # Convert BGR to RGB if needed (DeGirum typically expects RGB)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return image
    
    def _apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        try:
            if len(detections) <= 1:
                return detections
            
            # Extract bboxes and scores
            boxes = []
            scores = []
            
            for detection in detections:
                bbox = detection['bbox']
                boxes.append([bbox['x'], bbox['y'], 
                             bbox['x'] + bbox['width'], bbox['y'] + bbox['height']])
                scores.append(detection['confidence'])
            
            boxes = np.array(boxes, dtype=np.float32)
            scores = np.array(scores, dtype=np.float32)
            
            # Apply OpenCV NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), 
                scores.tolist(), 
                self.confidence_threshold, 
                self.nms_threshold
            )
            
            # Return filtered detections
            if len(indices) > 0:
                if isinstance(indices[0], list):
                    indices = [i[0] for i in indices]
                return [detections[i] for i in indices]
            else:
                return []
            
        except Exception as e:
            self.logger.error(f"Error applying NMS: {e}")
            return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'input_size': self.input_size,
            'backend': 'degirum' if self.model else 'opencv'
        }
        
        if self.model and hasattr(self.model, 'model_info'):
            info.update(self.model.model_info)
        
        return info
    
    def set_confidence_threshold(self, threshold: float):
        """Set detection confidence threshold"""
        self.confidence_threshold = threshold
        if self.model and hasattr(self.model, 'confidence_threshold'):
            self.model.confidence_threshold = threshold
        self.logger.info(f"Confidence threshold set to {threshold}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.model:
                # DeGirum models are typically auto-managed
                self.model = None
            self.logger.info("Face detector cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up detector: {e}")
"""
Configuration management for Dual RTSP Camera Facial Recognition System
Author: AI Assistant
Date: 2025
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List


class Config:
    """Configuration manager for the facial recognition system"""
    
    def __init__(self, config_file: str = "config.yaml"):
        """Initialize configuration from file"""
        self.config_file = config_file
        self.config_data = self._load_config()
        
        # Parse configuration sections
        self.cameras = self.config_data.get('cameras', {})
        self.degirum = self.config_data.get('degirum', {})
        self.database = self.config_data.get('database', {})
        self.display = self.config_data.get('display', {})
        self.logging = self.config_data.get('logging', {})
        self.processing = self.config_data.get('processing', {})
        
        # Create necessary directories
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_file):
            # Create default configuration if file doesn't exist
            return self._create_default_config()
        
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except Exception as e:
            print(f"Error loading config file: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        default_config = {
            'cameras': {
                'camera_0': {
                    'enabled': True,
                    'name': 'Front Door Camera',
                    'rtsp_url': 'rtsp://admin:password@192.168.1.100:554/stream1',
                    'connection': {
                        'timeout': 10,
                        'reconnect_attempts': 5,
                        'buffer_size': 1
                    }
                },
                'camera_1': {
                    'enabled': True,
                    'name': 'Back Door Camera',
                    'rtsp_url': 'rtsp://admin:password@192.168.1.101:554/stream1',
                    'connection': {
                        'timeout': 10,
                        'reconnect_attempts': 5,
                        'buffer_size': 1
                    }
                }
            },
            'degirum': {
                'device': {
                    'type': 'auto',  # auto, cpu, gpu, orca
                    'device_id': 0
                },
                'face_detection': {
                    'model_name': 'mobilenet_v2_ssd_coco--300x300_quant_n2x_orca1_1',
                    'confidence_threshold': 0.6,
                    'nms_threshold': 0.4,
                    'input_size': [300, 300]
                },
                'face_recognition': {
                    'model_name': 'facenet_keras--160x160_quant_n2x_orca1_1',
                    'similarity_threshold': 0.7,
                    'input_size': [160, 160],
                    'embedding_size': 512
                }
            },
            'database': {
                'path': 'data/face_database.db',
                'embeddings_path': 'data/embeddings/',
                'max_faces_per_person': 10,
                'similarity_threshold': 0.7
            },
            'display': {
                'enabled': True,
                'window_size': {
                    'width': 1280,
                    'height': 720
                },
                'show_fps': True,
                'show_confidence': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/system.log',
                'max_size': 10485760,  # 10MB
                'backup_count': 5,
                'console_output': True
            },
            'processing': {
                'max_faces_per_frame': 10,
                'min_face_size': 40,
                'face_padding': 0.2,
                'quality_threshold': 0.6
            }
        }
        
        # Save default configuration
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config_data: Dict[str, Any]):
        """Save configuration to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def _create_directories(self):
        """Create necessary directories based on configuration"""
        directories = [
            'data',
            'logs',
            self.database.get('embeddings_path', 'data/embeddings'),
            os.path.dirname(self.logging.get('file', 'logs/system.log'))
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        
        # Update parsed sections
        self._update_parsed_sections()
    
    def _update_parsed_sections(self):
        """Update parsed configuration sections"""
        self.cameras = self.config_data.get('cameras', {})
        self.degirum = self.config_data.get('degirum', {})
        self.database = self.config_data.get('database', {})
        self.display = self.config_data.get('display', {})
        self.logging = self.config_data.get('logging', {})
        self.processing = self.config_data.get('processing', {})
    
    def save(self):
        """Save current configuration to file"""
        self._save_config(self.config_data)
    
    def reload(self):
        """Reload configuration from file"""
        self.config_data = self._load_config()
        self._update_parsed_sections()
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate cameras
        cameras = self.config_data.get('cameras', {})
        if not cameras:
            errors.append("No cameras configured")
        
        for camera_id, camera_config in cameras.items():
            if camera_config.get('enabled', False):
                if not camera_config.get('rtsp_url'):
                    errors.append(f"Camera {camera_id}: No RTSP URL configured")
                if not camera_config.get('name'):
                    errors.append(f"Camera {camera_id}: No name configured")
        
        # Validate DeGirum configuration
        degirum = self.config_data.get('degirum', {})
        if not degirum.get('face_detection', {}).get('model_name'):
            errors.append("DeGirum face detection model not configured")
        if not degirum.get('face_recognition', {}).get('model_name'):
            errors.append("DeGirum face recognition model not configured")
        
        # Validate database configuration
        database = self.config_data.get('database', {})
        if not database.get('path'):
            errors.append("Database path not configured")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def get_camera_config(self, camera_id: int) -> Dict[str, Any]:
        """Get configuration for a specific camera"""
        return self.cameras.get(f'camera_{camera_id}', {})
    
    def get_enabled_cameras(self) -> List[int]:
        """Get list of enabled camera IDs"""
        enabled = []
        for camera_id in [0, 1]:
            camera_config = self.get_camera_config(camera_id)
            if camera_config.get('enabled', False):
                enabled.append(camera_id)
        return enabled
    
    def update_camera_url(self, camera_id: int, rtsp_url: str):
        """Update RTSP URL for a camera"""
        self.set(f'cameras.camera_{camera_id}.rtsp_url', rtsp_url)
    
    def get_degirum_device_config(self) -> Dict[str, Any]:
        """Get DeGirum device configuration"""
        return self.degirum.get('device', {})
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get face detection configuration"""
        return self.degirum.get('face_detection', {})
    
    def get_recognition_config(self) -> Dict[str, Any]:
        """Get face recognition configuration"""
        return self.degirum.get('face_recognition', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.database
    
    def get_display_config(self) -> Dict[str, Any]:
        """Get display configuration"""
        return self.display
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.logging
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return self.processing
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return yaml.dump(self.config_data, default_flow_style=False, indent=2)
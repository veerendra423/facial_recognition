"""
Logging Module for Dual RTSP Camera Facial Recognition System
Author: AI Assistant
Date: 2025
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Dict, Any


def setup_logger(name: str, config: Dict[str, Any]) -> logging.Logger:
    """Setup logger with configuration"""
    
    # Get logger
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set log level
    log_level = getattr(logging, config.get('level', 'INFO').upper())
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if config.get('console_output', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    log_file = config.get('file', 'logs/system.log')
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Rotating file handler
        max_size = config.get('max_size', 10485760)  # 10MB
        backup_count = config.get('backup_count', 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    return logger


class SystemLogger:
    """System-wide logger for the facial recognition system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize system logger"""
        self.config = config
        self.loggers = {}
        
        # Setup main system logger
        self.main_logger = setup_logger("system", config)
        self.loggers["system"] = self.main_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger for a specific component"""
        if name not in self.loggers:
            self.loggers[name] = setup_logger(name, self.config)
        return self.loggers[name]
    
    def log_system_event(self, level: str, message: str, component: str = "system"):
        """Log a system event"""
        logger = self.get_logger(component)
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(message)
    
    def log_detection_event(self, camera_id: int, detections_count: int, recognition_count: int):
        """Log face detection event"""
        logger = self.get_logger("detection")
        logger.info(f"Camera {camera_id}: Detected {detections_count} faces, recognized {recognition_count}")
    
    def log_recognition_event(self, camera_id: int, person_id: str, confidence: float):
        """Log face recognition event"""
        logger = self.get_logger("recognition")
        logger.info(f"Camera {camera_id}: Recognized {person_id} (confidence: {confidence:.3f})")
    
    def log_enrollment_event(self, person_name: str, success: bool, method: str = "unknown"):
        """Log face enrollment event"""
        logger = self.get_logger("enrollment")
        status = "successful" if success else "failed"
        logger.info(f"Enrollment {status}: {person_name} via {method}")
    
    def log_camera_event(self, camera_id: int, event: str, details: str = ""):
        """Log camera-related event"""
        logger = self.get_logger("camera")
        message = f"Camera {camera_id}: {event}"
        if details:
            message += f" - {details}"
        logger.info(message)
    
    def log_error(self, component: str, error: str, exception: Exception = None):
        """Log error event"""
        logger = self.get_logger(component)
        if exception:
            logger.error(f"{error}: {str(exception)}")
        else:
            logger.error(error)
    
    def log_performance(self, component: str, metrics: Dict[str, Any]):
        """Log performance metrics"""
        logger = self.get_logger("performance")
        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        logger.info(f"{component} - {metrics_str}")
    
    def cleanup(self):
        """Cleanup all loggers"""
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize performance logger"""
        self.logger = setup_logger("performance", config)
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            self.metrics[operation] = duration
            del self.start_times[operation]
            return duration
        return 0.0
    
    def log_timing(self, operation: str, duration: float):
        """Log timing information"""
        self.logger.info(f"{operation}: {duration:.3f}s")
    
    def log_fps(self, component: str, fps: float):
        """Log FPS information"""
        self.logger.info(f"{component} FPS: {fps:.1f}")
    
    def log_memory_usage(self, component: str, memory_mb: float):
        """Log memory usage"""
        self.logger.info(f"{component} Memory: {memory_mb:.1f} MB")
    
    def log_model_performance(self, model: str, inference_time: float, throughput: float):
        """Log model performance metrics"""
        self.logger.info(f"{model} - Inference: {inference_time:.3f}s, Throughput: {throughput:.1f} fps")


class DatabaseLogger:
    """Logger for database operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize database logger"""
        self.logger = setup_logger("database", config)
    
    def log_query(self, query: str, duration: float = None):
        """Log database query"""
        message = f"Query: {query}"
        if duration is not None:
            message += f" ({duration:.3f}s)"
        self.logger.debug(message)
    
    def log_enrollment(self, person_name: str, embedding_count: int):
        """Log person enrollment"""
        self.logger.info(f"Enrolled {person_name} with {embedding_count} embeddings")
    
    def log_removal(self, person_name: str):
        """Log person removal"""
        self.logger.info(f"Removed {person_name} from database")
    
    def log_database_stats(self, total_persons: int, total_embeddings: int):
        """Log database statistics"""
        self.logger.info(f"Database stats - Persons: {total_persons}, Embeddings: {total_embeddings}")


def create_logger_config(log_level: str = "INFO", 
                        log_file: str = "logs/system.log",
                        console_output: bool = True) -> Dict[str, Any]:
    """Create a default logger configuration"""
    return {
        'level': log_level,
        'file': log_file,
        'max_size': 10485760,  # 10MB
        'backup_count': 5,
        'console_output': console_output
    }


def setup_system_logging(config: Dict[str, Any] = None) -> SystemLogger:
    """Setup system-wide logging"""
    if config is None:
        config = create_logger_config()
    
    return SystemLogger(config)
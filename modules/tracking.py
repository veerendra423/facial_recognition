"""
Face Tracking Module
Author: AI Assistant
Date: 2025
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import cv2


class FaceTrack:
    """Individual face track"""
    
    def __init__(self, track_id: int, detection: Dict[str, Any], person_id: Optional[str] = None):
        """Initialize face track"""
        self.track_id = track_id
        self.person_id = person_id
        self.detections = [detection]
        self.confidences = [detection['confidence']]
        self.timestamps = [time.time()]
        self.last_seen = time.time()
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        
        # Kalman filter for position prediction
        self.kf = self._init_kalman_filter(detection['bbox'])
        
        # Recognition consistency
        self.recognition_votes = defaultdict(int)
        if person_id:
            self.recognition_votes[person_id] = 1
    
    def _init_kalman_filter(self, bbox: Dict[str, float]):
        """Initialize Kalman filter for tracking"""
        # State: [x, y, width, height, dx, dy, dw, dh]
        kf = cv2.KalmanFilter(8, 4)
        
        # Transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Error covariance
        kf.errorCovPost = np.eye(8, dtype=np.float32)
        
        # Initialize state
        kf.statePre = np.array([
            bbox['x'], bbox['y'], bbox['width'], bbox['height'],
            0, 0, 0, 0
        ], dtype=np.float32)
        
        kf.statePost = kf.statePre.copy()
        
        return kf
    
    def predict(self) -> Dict[str, float]:
        """Predict next position"""
        prediction = self.kf.predict()
        
        # Convert back to bbox format
        bbox = {
            'x': float(prediction[0]),
            'y': float(prediction[1]),
            'width': float(prediction[2]),
            'height': float(prediction[3])
        }
        
        return bbox
    
    def update(self, detection: Dict[str, Any], person_id: Optional[str] = None):
        """Update track with new detection"""
        bbox = detection['bbox']
        
        # Update Kalman filter
        measurement = np.array([
            bbox['x'], bbox['y'], bbox['width'], bbox['height']
        ], dtype=np.float32)
        
        self.kf.correct(measurement)
        
        # Update track data
        self.detections.append(detection)
        self.confidences.append(detection['confidence'])
        self.timestamps.append(time.time())
        self.last_seen = time.time()
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
        # Update recognition votes
        if person_id:
            self.recognition_votes[person_id] += 1
        
        # Keep only recent detections (last 10)
        if len(self.detections) > 10:
            self.detections = self.detections[-10:]
            self.confidences = self.confidences[-10:]
            self.timestamps = self.timestamps[-10:]
    
    def get_current_bbox(self) -> Dict[str, float]:
        """Get current bounding box"""
        if self.detections:
            return self.detections[-1]['bbox']
        else:
            return self.predict()
    
    def get_stable_person_id(self) -> Optional[str]:
        """Get stable person ID based on votes"""
        if not self.recognition_votes:
            return None
        
        # Return person with most votes
        return max(self.recognition_votes.items(), key=lambda x: x[1])[0]
    
    def get_confidence(self) -> float:
        """Get average confidence"""
        return np.mean(self.confidences) if self.confidences else 0.0
    
    def mark_missed(self):
        """Mark track as missed in current frame"""
        self.time_since_update += 1
        self.hit_streak = 0
        self.age += 1


class FaceTracker:
    """Multi-object face tracker"""
    
    def __init__(self, config):
        """Initialize face tracker"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Tracking parameters
        self.max_disappeared = 30  # frames
        self.max_distance = 0.3  # normalized distance
        self.min_hits = 3  # minimum hits before considering stable
        
        # Track management
        self.tracks = {}  # camera_id -> {track_id: FaceTrack}
        self.next_track_id = 0
        
        self.logger.info("Face tracker initialized")
    
    def update_track(self, camera_id: int, detection: Dict[str, Any], person_id: Optional[str] = None) -> int:
        """Update tracking with new detection and return track ID"""
        try:
            if camera_id not in self.tracks:
                self.tracks[camera_id] = {}
            
            camera_tracks = self.tracks[camera_id]
            
            # Find best matching track
            best_track_id = self._find_best_match(camera_tracks, detection)
            
            if best_track_id is not None:
                # Update existing track
                camera_tracks[best_track_id].update(detection, person_id)
                track_id = best_track_id
            else:
                # Create new track
                track_id = self._create_new_track(camera_id, detection, person_id)
            
            # Clean up old tracks
            self._cleanup_tracks(camera_id)
            
            return track_id
            
        except Exception as e:
            self.logger.error(f"Error updating track: {e}")
            return -1
    
    def _find_best_match(self, camera_tracks: Dict[int, FaceTrack], detection: Dict[str, Any]) -> Optional[int]:
        """Find best matching track for detection"""
        try:
            bbox = detection['bbox']
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, track in camera_tracks.items():
                # Predict track position
                predicted_bbox = track.predict()
                
                # Calculate distance
                distance = self._calculate_distance(bbox, predicted_bbox)
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id
            
            return best_track_id
            
        except Exception as e:
            self.logger.error(f"Error finding best match: {e}")
            return None
    
    def _calculate_distance(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
        """Calculate distance between two bounding boxes"""
        try:
            # Calculate center points
            cx1 = bbox1['x'] + bbox1['width'] / 2
            cy1 = bbox1['y'] + bbox1['height'] / 2
            cx2 = bbox2['x'] + bbox2['width'] / 2
            cy2 = bbox2['y'] + bbox2['height'] / 2
            
            # Euclidean distance
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            return distance
            
        except Exception as e:
            self.logger.error(f"Error calculating distance: {e}")
            return float('inf')
    
    def _create_new_track(self, camera_id: int, detection: Dict[str, Any], person_id: Optional[str] = None) -> int:
        """Create new track"""
        try:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            track = FaceTrack(track_id, detection, person_id)
            self.tracks[camera_id][track_id] = track
            
            self.logger.debug(f"Created new track {track_id} for camera {camera_id}")
            return track_id
            
        except Exception as e:
            self.logger.error(f"Error creating new track: {e}")
            return -1
    
    def _cleanup_tracks(self, camera_id: int):
        """Remove old or stale tracks"""
        try:
            camera_tracks = self.tracks[camera_id]
            current_time = time.time()
            tracks_to_remove = []
            
            for track_id, track in camera_tracks.items():
                # Remove tracks that haven't been seen for too long
                if current_time - track.last_seen > self.max_disappeared:
                    tracks_to_remove.append(track_id)
                else:
                    # Mark as missed if not updated this frame
                    if track.time_since_update == 0:
                        track.mark_missed()
            
            # Remove stale tracks
            for track_id in tracks_to_remove:
                del camera_tracks[track_id]
                self.logger.debug(f"Removed stale track {track_id} from camera {camera_id}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up tracks: {e}")
    
    def update_tracks(self, camera_id: int, detections: List[Dict[str, Any]]) -> List[int]:
        """Update tracks with multiple detections"""
        try:
            track_ids = []
            
            for detection in detections:
                person_id = detection.get('person_id')
                track_id = self.update_track(camera_id, detection, person_id)
                track_ids.append(track_id)
            
            return track_ids
            
        except Exception as e:
            self.logger.error(f"Error updating tracks: {e}")
            return []
    
    def get_track_info(self, camera_id: int, track_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific track"""
        try:
            if camera_id in self.tracks and track_id in self.tracks[camera_id]:
                track = self.tracks[camera_id][track_id]
                
                return {
                    'track_id': track.track_id,
                    'person_id': track.get_stable_person_id(),
                    'bbox': track.get_current_bbox(),
                    'confidence': track.get_confidence(),
                    'hits': track.hits,
                    'age': track.age,
                    'last_seen': track.last_seen,
                    'hit_streak': track.hit_streak
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting track info: {e}")
            return None
    
    def get_active_tracks(self, camera_id: int) -> List[Dict[str, Any]]:
        """Get all active tracks for a camera"""
        try:
            if camera_id not in self.tracks:
                return []
            
            active_tracks = []
            for track_id, track in self.tracks[camera_id].items():
                # Only include tracks with minimum hits
                if track.hits >= self.min_hits:
                    track_info = self.get_track_info(camera_id, track_id)
                    if track_info:
                        active_tracks.append(track_info)
            
            return active_tracks
            
        except Exception as e:
            self.logger.error(f"Error getting active tracks: {e}")
            return []
    
    def get_track_count(self, camera_id: int) -> int:
        """Get number of active tracks for a camera"""
        return len(self.get_active_tracks(camera_id))
    
    def get_all_tracks_count(self) -> Dict[int, int]:
        """Get track counts for all cameras"""
        counts = {}
        for camera_id in self.tracks.keys():
            counts[camera_id] = self.get_track_count(camera_id)
        return counts
    
    def reset_tracks(self, camera_id: Optional[int] = None):
        """Reset tracks for a camera or all cameras"""
        try:
            if camera_id is not None:
                if camera_id in self.tracks:
                    self.tracks[camera_id].clear()
                    self.logger.info(f"Reset tracks for camera {camera_id}")
            else:
                self.tracks.clear()
                self.next_track_id = 0
                self.logger.info("Reset all tracks")
                
        except Exception as e:
            self.logger.error(f"Error resetting tracks: {e}")
    
    def get_tracker_stats(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        try:
            total_tracks = sum(len(camera_tracks) for camera_tracks in self.tracks.values())
            active_tracks = sum(self.get_track_count(cid) for cid in self.tracks.keys())
            
            stats = {
                'total_tracks': total_tracks,
                'active_tracks': active_tracks,
                'cameras': list(self.tracks.keys()),
                'next_track_id': self.next_track_id,
                'track_counts_per_camera': self.get_all_tracks_count()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting tracker stats: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup tracker resources"""
        try:
            self.tracks.clear()
            self.next_track_id = 0
            self.logger.info("Face tracker cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up tracker: {e}")
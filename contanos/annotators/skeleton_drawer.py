"""
Keypoints and Skeleton Visualization Module

Handles drawing of human pose keypoints and skeleton connections with
consistent color mapping and stateful tracking ID management.
"""

from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from contanos.visualizer.color_mapper import DeterministicColorMapper


class SkeletonDrawer:
    """Draws human pose keypoints and skeleton connections with ID-based coloring."""
    
    # COCO skeleton connections (zero-based indices)
    SKELETON_CONNECTIONS = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6)
    ]
    
    def __init__(self, 
                 keypoint_radius: int = 2,
                 skeleton_thickness: int = 1,
                 default_color: Tuple[int, int, int] = (255, 0, 255)):
        """
        Initialize keypoints skeleton visualizer.
        
        Args:
            keypoint_radius: Radius of keypoint circles
            skeleton_thickness: Thickness of skeleton lines
            default_color: Default BGR color for unknown IDs
        """
        self.keypoint_radius = keypoint_radius
        self.skeleton_thickness = skeleton_thickness
        self.default_color = default_color
        
        # Deterministic color mapper with caching
        self._color_mapper = DeterministicColorMapper(default_color)
        
    def _get_id_color(self, track_id: Optional[int]) -> Tuple[int, int, int]:
        """Get consistent BGR color for a tracking ID using deterministic hash mapping."""
        return self._color_mapper.get_color(track_id)
    
    def draw_keypoints(self,
                      frame: np.ndarray,
                      keypoints: List[List[Tuple[float, float]]],
                      track_ids: Optional[List[Optional[int]]] = None,
                      scale: float = 1.0,
                      draw_skeleton: bool = True) -> np.ndarray:
        """
        Draw keypoints and skeleton on frame.
        
        Args:
            frame: Input image frame (BGR)
            keypoints: List of person keypoints, each person is list of (x,y) tuples
            track_ids: Optional list of tracking IDs parallel to keypoints
            scale: Scale factor to apply to coordinates
            draw_skeleton: Whether to draw skeleton connections
            
        Returns:
            Frame with keypoints and skeleton drawn
        """
        if track_ids is None:
            track_ids = [None] * len(keypoints)
            
        # Ensure lists are same length
        min_len = min(len(keypoints), len(track_ids))
        keypoints = keypoints[:min_len]
        track_ids = track_ids[:min_len]
        
        for person_kpts, track_id in zip(keypoints, track_ids):
            if len(person_kpts) < 17:  # Skip malformed detections
                continue
                
            color = self._get_id_color(track_id)
            
            # Draw keypoints as circles
            for x, y in person_kpts:
                scaled_x, scaled_y = int(x / scale), int(y / scale)
                cv2.circle(frame, (scaled_x, scaled_y), 
                          self.keypoint_radius, color, -1)
            
            # Draw skeleton connections
            if draw_skeleton:
                self._draw_skeleton(frame, person_kpts, color, scale)
                
        return frame
    
    def _draw_skeleton(self,
                      frame: np.ndarray,
                      keypoints: List[Tuple[float, float]],
                      color: Tuple[int, int, int],
                      scale: float) -> None:
        """Draw skeleton connections for a single person."""
        for i1, i2 in self.SKELETON_CONNECTIONS:
            if i1 >= len(keypoints) or i2 >= len(keypoints):
                continue
                
            x1, y1 = keypoints[i1]
            x2, y2 = keypoints[i2]
            
            # Scale coordinates
            pt1 = (int(x1 / scale), int(y1 / scale))
            pt2 = (int(x2 / scale), int(y2 / scale))
            
            cv2.line(frame, pt1, pt2, color, self.skeleton_thickness)
    
    def get_ankle_positions(self,
                           keypoints: List[List[Tuple[float, float]]],
                           track_ids: Optional[List[Optional[int]]] = None,
                           scale: float = 1.0) -> Dict[int, Dict[str, Tuple[float, float]]]:
        """
        Extract ankle positions for trajectory tracking.
        
        Args:
            keypoints: List of person keypoints
            track_ids: Optional list of tracking IDs
            scale: Scale factor for coordinates
            
        Returns:
            Dict mapping track_id to {"left": (x,y), "right": (x,y)} ankle positions
        """
        ankle_positions = {}
        
        if track_ids is None:
            return ankle_positions
            
        for person_kpts, track_id in zip(keypoints, track_ids):
            if track_id is None or len(person_kpts) < 17:
                continue
                
            # COCO keypoint indices: 15=left_ankle, 16=right_ankle
            left_ankle = person_kpts[15]
            right_ankle = person_kpts[16]
            
            ankle_positions[track_id] = {
                "left": (left_ankle[0] / scale, left_ankle[1] / scale),
                "right": (right_ankle[0] / scale, right_ankle[1] / scale)
            }
            
        return ankle_positions
    
    def clear_id_colors(self) -> None:
        """Clear stored ID color mappings."""
        self._color_mapper.clear_cache()
        
    def get_color_for_id(self, track_id: Optional[int]) -> Tuple[int, int, int]:
        """Get the color associated with a tracking ID (for external use)."""
        return self._color_mapper.get_color(track_id)
    
    def set_color_for_id(self, track_id: int, color: Tuple[int, int, int]) -> None:
        """Manually set color for a specific tracking ID."""
        self._color_mapper.set_color(track_id, color)
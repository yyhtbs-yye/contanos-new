"""
Point Trajectories Visualization Module

Handles drawing and management of point trajectories with history tracking,
gap detection, and automatic cleanup of stale trajectories.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import cv2
import numpy as np
from contanos.visualizer.color_mapper import DeterministicColorMapper

class TrajectoryDrawer:
    """Manages and draws point trajectories with history and gap detection."""
    
    def __init__(self,
                 max_trajectory_length: int = 50,
                 gap_threshold: float = 75.0,
                 stale_frames: int = 5,
                 line_thickness: int = 1,
                 default_color: Tuple[int, int, int] = (255, 0, 255)):
        """
        Initialize trajectory visualizer.
        
        Args:
            max_trajectory_length: Maximum number of points to keep per trajectory
            gap_threshold: Pixel distance threshold for detecting trajectory gaps
            stale_frames: Number of frames after which to remove unused trajectories
            line_thickness: Thickness of trajectory lines
            default_color: Default BGR color for unknown IDs
        """
        self.max_trajectory_length = max_trajectory_length
        self.gap_threshold = gap_threshold
        self.stale_frames = stale_frames
        self.line_thickness = line_thickness
        self.default_color = default_color
        
        # Deterministic color mapper with caching
        self._color_mapper = DeterministicColorMapper(default_color)
        
        # Trajectory storage: track_id -> {"points": deque, "last_frame": int}
        self._trajectories: Dict[int, Dict] = defaultdict(
            lambda: {
                "points": deque(maxlen=self.max_trajectory_length),
                "last_frame": -1
            }
        )
        
    def _get_id_color(self, track_id: Optional[int]) -> Tuple[int, int, int]:
        """Get consistent BGR color for a tracking ID using deterministic hash mapping."""
        return self._color_mapper.get_color(track_id)
    
    def update_trajectories(self,
                           points: Dict[int, Tuple[float, float]],
                           frame_id: int) -> None:
        """
        Update trajectories with new points.
        
        Args:
            points: Dict mapping track_id to (x, y) coordinate
            frame_id: Current frame identifier
        """
        for track_id, (x, y) in points.items():
            traj = self._trajectories[track_id]
            point_deque = traj["points"]
            
            # Check for large gaps (potential tracking errors)
            if point_deque:
                last_x, last_y = point_deque[-1]
                distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
                
                if distance > self.gap_threshold:
                    # Large gap detected - reset trajectory
                    point_deque.clear()
            
            # Add new point
            point_deque.append((x, y))
            traj["last_frame"] = frame_id
    
    def update_multi_point_trajectories(self,
                                      multi_points: Dict[int, Dict[str, Tuple[float, float]]],
                                      frame_id: int) -> None:
        """
        Update trajectories for multiple points per track ID (e.g., left/right feet).
        
        Args:
            multi_points: Dict mapping track_id to dict of point_name -> (x, y)
            frame_id: Current frame identifier
        """
        for track_id, point_dict in multi_points.items():
            # Store multi-point trajectories with nested structure
            if track_id not in self._trajectories:
                self._trajectories[track_id] = {
                    "points": defaultdict(lambda: deque(maxlen=self.max_trajectory_length)),
                    "last_frame": -1
                }
            
            traj = self._trajectories[track_id]
            
            for point_name, (x, y) in point_dict.items():
                point_deque = traj["points"][point_name]
                
                # Check for large gaps
                if point_deque:
                    last_x, last_y = point_deque[-1]
                    distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
                    
                    if distance > self.gap_threshold:
                        point_deque.clear()
                
                point_deque.append((x, y))
            
            traj["last_frame"] = frame_id
    
    def purge_stale_trajectories(self, current_frame: int) -> None:
        """
        Remove trajectories that haven't been updated recently.
        
        Args:
            current_frame: Current frame identifier
        """
        to_remove = [
            track_id for track_id, traj in self._trajectories.items()
            if current_frame - traj["last_frame"] > self.stale_frames
        ]
        
        for track_id in to_remove:
            del self._trajectories[track_id]
            # Color mapping is handled by the mapper's cache
    
    def draw_trajectories(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all stored trajectories on the frame.
        
        Args:
            frame: Input image frame (BGR)
            
        Returns:
            Frame with trajectories drawn
        """
        for track_id, traj in self._trajectories.items():
            color = self._get_id_color(track_id)
            points = traj["points"]
            
            # Handle single point trajectories
            if isinstance(points, deque):
                self._draw_single_trajectory(frame, points, color)
            # Handle multi-point trajectories (e.g., feet)
            elif isinstance(points, defaultdict):
                for point_name, point_deque in points.items():
                    self._draw_single_trajectory(frame, point_deque, color)
                    
        return frame
    
    def draw_trajectories_for_ids(self,
                                frame: np.ndarray,
                                track_ids: List[int]) -> np.ndarray:
        """
        Draw trajectories only for specific track IDs.
        
        Args:
            frame: Input image frame (BGR)
            track_ids: List of track IDs to draw
            
        Returns:
            Frame with selected trajectories drawn
        """
        for track_id in track_ids:
            if track_id not in self._trajectories:
                continue
                
            color = self._get_id_color(track_id)
            traj = self._trajectories[track_id]
            points = traj["points"]
            
            if isinstance(points, deque):
                self._draw_single_trajectory(frame, points, color)
            elif isinstance(points, defaultdict):
                for point_name, point_deque in points.items():
                    self._draw_single_trajectory(frame, point_deque, color)
                    
        return frame
    
    def _draw_single_trajectory(self,
                              frame: np.ndarray,
                              points: deque,
                              color: Tuple[int, int, int]) -> None:
        """Draw a single trajectory as connected line segments."""
        if len(points) < 2:
            return
            
        # Convert to numpy array for OpenCV
        pts_array = np.array([(int(x), int(y)) for x, y in points], dtype=np.int32)
        
        # Draw polyline
        cv2.polylines(frame, [pts_array], 
                     isClosed=False, 
                     color=color, 
                     thickness=self.line_thickness)
    
    def get_trajectory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current trajectories.
        
        Returns:
            Dict with trajectory statistics
        """
        stats = {
            "total_trajectories": len(self._trajectories),
            "trajectory_lengths": {},
            "active_ids": list(self._trajectories.keys())
        }
        
        for track_id, traj in self._trajectories.items():
            points = traj["points"]
            if isinstance(points, deque):
                stats["trajectory_lengths"][track_id] = len(points)
            elif isinstance(points, defaultdict):
                stats["trajectory_lengths"][track_id] = {
                    name: len(pts) for name, pts in points.items()
                }
                
        return stats
    
    def clear_all_trajectories(self) -> None:
        """Clear all stored trajectories and colors."""
        self._trajectories.clear()
        self._color_mapper.clear_cache()
    
    def clear_trajectory(self, track_id: int) -> None:
        """Clear trajectory for a specific track ID."""
        self._trajectories.pop(track_id, None)
        # Note: We don't remove from color cache to maintain consistency
    
    def get_color_for_id(self, track_id: Optional[int]) -> Tuple[int, int, int]:
        """Get the color associated with a tracking ID (for external use)."""
        return self._color_mapper.get_color(track_id)
    
    def set_color_for_id(self, track_id: int, color: Tuple[int, int, int]) -> None:
        """Manually set color for a specific tracking ID."""
        self._color_mapper.set_color(track_id, color)
    
    def get_latest_points(self, track_id: int) -> Optional[Dict]:
        """
        Get the most recent points for a track ID.
        
        Args:
            track_id: Track ID to query
            
        Returns:
            Latest points dict or None if track not found
        """
        if track_id not in self._trajectories:
            return None
            
        traj = self._trajectories[track_id]
        points = traj["points"]
        
        if isinstance(points, deque) and points:
            return {"point": points[-1]}
        elif isinstance(points, defaultdict):
            return {name: pts[-1] if pts else None for name, pts in points.items()}
            
        return None
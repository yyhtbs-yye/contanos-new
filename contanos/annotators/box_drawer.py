"""
Bounding Boxes Visualization Module

Handles drawing of tracking bounding boxes with consistent ID-based coloring,
labels, and confidence scores.
"""

from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from contanos.visualizer.color_mapper import DeterministicColorMapper


class BoxDrawer:
    """Draws bounding boxes with tracking IDs and confidence scores."""
    
    def __init__(self,
                 box_thickness: int = 1,
                 font_scale: float = 0.5,
                 font_thickness: int = 1,
                 label_offset: int = 5,
                 default_color: Tuple[int, int, int] = (255, 0, 255)):
        """
        Initialize bounding box visualizer.
        
        Args:
            box_thickness: Thickness of bounding box lines
            font_scale: Scale factor for text labels
            font_thickness: Thickness of text labels
            label_offset: Vertical offset for labels above boxes
            default_color: Default BGR color for unknown IDs
        """
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.label_offset = label_offset
        self.default_color = default_color
        
        # Deterministic color mapper with caching
        self._color_mapper = DeterministicColorMapper(default_color)
        
        # Font for labels
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def _get_id_color(self, track_id: Optional[int]) -> Tuple[int, int, int]:
        """Get consistent BGR color for a tracking ID using deterministic hash mapping."""
        return self._color_mapper.get_color(track_id)
    
    def draw_boxes(self,
                   frame: np.ndarray,
                   bboxes: List[List[float]],
                   track_ids: Optional[List[Optional[int]]] = None,
                   scores: Optional[List[Optional[float]]] = None,
                   scale: float = 1.0,
                   draw_labels: bool = True) -> Dict[int, Tuple[int, int, int]]:
        """
        Draw bounding boxes on frame.
        
        Args:
            frame: Input image frame (BGR)
            bboxes: List of bounding boxes as [x1, y1, x2, y2]
            track_ids: Optional list of tracking IDs parallel to bboxes
            scores: Optional list of confidence scores parallel to bboxes
            scale: Scale factor to apply to coordinates
            draw_labels: Whether to draw ID and score labels
            
        Returns:
            Dictionary mapping track_id to color for external use
        """
        if not bboxes:
            return {}
            
        # Ensure all lists are same length
        num_boxes = len(bboxes)
        if track_ids is None:
            track_ids = [None] * num_boxes
        if scores is None:
            scores = [None] * num_boxes
            
        min_len = min(len(bboxes), len(track_ids), len(scores))
        bboxes = bboxes[:min_len]
        track_ids = track_ids[:min_len]
        scores = scores[:min_len]
        
        id_to_color = {}
        
        for bbox, track_id, score in zip(bboxes, track_ids, scores):
            if len(bbox) < 4:  # Skip malformed boxes
                continue
                
            # Scale and convert coordinates
            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            
            color = self._get_id_color(track_id)
            if track_id is not None:
                id_to_color[track_id] = color
            
            # Draw bounding box rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
            
            # Draw label if requested
            if draw_labels:
                self._draw_label(frame, x1, y1, track_id, score, color)
                
        return id_to_color
    
    def _draw_label(self,
                   frame: np.ndarray,
                   x: int,
                   y: int,
                   track_id: Optional[int],
                   score: Optional[float],
                   color: Tuple[int, int, int]) -> None:
        """Draw ID and score label above bounding box."""
        label_parts = []
        
        if track_id is not None:
            label_parts.append(f"ID:{track_id}")
            
        if score is not None:
            label_parts.append(f"{score:.2f}")
            
        if not label_parts:
            return
            
        label = " ".join(label_parts)
        label_y = max(y - self.label_offset, 10)  # Ensure label stays on screen
        
        cv2.putText(frame, label,
                   (x, label_y),
                   self.font,
                   self.font_scale,
                   color,
                   self.font_thickness,
                   cv2.LINE_AA)
    
    def draw_simple_boxes(self,
                         frame: np.ndarray,
                         bboxes: List[List[float]],
                         scale: float = 1.0,
                         color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Draw simple bounding boxes without IDs or scores.
        
        Args:
            frame: Input image frame (BGR)
            bboxes: List of bounding boxes as [x1, y1, x2, y2]
            scale: Scale factor to apply to coordinates
            color: Optional fixed color for all boxes
            
        Returns:
            Frame with boxes drawn
        """
        if color is None:
            color = self.default_color
            
        for bbox in bboxes:
            if len(bbox) < 4:
                continue
                
            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
            
        return frame
    
    def get_box_centers(self,
                       bboxes: List[List[float]],
                       track_ids: Optional[List[Optional[int]]] = None,
                       scale: float = 1.0) -> Dict[int, Tuple[float, float]]:
        """
        Get center points of bounding boxes.
        
        Args:
            bboxes: List of bounding boxes as [x1, y1, x2, y2]
            track_ids: Optional list of tracking IDs
            scale: Scale factor for coordinates
            
        Returns:
            Dict mapping track_id to (center_x, center_y)
        """
        centers = {}
        
        if track_ids is None:
            return centers
            
        for bbox, track_id in zip(bboxes, track_ids):
            if track_id is None or len(bbox) < 4:
                continue
                
            center_x = (bbox[0] + bbox[2]) / (2 * scale)
            center_y = (bbox[1] + bbox[3]) / (2 * scale)
            centers[track_id] = (center_x, center_y)
            
        return centers
    
    def clear_id_colors(self) -> None:
        """Clear stored ID color mappings."""
        self._color_mapper.clear_cache()
        
    def get_color_for_id(self, track_id: Optional[int]) -> Tuple[int, int, int]:
        """Get the color associated with a tracking ID (for external use)."""
        return self._color_mapper.get_color(track_id)
    
    def set_color_for_id(self, track_id: int, color: Tuple[int, int, int]) -> None:
        """Manually set color for a specific tracking ID."""
        self._color_mapper.set_color(track_id, color)
"""
Deterministic Color Mapping Module

Provides efficient hash-based color generation with caching for consistent
ID-to-color mapping across visualization components.
"""

from typing import Dict, List, Optional, Tuple
import hashlib


class DeterministicColorMapper:
    """Deterministic color mapper for consistent track ID coloring."""
    
    def __init__(self, default_color: Tuple[int, int, int] = (255, 0, 255)):
        """
        Initialize color mapper.
        
        Args:
            default_color: Default BGR color for None/invalid IDs
        """
        self.default_color = default_color
        self._color_cache: Dict[int, Tuple[int, int, int]] = {}
    
    def get_color(self, track_id: Optional[int]) -> Tuple[int, int, int]:
        """
        Get deterministic color for a track ID.
        
        Args:
            track_id: Tracking ID (None returns default color)
            
        Returns:
            BGR color tuple
        """
        if track_id is None:
            return self.default_color
            
        # Check cache first
        if track_id in self._color_cache:
            return self._color_cache[track_id]
        
        # Generate new color
        color = self._generate_hash_color(track_id)
        self._color_cache[track_id] = color
        return color
    
    def _generate_hash_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate deterministic color from track ID using hash."""
        # Convert ID to bytes and hash
        id_bytes = str(track_id).encode('utf-8')
        hash_digest = hashlib.md5(id_bytes).hexdigest()
        
        # Extract 3 color components from hash (use first 6 hex chars)
        r = int(hash_digest[0:2], 16)
        g = int(hash_digest[2:4], 16) 
        b = int(hash_digest[4:6], 16)
        
        # Ensure colors are not too dark (minimum brightness)
        min_brightness = 50
        r = max(r, min_brightness)
        g = max(g, min_brightness)
        b = max(b, min_brightness)
        
        # Return as BGR for OpenCV
        return (b, g, r)
    
    def set_color(self, track_id: int, color: Tuple[int, int, int]) -> None:
        """
        Manually set color for a specific tracking ID.
        
        Args:
            track_id: Tracking ID
            color: BGR color tuple
        """
        self._color_cache[track_id] = color
    
    def clear_cache(self) -> None:
        """Clear all cached colors."""
        self._color_cache.clear()
    
    def remove_color(self, track_id: int) -> None:
        """Remove specific ID from cache."""
        self._color_cache.pop(track_id, None)
    
    def get_cache_size(self) -> int:
        """Get number of cached colors."""
        return len(self._color_cache)
    
    def get_cached_ids(self) -> List[int]:
        """Get list of all cached track IDs."""
        return list(self._color_cache.keys())


# Alternative implementation using different hash algorithms
class AdvancedColorMapper(DeterministicColorMapper):
    """Enhanced color mapper with multiple hash strategies and color balancing."""
    
    def __init__(self, 
                 default_color: Tuple[int, int, int] = (255, 0, 255),
                 hash_algorithm: str = 'md5',
                 color_balance: bool = True):
        """
        Initialize advanced color mapper.
        
        Args:
            default_color: Default BGR color for None/invalid IDs
            hash_algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
            color_balance: Whether to apply color balancing for better distribution
        """
        super().__init__(default_color)
        self.hash_algorithm = hash_algorithm
        self.color_balance = color_balance
        
        # Validate hash algorithm
        if hash_algorithm not in ['md5', 'sha1', 'sha256']:
            raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
    
    def _generate_hash_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate color using specified hash algorithm with optional balancing."""
        # Convert ID to bytes
        id_bytes = str(track_id).encode('utf-8')
        
        # Generate hash using specified algorithm
        if self.hash_algorithm == 'md5':
            hash_digest = hashlib.md5(id_bytes).hexdigest()
        elif self.hash_algorithm == 'sha1':
            hash_digest = hashlib.sha1(id_bytes).hexdigest()
        elif self.hash_algorithm == 'sha256':
            hash_digest = hashlib.sha256(id_bytes).hexdigest()
        
        if self.color_balance:
            return self._balanced_color_from_hash(hash_digest, track_id)
        else:
            return self._simple_color_from_hash(hash_digest)
    
    def _simple_color_from_hash(self, hash_digest: str) -> Tuple[int, int, int]:
        """Extract color from hash digest (simple method)."""
        r = int(hash_digest[0:2], 16)
        g = int(hash_digest[2:4], 16)
        b = int(hash_digest[4:6], 16)
        
        # Ensure minimum brightness
        min_brightness = 50
        r = max(r, min_brightness)
        g = max(g, min_brightness)
        b = max(b, min_brightness)
        
        return (b, g, r)  # BGR for OpenCV
    
    def _balanced_color_from_hash(self, hash_digest: str, track_id: int) -> Tuple[int, int, int]:
        """Generate balanced color to avoid too similar colors."""
        # Use multiple parts of hash for better distribution
        r = int(hash_digest[0:2], 16)
        g = int(hash_digest[8:10], 16)  # Skip some chars for better distribution
        b = int(hash_digest[16:18], 16)
        
        # Apply golden ratio-based spacing for better color distribution
        golden_ratio = 0.618033988749
        hue_offset = (track_id * golden_ratio) % 1.0
        
        # Adjust colors based on hue offset
        r = int((r + hue_offset * 255) % 256)
        g = int((g + hue_offset * 128) % 256)
        b = int((b + hue_offset * 64) % 256)
        
        # Ensure good contrast and brightness
        min_val = 60
        max_val = 220
        r = min_val + (r % (max_val - min_val))
        g = min_val + (g % (max_val - min_val))
        b = min_val + (b % (max_val - min_val))
        
        return (b, g, r)  # BGR for OpenCV


# Singleton pattern for global color consistency
class GlobalColorMapper:
    """Singleton color mapper for consistent colors across all visualization components."""
    
    _instance = None
    _mapper = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._mapper = DeterministicColorMapper()
        return cls._instance
    
    def get_color(self, track_id: Optional[int]) -> Tuple[int, int, int]:
        """Get color using global mapper."""
        return self._mapper.get_color(track_id)
    
    def set_color(self, track_id: int, color: Tuple[int, int, int]) -> None:
        """Set color using global mapper."""
        self._mapper.set_color(track_id, color)
    
    def clear_cache(self) -> None:
        """Clear global color cache."""
        self._mapper.clear_cache()
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
        cls._mapper = None


# Convenience functions for easy integration
def get_deterministic_color(track_id: Optional[int], 
                          default_color: Tuple[int, int, int] = (255, 0, 255)) -> Tuple[int, int, int]:
    """
    Standalone function to get deterministic color for a track ID.
    
    Args:
        track_id: Tracking ID (None returns default color)
        default_color: Default BGR color for None IDs
        
    Returns:
        BGR color tuple
    """
    if track_id is None:
        return default_color
    
    # Simple hash-based color generation
    id_bytes = str(track_id).encode('utf-8')
    hash_digest = hashlib.md5(id_bytes).hexdigest()
    
    r = max(int(hash_digest[0:2], 16), 50)
    g = max(int(hash_digest[2:4], 16), 50)
    b = max(int(hash_digest[4:6], 16), 50)
    
    return (b, g, r)  # BGR for OpenCV


def create_color_mapper(advanced: bool = False, **kwargs) -> DeterministicColorMapper:
    """
    Factory function to create appropriate color mapper.
    
    Args:
        advanced: Whether to use advanced color mapper
        **kwargs: Additional arguments for mapper initialization
        
    Returns:
        Color mapper instance
    """
    if advanced:
        return AdvancedColorMapper(**kwargs)
    else:
        return DeterministicColorMapper(**kwargs)
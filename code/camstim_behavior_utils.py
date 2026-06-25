from sync_dataset import Sync
import numpy as np
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

# Configuration constants
@dataclass
class MonitorDelayConfig:
    """Configuration parameters for monitor delay calculation."""
    ASSUMED_DELAY: float = 0.0356
    DELAY_THRESHOLD: float = 0.002
    PHOTODIODE_ERROR_THRESHOLD: float = 1.8
    VSYNC_EVENTS_PER_PHOTODIODE: int = 120  # 60 * 2
    HALF_VSYNC_EVENTS: int = 60
    
    # Photodiode rise time thresholds (seconds)
    SHORT_RISE_MIN: float = 0.1
    SHORT_RISE_MAX: float = 0.3
    MEDIUM_RISE_MIN: float = 0.5
    MEDIUM_RISE_MAX: float = 1.5
    LARGE_RISE_MIN: float = 1.9
    LARGE_RISE_MAX: float = 2.1
    
    # Pattern detection
    START_PATTERN_OFFSET: int = 2
    END_PATTERN_OFFSET: int = 3


class PhotodiodeProcessor:
    """Handles photodiode signal processing and error correction."""
    
    def __init__(self, config: MonitorDelayConfig = None):
        self.config = config or MonitorDelayConfig()
        
    def get_photodiode_times(self, sync: Sync) -> np.ndarray:
        """Get photodiode rising edge times in seconds."""
        try:
            sample_freq = sync.sample_freq
            photodiode_rise = sync.get_rising_edges("stim_photodiode")
            return np.array(photodiode_rise, dtype=np.float64) / sample_freq
        except Exception as e:
            logging.warning(f"Failed to get photodiode times: {e}")
            return np.array([])
    
    def find_stimulus_boundaries(self, photodiode_times: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """
        Find the start and end indices of the main stimulus period.
        
        Parameters
        ----------
        photodiode_times : np.ndarray
            Array of photodiode rising edge times in seconds
            
        Returns
        -------
        Tuple[Optional[int], Optional[int]]
            Start and end indices, or (None, None) if not found
        """
        if len(photodiode_times) < 4:
            return None, None
            
        # Calculate time differences between consecutive photodiode events
        time_diffs = np.diff(photodiode_times)
        
        # Classify time differences into categories
        short_mask = (time_diffs >= self.config.SHORT_RISE_MIN) & (time_diffs <= self.config.SHORT_RISE_MAX)
        medium_mask = (time_diffs >= self.config.MEDIUM_RISE_MIN) & (time_diffs <= self.config.MEDIUM_RISE_MAX)
        large_mask = (time_diffs >= self.config.LARGE_RISE_MIN) & (time_diffs <= self.config.LARGE_RISE_MAX)
        
        short_indices = np.where(short_mask)[0]
        medium_indices = np.where(medium_mask)[0]
        large_indices = np.where(large_mask)[0]
        
        return self._detect_stimulus_pattern(short_indices, medium_indices, large_indices)
    
    def _detect_stimulus_pattern(self, short_indices: np.ndarray, medium_indices: np.ndarray, 
                                large_indices: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """Detect stimulus start/end pattern from photodiode timing."""
        short_set = set(short_indices)
        
        # Try medium indices first (preferred)
        if len(medium_indices) >= 3:
            return self._find_pattern_boundaries(medium_indices, short_set)
        
        # Fall back to large indices if medium not sufficient
        if len(large_indices) > 0:
            return self._find_pattern_boundaries(large_indices, short_set)
            
        return None, None
    
    def _find_pattern_boundaries(self, target_indices: np.ndarray, 
                                short_set: set) -> Tuple[Optional[int], Optional[int]]:
        """Find start and end boundaries based on pattern detection."""
        ptd_start = None
        ptd_end = None
        
        for idx in target_indices:
            # Check for start pattern: short events before this index
            start_range = set(range(
                max(0, idx - self.config.START_PATTERN_OFFSET), 
                idx
            ))
            if start_range.issubset(short_set):
                ptd_start = idx + 1
            
            # Check for end pattern: short events after this index
            end_range = set(range(
                idx + 1,
                idx + self.config.END_PATTERN_OFFSET + 1
            ))
            if end_range.issubset(short_set):
                ptd_end = idx
                
        return ptd_start, ptd_end
    
    def correct_photodiode_errors(self, photodiode_times: np.ndarray, 
                                 ptd_start: int, ptd_end: int) -> Tuple[np.ndarray, int]:
        """
        Remove photodiode timing errors (consecutive events too close together).
        
        Parameters
        ----------
        photodiode_times : np.ndarray
            Array of photodiode times
        ptd_start : int
            Start index of stimulus period
        ptd_end : int
            End index of stimulus period
            
        Returns
        -------
        Tuple[np.ndarray, int]
            Corrected photodiode times and new end index
        """
        corrected_times = photodiode_times.copy()
        current_end = ptd_end
        
        while True:
            time_diffs = np.diff(corrected_times)
            
            # Check for errors in the stimulus period
            stimulus_diffs = time_diffs[ptd_start:current_end]
            error_mask = stimulus_diffs < self.config.PHOTODIODE_ERROR_THRESHOLD
            
            if not np.any(error_mask):
                break
                
            # Find and remove the first error
            error_indices = np.where(error_mask)[0] + ptd_start
            if len(error_indices) > 0:
                corrected_times = np.delete(corrected_times, error_indices[-1])
                current_end -= 1
            else:
                break
                
        return corrected_times, current_end


class MonitorDelayCalculator:
    """Calculates monitor delay from photodiode and vsync signals."""
    
    def __init__(self, config: MonitorDelayConfig = None):
        self.config = config or MonitorDelayConfig()
        self.photodiode_processor = PhotodiodeProcessor(config)
        
    def calculate_monitor_delay(self, sync: Sync) -> float:
        """
        Calculate the monitor delay from sync data.
        
        Parameters
        ----------
        sync : Sync
            Sync object containing photodiode and vsync data
            
        Returns
        -------
        float
            Calculated monitor delay in seconds
        """
        logging.info("Calculating monitor delay")
        
        try:
            # Get photodiode and vsync data
            photodiode_times = self.photodiode_processor.get_photodiode_times(sync)
            vsync_fall_times = sync.get_falling_edges("vsync_stim", units="seconds")
            
            if len(photodiode_times) == 0:
                logging.warning("No photodiode signal found, using assumed delay")
                return self.config.ASSUMED_DELAY
                
            # Find stimulus boundaries
            ptd_start, ptd_end = self.photodiode_processor.find_stimulus_boundaries(photodiode_times)
            
            if ptd_start is None or ptd_end is None:
                logging.warning("Could not find stimulus boundaries, using assumed delay")
                return self.config.ASSUMED_DELAY
                
            # Correct photodiode errors
            corrected_times, corrected_end = self.photodiode_processor.correct_photodiode_errors(
                photodiode_times, ptd_start, ptd_end
            )
            
            # Calculate delay from corrected data
            delay = self._calculate_delay_from_signals(
                corrected_times, vsync_fall_times, ptd_start, corrected_end
            )
            
            return self._validate_delay(delay)
            
        except Exception as e:
            logging.error(f"Error calculating monitor delay: {e}")
            return self.config.ASSUMED_DELAY
    
    def _calculate_delay_from_signals(self, photodiode_times: np.ndarray, 
                                    vsync_fall_times: np.ndarray,
                                    ptd_start: int, ptd_end: int) -> float:
        """Calculate delay from photodiode and vsync signals."""
        num_photodiode_events = ptd_end - ptd_start
        
        if num_photodiode_events <= 0:
            raise ValueError("No valid photodiode events in stimulus period")
            
        # Calculate delays for each photodiode event
        delays = []
        
        for i in range(num_photodiode_events):
            photodiode_idx = ptd_start + i
            
            # Calculate corresponding vsync index
            vsync_idx = (i * self.config.VSYNC_EVENTS_PER_PHOTODIODE + 
                        self.config.HALF_VSYNC_EVENTS)
            
            if (photodiode_idx < len(photodiode_times) and 
                vsync_idx < len(vsync_fall_times)):
                
                delay = photodiode_times[photodiode_idx] - vsync_fall_times[vsync_idx]
                delays.append(delay)
        
        if not delays:
            raise ValueError("No valid delay measurements found")
            
        delays = np.array(delays)
        
        # Use all but the last delay measurement for stability
        if len(delays) > 1:
            delays = delays[:-1]
            
        return np.mean(delays)
    
    def _validate_delay(self, delay: float) -> float:
        """Validate and potentially correct the calculated delay."""
        delays_array = np.array([delay]) if np.isscalar(delay) else delay
        
        if len(delays_array) > 1:
            delay_std = np.std(delays_array)
            delay_mean = np.mean(delays_array)
        else:
            delay_std = 0
            delay_mean = delays_array[0] if len(delays_array) > 0 else self.config.ASSUMED_DELAY
        
        # Check if delay is valid
        if delay_std > self.config.DELAY_THRESHOLD or np.isnan(delay_mean):
            # Check for one-second offset error
            corrected_delay = delay_mean - 1.0
            if np.abs(corrected_delay - self.config.ASSUMED_DELAY) < self.config.DELAY_THRESHOLD:
                logging.info("Detected one-second offset, correcting delay")
                return corrected_delay
            
            logging.warning(f"Delay validation failed (std: {delay_std:.4f}), using assumed delay")
            return self.config.ASSUMED_DELAY
            
        return delay_mean


def extract_frame_times_with_delay(sync: Sync, 
                                  config: MonitorDelayConfig = None) -> float:
    """
    Extract frame times from vsync signal and calculate monitor delay.
    
    This is the main entry point that maintains compatibility with the original API
    while using the optimized implementation.
    
    Parameters
    ----------
    sync : Sync
        Sync object containing sync data
    config : MonitorDelayConfig, optional
        Configuration parameters for delay calculation
        
    Returns
    -------
    float
        Calculated monitor delay in seconds
    """
    calculator = MonitorDelayCalculator(config)
    return calculator.calculate_monitor_delay(sync)


def get_corrected_frame_times(sync: Sync, 
                             frame_line: str = "vsync_stim",
                             edge_type: str = "falling",
                             config: MonitorDelayConfig = None) -> np.ndarray:
    """
    Get frame times corrected for monitor delay.
    
    Parameters
    ----------
    sync : Sync
        Sync object containing sync data
    frame_line : str
        Line name for frame timing signal
    edge_type : str
        Edge type to use ("rising" or "falling")
    config : MonitorDelayConfig, optional
        Configuration parameters
        
    Returns
    -------
    np.ndarray
        Frame times corrected for monitor delay
    """
    # Calculate monitor delay
    delay = extract_frame_times_with_delay(sync, config=config)
    
    # Get frame times
    if edge_type.lower() == "falling":
        frame_times = sync.get_falling_edges(frame_line, units="seconds")
    else:
        frame_times = sync.get_rising_edges(frame_line, units="seconds")
    
    # Apply delay correction
    corrected_times = frame_times + delay
    
    logging.info(f"Applied monitor delay correction: {delay:.4f} seconds to {len(frame_times)} frames")
    
    return corrected_times


# Maintain backward compatibility
def calculate_frame_mean_time(photo_diode_rising_edges: np.ndarray, 
                             sample_freq: float) -> Tuple[Optional[int], Optional[int]]:
    """
    Legacy function maintained for backward compatibility.
    
    Use PhotodiodeProcessor.find_stimulus_boundaries() for new code.
    """
    # Convert to seconds if needed
    if np.max(photo_diode_rising_edges) > 1000:  # Assume samples if large values
        photodiode_times = photo_diode_rising_edges / sample_freq
    else:
        photodiode_times = photo_diode_rising_edges
    
    processor = PhotodiodeProcessor()
    return processor.find_stimulus_boundaries(photodiode_times)

def get_draw_epochs(
    draw_log: List[int], start_frame: int, stop_frame: int
) -> List[Tuple[int, int]]:
    """
    Gets the frame numbers of the active frames within a stimulus window.
    Stimulus epochs come in the form [0, 0, 1, 1, 0, 0] where the stimulus is
    active for some amount of time in the window indicated by int 1 at that
    frame. This function returns the ranges for which the set_log is 1 within
    the draw_log window.
    Parameters
    ----------
    draw_log: List[int]
        A list of ints indicating for what frames stimuli were active
    start_frame: int
        The start frame to search within the draw_log for active values
    stop_frame: int
        The end frame to search within the draw_log for active values

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples indicating the start and end frames of every
        contiguous set of active values within the specified window
        of the draw log.
    """
    draw_epochs = []
    current_frame = start_frame

    while current_frame <= stop_frame:
        epoch_length = 0
        while current_frame < stop_frame and draw_log[current_frame] == 1:
            epoch_length += 1
            current_frame += 1
        else:
            current_frame += 1

        if epoch_length:
            draw_epochs.append(
                (
                    current_frame - epoch_length - 1,
                    current_frame - 1,
                )
            )

    return draw_epochs
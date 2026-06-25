"""
Lazy Pickle Loader Library for Camstim Behavioral and Foraging Data

This library provides lazy loading capabilities for pickle files containing
behavioral and foraging experiment data, allowing selective access to parameters
without loading the entire file into memory.
"""

import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Set
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LazyPickleLoader(ABC):
    """
    Abstract base class for lazy loading pickle files.
    
    This class provides the foundation for lazy loading of large pickle files,
    caching accessed data and providing a common interface for different
    pickle file types.
    """
    
    def __init__(self, file_path: Union[str, Path], encoding: str = "latin1"):
        """
        Initialize the lazy loader.
        
        Args:
            file_path: Path to the pickle file
            encoding: Encoding to use when loading pickle (default: latin1)
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self._raw_data = None
        self._cache = {}
        self._loaded_keys = set()
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {self.file_path}")
    
    @property
    def raw_data(self) -> Dict[str, Any]:
        """Lazy load the raw pickle data."""
        if self._raw_data is None:
            logger.info(f"Loading pickle file: {self.file_path}")
            with open(self.file_path, "rb") as f:
                self._raw_data = pickle.load(f, encoding=self.encoding)
        return self._raw_data
    
    def get_top_level_keys(self) -> List[str]:
        """Get all top-level keys in the pickle file."""
        return list(self.raw_data.keys())
    
    def get_nested_keys(self, path: str) -> List[str]:
        """
        Get keys from a nested dictionary.
        
        Args:
            path: Dot-separated path to the nested dict (e.g., 'items.behavior')
        
        Returns:
            List of keys in the nested dictionary
        """
        try:
            obj = self._navigate_to_path(path)
            if isinstance(obj, dict):
                return list(obj.keys())
            else:
                return []
        except (KeyError, TypeError, AttributeError):
            return []
    
    def _navigate_to_path(self, path: str) -> Any:
        """Navigate to a specific path in the data structure."""
        obj = self.raw_data
        for key in path.split('.'):
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            elif isinstance(obj, list) and key.isdigit():
                obj = obj[int(key)]
            else:
                raise KeyError(f"Path '{path}' not found")
        return obj
    
    def get_parameter(self, path: str, default: Any = None) -> Any:
        """
        Get a specific parameter using dot notation.
        
        Args:
            path: Dot-separated path to the parameter (e.g., 'items.behavior.lick_sensors.0.lick_events')
            default: Default value if parameter not found
            
        Returns:
            The parameter value or default if not found
        """
        if path in self._cache:
            return self._cache[path]
        
        try:
            value = self._navigate_to_path(path)
            self._cache[path] = value
            self._loaded_keys.add(path)
            return value
        except (KeyError, TypeError, AttributeError):
            logger.warning(f"Parameter '{path}' not found, returning default")
            return default
    
    def get_parameters(self, paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Get multiple parameters at once.
        
        Args:
            paths: Dict of id keys and dot-separated path values
            
        Returns:
            Dictionary mapping paths to their values
        """
        result = {}
        for name, path in paths.items():
            result[name] = self.get_parameter(path)
        return result
    
    def clear_cache(self):
        """Clear the parameter cache."""
        self._cache.clear()
        self._loaded_keys.clear()
    
    def get_loaded_keys(self) -> Set[str]:
        """Get the set of keys that have been loaded."""
        return self._loaded_keys.copy()
    
    @abstractmethod
    def get_experiment_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata in a standardized format."""
        pass
    
    @abstractmethod
    def get_timing_data(self) -> Dict[str, Any]:
        """Get timing-related data in a standardized format."""
        pass


class BehaviorPickleLoader(LazyPickleLoader):
    """
    Specialized loader for behavior pickle files.
    
    This loader understands the structure of behavioral experiment pickle files
    and provides convenient access to commonly used parameters.
    """
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all parameters in the pickle file."""
        return self.raw_data
        
    def get_experiment_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata in standardized format."""
        metadata_paths = dict(
            stage='items.behavior.params.stage',
            task_id='items.behavior.params.task_id',
            change_time_distribution='items.behavior.cl_params.change_time_dist',
            change_flashes_min='items.behavior.cl_params.change_flashes_min',
            end_after_response_seconds='items.behavior.cl_params.end_after_response_sec',
            failure_repeats='items.behavior.cl_params.failure_repeats',
            response_window='items.behavior.params.response_window',
            auto_reward_delay='items.behavior.cl_params.auto_reward_delay',
            free_reward_trials='items.behavior.cl_params.free_reward_trials',
            end_after_response='items.behavior.cl_params.end_after_response',
            stimulus_window='items.behavior.params.stimulus_window',
            change_flashes_max='items.behavior.cl_params.change_flashes_max',
            change_time_dist='items.behavior.cl_params.change_time_dist',
            flash_omit_probability='items.behavior.cl_params.flash_omit_probability',
            catch_frequency='items.behavior.cl_params.catch_frequency',
            min_no_lick_time='items.behavior.cl_params.min_no_lick_time',
            max_task_duration_min='items.behavior.cl_params.max_task_duration_min',
            start_stop_padding='itmes.behavior.cl_params.start_stop_padding',
            periodic_flash='items.behavior.cl_params.periodic_flash',
            timeout_duration='items.behavior.cl_params.timeout_duration',
            epilogue='items.behavior.cl_params.epilogue',
            warm_up_trials='items.behavior.cl_params.warm_up_trials',
            pre_change_time='items.behavior.cl_params.pre_change_time'
        )
        return self.get_parameters(metadata_paths)
        
    
    def get_timing_data(self) -> Dict[str, Any]:
        """Get timing-related data."""
        timing_paths = dict(
            start_time='start_time',
            stop_time='stop_time',
            intervals_ms='items.behavior.intervalsms',
            reward_times='items.behavior.rewards.0.reward_times',
            lick_times='items.behavior.lick_sensors.0.lick_events'
        )
        return self.get_parameters(timing_paths)
    
    def get_encoder_data(self) -> Dict[str, Any]:
        """Get all lick-related data."""
        encoder_data = dict(
            encoder_disctance_cm='items.behavior.encoders.0.distance',
            encoder_distance_degrees='items.behavior.encoders.0.degrees',
            encoder_delta_degree_change='items.behavior.encoders.0.dx',
            encoder_counts='items.behavior.encoders.0.counts',
        )
        return self.get_parameters(encoder_data)
    
    def get_reward_metadata(self) -> Dict[str, Any]:
        """Get all reward-related data."""
        reward_paths = dict(
            reward_count='items.behavior.rewards.0.reward_count',
            volume_dispensed='items.behavior.rewards.0.volume_dispensed',
            reward_calibration='items.behavior.rewards.0.calibration',
            reward_volume='items.behavior.params.reward_volume'
        )
        return self.get_parameters(reward_paths)
    
    def get_stimulus_parameters(self) -> Dict[str, Any]:
        """Get stimulus-related parameters."""
        stimulus_data = dict(
            change_log='items.behavior.stimuli.images.change_log',
            image_walk='items.behavior.stimuli.images.image_walk',
            omitted_flashes='items.behavior.stimuli.images.flashes_omitted',
            stim_fps='items.behavior.stimuli.images.fps',
            set_log='items.behavior.stimuli.images.set_log',
            image_set='items.behavior.stimuli.images.image_set',
            trial_log='items.behavior.trial_log',
            draw_log='items.behavior.stimuli.images.draw_log'
        )
        return self.get_parameters(stimulus_data)


class ForagingPickleLoader(LazyPickleLoader):
    """
    Specialized loader for foraging pickle files.
    
    This loader understands the structure of foraging experiment pickle files
    and provides convenient access to commonly used parameters.
    """
    
    def get_experiment_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata in standardized format."""
        metadata_paths = dict(
            session_id='session_uuid',
            computer_id='comp_id',
            rig_id='rig_id',
            start_time='start_time',
            stop_time='stop_time',
            script_name= 'script',
            subject_id='mouse_id',
            user_id='user_id',
            experiment_stage='stage',
            software_version='platform.camstim',
            python_version='platform.python',
            operating_system='platform.os',
            frame_rate='stimuli.0.fps',
            stimulus_path='stimuli.0.stim_path'
        )
        return self.get_parameters(metadata_paths)
    
    def get_timing_data(self) -> Dict[str, Any]:
        """Get timing-related data."""
        timing_paths = dict(
            intervals_ms='intervalsms',
            stimulus_start_time='stimuli.0.start_time',
            stimulus_stop_time='stimuli.0.stop_time',
        )
        return self.get_parameters(timing_paths)
    
    def get_stimulus_data(self) -> Dict[str, Any]:
        """Get stimulus-related data."""
        stim_paths = dict(
            stim='stimuli.0.stim',
            sweep_table='stimuli.0.sweep_table',
            frame_list='stimuli.0.frame_list',
            runs='stimuli.0.runs',
            sweep_length='stimuli.0.sweep_length',
            sweep_order='stimuli.0.sweep_order',
            blank_length='stimuli.0.blank_length',
            blank_sweeps='stimuli.0.blank_sweeps'
        )
        return self.get_parameters(stim_paths)
    
    def get_encoder_data(self) -> Dict[str, Any]:
        """Get all lick-related data."""
        encoder_data = dict(
            encoder_disctance_cm='items.foraging.encoders.0.distance',
            encoder_distance_degrees='items.foraging.encoders.0.degrees',
            encoder_delta_degree_change='items.foraging.encoders.0.dx',
            encoder_counts='items.foraging.encoders.0.counts',
        )
        return self.get_parameters(encoder_data)


def create_loader(file_path: Union[str, Path], encoding: str = "latin1") -> LazyPickleLoader:
    """
    Factory function to create the appropriate loader based on file content.
    
    Args:
        file_path: Path to the pickle file
        encoding: Encoding to use when loading pickle
        
    Returns:
        Appropriate LazyPickleLoader subclass instance
        
    Raises:
        ValueError: If file type cannot be determined
    """
    import pickle
    from pathlib import Path
    
    # Load just the top-level structure to determine file type
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    # Load just enough to inspect the structure
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding=encoding)
    
    top_level_keys = list(data.keys()) if isinstance(data, dict) else []
    
    # Determine file type based on structure
    if 'items' in top_level_keys and isinstance(data['items'], dict):
        items_keys = list(data['items'].keys())
        if 'behavior' in items_keys:
            print("Detected behavior pickle file")
            return BehaviorPickleLoader(file_path, encoding)
        elif 'foraging' in items_keys:
            print("Detected foraging pickle file")
            return ForagingPickleLoader(file_path, encoding)
    
    # Default to behavior if structure is ambiguous
    print("Could not determine file type, defaulting to BehaviorPickleLoader")
    return BehaviorPickleLoader(file_path, encoding)
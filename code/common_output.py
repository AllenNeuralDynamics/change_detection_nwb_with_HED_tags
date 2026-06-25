"""
Common Output Format Module

This module provides standardized data structures and output formats
for converting diverse pickle file data into common, interoperable formats.
"""

import json
import pandas as pd
from typing import Any, Dict, List, Union, Optional
from datetime import datetime
from pathlib import Path
import numpy as np


class StandardizedExperimentData:
    """
    Standardized container for experiment data from any pickle file type.
    
    This class provides a common interface and data format that can accommodate
    both behavioral and foraging experiment data, making it easier to analyze
    and compare data across different experiment types.
    """
    
    def __init__(self, loader_instance):
        """
        Initialize with data from a lazy loader instance.
        
        Args:
            loader_instance: Instance of BehaviorPickleLoader or ForagingPickleLoader
        """
        self.loader = loader_instance
        self._metadata = None
        self._timing_data = None
        self._processed_data = None
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get standardized metadata."""
        if self._metadata is None:
            self._metadata = self.loader.get_experiment_metadata()
        return self._metadata
    
    @property
    def timing_data(self) -> Dict[str, Any]:
        """Get timing-related data."""
        if self._timing_data is None:
            self._timing_data = self.loader.get_timing_data()
        return self._timing_data
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a high-level summary of the experimental session."""
        summary = {
            'session_id': self.metadata.get('session_id'),
            'experiment_type': self.metadata.get('experiment_type'),
            'start_time': self.metadata.get('start_time'),
            'stop_time': self.metadata.get('stop_time'),
            'duration_seconds': self._calculate_duration(),
            'rig_id': self.metadata.get('rig_id'),
            'software_version': self.metadata.get('software_version'),
            'experiment_stage': self.metadata.get('experiment_stage')
        }
        
        # Add experiment-specific summary data
        if self.metadata.get('experiment_type') == 'behavior':
            summary.update(self._get_behavior_summary())
        elif self.metadata.get('experiment_type') == 'foraging':
            summary.update(self._get_foraging_summary())
        
        return summary
    
    def _calculate_duration(self) -> Optional[float]:
        """Calculate session duration in seconds."""
        start_time = self.metadata.get('start_time')
        stop_time = self.metadata.get('stop_time')
        
        if start_time and stop_time:
            if isinstance(start_time, (int, float)) and isinstance(stop_time, (int, float)):
                return stop_time - start_time
        return None
    
    def _get_behavior_summary(self) -> Dict[str, Any]:
        """Get behavior-specific summary data."""
        reward_data = self.loader.get_reward_data()
        lick_data = self.loader.get_lick_data()
        
        return {
            'total_rewards': reward_data.get('items.behavior.rewards_dispensed', 0),
            'total_licks': len(lick_data.get('items.behavior.lick_sensors.0.lick_events', [])),
            'task_id': self.metadata.get('task_id'),
            'reward_volume_ul': reward_data.get('items.behavior.params.reward_volume')
        }
    
    def _get_foraging_summary(self) -> Dict[str, Any]:
        """Get foraging-specific summary data."""
        encoder_data = self.loader.get_encoder_data()
        stimulus_data = self.loader.get_stimulus_data()
        
        distance = encoder_data.get('items.foraging.encoders.0.distance', 0)
        
        return {
            'total_distance_cm': distance,
            'mouse_id': self.metadata.get('mouse_id'),
            'user_id': self.metadata.get('user_id'),
            'stimulus_fps': stimulus_data.get('stimuli.0.fps'),
            'total_sweeps': len(stimulus_data.get('stimuli.0.sweep_table', []))
        }
    
    def to_pandas_summary(self) -> pd.DataFrame:
        """Convert session summary to a pandas DataFrame."""
        summary = self.get_session_summary()
        return pd.DataFrame([summary])
    
    def to_dict(self, include_raw_data: bool = False) -> Dict[str, Any]:
        """
        Convert to a comprehensive dictionary format.
        
        Args:
            include_raw_data: Whether to include raw data from the loader
            
        Returns:
            Dictionary with standardized structure
        """
        result = {
            'metadata': self.metadata,
            'timing_data': self.timing_data,
            'session_summary': self.get_session_summary()
        }
        
        if include_raw_data:
            result['raw_data'] = self._get_raw_data_subset()
        
        return result
    
    def _get_raw_data_subset(self) -> Dict[str, Any]:
        """Get a subset of raw data based on experiment type."""
        if self.metadata.get('experiment_type') == 'behavior':
            return {
                'lick_data': self.loader.get_lick_data(),
                'reward_data': self.loader.get_reward_data(),
                'stimulus_parameters': self.loader.get_stimulus_parameters()
            }
        elif self.metadata.get('experiment_type') == 'foraging':
            return {
                'encoder_data': self.loader.get_encoder_data(),
                'stimulus_data': self.loader.get_stimulus_data()
            }
        return {}
    
    def to_json(self, file_path: Optional[Union[str, Path]] = None, 
                include_raw_data: bool = False) -> str:
        """
        Convert to JSON format.
        
        Args:
            file_path: Optional path to save JSON file
            include_raw_data: Whether to include raw data
            
        Returns:
            JSON string
        """
        data = self.to_dict(include_raw_data)
        
        # Convert numpy arrays and other non-serializable types
        json_data = self._make_json_serializable(data)
        
        json_str = json.dumps(json_data, indent=2, default=str)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Recursively convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def save_summary_csv(self, file_path: Union[str, Path]):
        """Save session summary as CSV file."""
        df = self.to_pandas_summary()
        df.to_csv(file_path, index=False)
    
    def compare_with(self, other: 'StandardizedExperimentData') -> Dict[str, Any]:
        """
        Compare this experiment with another standardized experiment.
        
        Args:
            other: Another StandardizedExperimentData instance
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'same_experiment_type': self.metadata.get('experiment_type') == other.metadata.get('experiment_type'),
            'same_rig': self.metadata.get('rig_id') == other.metadata.get('rig_id'),
            'same_stage': self.metadata.get('experiment_stage') == other.metadata.get('experiment_stage'),
            'time_difference_seconds': None,
            'summary_comparison': {}
        }
        
        # Calculate time difference
        if (self.metadata.get('start_time') and other.metadata.get('start_time') and
            isinstance(self.metadata.get('start_time'), (int, float)) and
            isinstance(other.metadata.get('start_time'), (int, float))):
            comparison['time_difference_seconds'] = abs(
                self.metadata.get('start_time') - other.metadata.get('start_time')
            )
        
        # Compare summaries
        self_summary = self.get_session_summary()
        other_summary = other.get_session_summary()
        
        for key in set(self_summary.keys()) & set(other_summary.keys()):
            if isinstance(self_summary[key], (int, float)) and isinstance(other_summary[key], (int, float)):
                comparison['summary_comparison'][key] = {
                    'self': self_summary[key],
                    'other': other_summary[key],
                    'difference': self_summary[key] - other_summary[key]
                }
            else:
                comparison['summary_comparison'][key] = {
                    'self': self_summary[key],
                    'other': other_summary[key],
                    'same': self_summary[key] == other_summary[key]
                }
        
        return comparison


class ExperimentDataCollection:
    """
    Collection class for managing multiple experiment data instances.
    
    This class allows you to load and analyze multiple pickle files together,
    providing aggregate statistics and comparison capabilities.
    """
    
    def __init__(self):
        """Initialize empty collection."""
        self.experiments = []
    
    def add_experiment(self, loader_instance) -> StandardizedExperimentData:
        """
        Add an experiment to the collection.
        
        Args:
            loader_instance: Instance of a pickle loader
            
        Returns:
            StandardizedExperimentData instance
        """
        exp_data = StandardizedExperimentData(loader_instance)
        self.experiments.append(exp_data)
        return exp_data
    
    def add_from_file(self, file_path: Union[str, Path]) -> StandardizedExperimentData:
        """
        Add an experiment from a pickle file path.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            StandardizedExperimentData instance
        """
        from lazy_pickle_loader import create_loader
        loader = create_loader(file_path)
        return self.add_experiment(loader)
    
    def get_collection_summary(self) -> pd.DataFrame:
        """Get summary of all experiments in the collection."""
        summaries = []
        for i, exp in enumerate(self.experiments):
            summary = exp.get_session_summary()
            summary['collection_index'] = i
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def filter_by_type(self, experiment_type: str) -> List[StandardizedExperimentData]:
        """Filter experiments by type."""
        return [exp for exp in self.experiments 
                if exp.metadata.get('experiment_type') == experiment_type]
    
    def filter_by_rig(self, rig_id: str) -> List[StandardizedExperimentData]:
        """Filter experiments by rig ID."""
        return [exp for exp in self.experiments 
                if exp.metadata.get('rig_id') == rig_id]
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all experiments."""
        df = self.get_collection_summary()
        
        stats = {
            'total_experiments': len(self.experiments),
            'experiment_types': df['experiment_type'].value_counts().to_dict(),
            'unique_rigs': df['rig_id'].nunique(),
            'date_range': {
                'earliest': df['start_time'].min() if 'start_time' in df.columns else None,
                'latest': df['start_time'].max() if 'start_time' in df.columns else None
            }
        }
        
        # Add type-specific stats
        for exp_type in df['experiment_type'].unique():
            if pd.isna(exp_type):
                continue
            type_df = df[df['experiment_type'] == exp_type]
            stats[f'{exp_type}_stats'] = self._get_type_specific_stats(type_df, exp_type)
        
        return stats
    
    def _get_type_specific_stats(self, df: pd.DataFrame, exp_type: str) -> Dict[str, Any]:
        """Get statistics specific to an experiment type."""
        stats = {'count': len(df)}
        
        if exp_type == 'behavior':
            if 'total_rewards' in df.columns:
                stats['total_rewards'] = {
                    'mean': df['total_rewards'].mean(),
                    'std': df['total_rewards'].std(),
                    'min': df['total_rewards'].min(),
                    'max': df['total_rewards'].max()
                }
            if 'total_licks' in df.columns:
                stats['total_licks'] = {
                    'mean': df['total_licks'].mean(),
                    'std': df['total_licks'].std(),
                    'min': df['total_licks'].min(),
                    'max': df['total_licks'].max()
                }
        elif exp_type == 'foraging':
            if 'total_distance_cm' in df.columns:
                stats['total_distance_cm'] = {
                    'mean': df['total_distance_cm'].mean(),
                    'std': df['total_distance_cm'].std(),
                    'min': df['total_distance_cm'].min(),
                    'max': df['total_distance_cm'].max()
                }
        
        return stats
    
    def save_collection_summary(self, file_path: Union[str, Path]):
        """Save collection summary as CSV."""
        df = self.get_collection_summary()
        df.to_csv(file_path, index=False)
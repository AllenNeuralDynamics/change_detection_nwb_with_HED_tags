"""Change detection dataframe """
import numpy as np
from lazy_pickle_loader import BehaviorPickleLoader
import pandas as pd
from ndx_events import EventsTable, CategoricalVectorData, MeaningsTable
from ndx_hed import HedValueVector


MEANINGS = {
    "image_onset" : {
        "HED": "(Def/Image-onset, (Sensory-event, Visual-presentation, (Photograph, {image_name})), Onset)",
        "Description": "A two-dimensional photographic image has begun flashing onscreen to the participant."
        },
    "image_offset" : {
        "HED": "(Def/Image-offset, Offset)",
        "Description": "A two-dimensional photographic image has stopped flashing onscreen to the participant."
        },
    "grating_onset" : {
        "HED": "Sensory-event, Controller-agent, (Visual-presentation, Grating, Experimental-stimulus, Description/#, Onset)",
        "Description": "A grating stimulus with a specific orientation has begun flashing onscreen to the participant."
    },
    "grating_offset" : {
        "HED": "Sensory-event, Controller-agent, (Visual-presentation, Grating, Experimental-stimulus, Description/#, Offset)",
        "Description": "A grating stimulus with a specific orientation has stopped flashing onscreen to the participant."
    },
    "miss" : {
        "HED": "Agent-action, (Animal-agent, miss)",
        "Description": "The participant failed to respond within the designated response window following a stimulus change."
    },
    "correct_reject" : {
        "HED": "Agent-action, (Animal-agent, correct-reject)",
        "Description": "The participant correctly withheld a response during a catch trial where no stimulus change occurred."
    },
    "lick_correct_action": {
        "HED": "Agent-action, (Animal-agent, lick), Participant-response, Correct-action",
        "Description": "The participant made a correct lick response within the designated response window following a stimulus change."
    },
    "lick_abort": {
        "HED": "Agent-action, (Animal-agent, lick), Participant-response, Aborted-trial",
        "Description": "The participant made an early lick response before the designated response window, resulting in trial abortion."
    },
    "lick": {
        "HED": "Agent-action, (Animal-agent, lick), Participant-response",
        "Description": "The participant made a lick response."
    },
    "reward": {
        "HED": "Sensory-event, Gustatory-presentation, (Ingestible-object/Water, Reward)",
        "Description": "The participant received a water reward following a correct response."
    },
    "omitted_flash": {
        "HED": "Experiment-structure, (Sensory-event, Visual-presentation, (Image, Grey), Unexpected)",
        "Description": "An expected visual stimulus flash was omitted, resulting in a gray screen."
    },
    "fingerprint_onset": {
        "HED": "(Def/Movie, (Sensory-event, Visual-presentation (Movie, {movie_name})), Onset)",
        "Description": "A fingerprint movie stimulus has begun playing onscreen to the participant."
    },
    "fingerprint_offset": {
        "HED": "(Def/Movie, Offset)",
        "Description": "A fingerprint movie stimulus has stopped playing onscreen to the participant."
    },
    "epoch_start": {
        "HED": "Def/Epoch, Experimental-structure, (Controller-agent (Time-block, {epoch_number}), onset)",
        "Description": "The beginning of a defined experimental epoch or block.",
    },
    "epoch_end": {
        "HED": "Def/Epoch, Experimental-structure, (Controller-agent (Time-block, {epoch_number}), offset)",
        "Description": "The end of a defined experimental epoch or block.",
    },
}


def get_meanings_placeholders() -> dict:
    """
    Extract all distinct placeholder names from the MEANINGS dictionary.
    
    Parses the MEANINGS HED values to find patterns like {placeholder_name} and
    maps each event type to its placeholder name (if any).
    
    Returns
    -------
    dict
        Dictionary with two keys:
        - 'placeholders': set of all unique placeholder names found
        - 'event_to_placeholder': dict mapping event_type to its placeholder name
    """
    import re
    placeholders = set()
    event_to_placeholder = {}
    
    placeholder_pattern = re.compile(r'\{(\w+)\}')
    
    for event_type, meaning_data in MEANINGS.items():
        hed_tag = meaning_data.get('HED', '')
        match = placeholder_pattern.search(hed_tag)
        if match:
            placeholder_name = match.group(1)
            placeholders.add(placeholder_name)
            event_to_placeholder[event_type] = placeholder_name
    
    return {
        'placeholders': placeholders,
        'event_to_placeholder': event_to_placeholder
    }


HED_TAGS = {
    "image_onset": "Sensory-event, Controller-agent, (Visual-presentation, Photograph, Experimental-stimulus, Description/{}, Onset)",
    "image_offset": "Sensory-event, Controller-agent, (Visual-presentation, Photograph, Experimental-stimulus, Description/{}, Offset)",
    "image_change_onset":
    "change_window_onset":
    "change_window_offset":
    "response_window_onset":
    "response_window_offset"
    "grating_onset": "Sensory-event, Controller-agent, (Visual-presentation, Grating, Experimental-stimulus, Description/{}, Onset)",
    "grating_offset": "Sensory-event, Controller-agent, (Visual-presentation, Grating, Experimental-stimulus, Description/{}, Offset)",
    "miss": "Agent-action, (Animal-agent, Experiment-participant), Miss",
    "correct_reject": "Agent-action, (Animal-agent, Experiment-participant), Correct-action",
    "lick_correct_action": "Agent-action, (Animal-agent, Experiment-participant), Participant-response, (Lick, Correct-action)",
    "lick_incorrect_action": "Agent-action, (Animal-agent, Experiment-participant), Participant-response, (Lick, Incorrect-action)",
    "lick_abort": "Agent-action, (Animal-agent, Experiment-participant), Participant-response, (Lick, Abort)",
    "lick": "Agent-action, (Animal-agent, Experiment-participant), Participant-response, Lick",
    "reward": "Sensory-event, Controller-agent, (Gustatory-presentation, (Ingestible-object/Water, Reward))",
    "free_reward":
    "omitted_flash": "Sensory-event, Controller-agent, (Visual-presentation, Experimental-stimulus, Unexpected)",
    "fingerprint_onset": "Sensory-event, Controller-agent, (Visual-presentation, Movie, Description/{}, Onset)",
    "fingerprint_offset": "Sensory-event, Controller-agent, (Visual-presentation, Movie, Description/{}, Offset)"
}
"""
Event table:

timestamp   |  event_type (vectordata) |  image (hedvalue)


Meanings:

event_type  |  

"""

def get_interval_time_in_sec(intervals_ms: np.array):
    """Get the intervals in seconds
    Parameters
    ----------
    intervals_ms : np.array
        Array of intervals in milliseconds
    
    Returns
    -------
    np.array
        Array of intervals in seconds
    """
    return np.insert(np.cumsum(intervals_ms / 1000), 0, 0)[:-1]

def format_hed_tag(event_type, description=None):
    """Format HED tag for a given event type.
    
    Parameters
    ----------
    event_type : str
        The type of event (e.g., 'image_onset', 'lick', 'reward')
    description : str, optional
        Dynamic description to fill into the {} placeholder in the HED tag template
        
    Returns
    -------
    str
        Formatted HED tag string with logical groupings
    """
    if event_type not in HED_TAGS:
        return ""
    
    hed_template = HED_TAGS[event_type]
    
    # If template has {} placeholder and description is provided, fill it in
    if '{}' in hed_template:
        if description:
            return hed_template.format(description)
        else:
            # Remove the empty Description/{} placeholder if no description provided
            return hed_template.replace(', Description/{}', '').replace('Description/{}, ', '')
    
    return hed_template
    
def _create_event_row(timestamp: float, event_type: str, description: str = None) -> dict:
    """
    Create an event row with the appropriate placeholder column filled in.
    
    Parameters
    ----------
    timestamp : float
        Event timestamp
    event_type : str
        Type of event (e.g., 'image_onset', 'fingerprint_onset')
    description : str, optional
        Description value to fill into the appropriate placeholder column
        
    Returns
    -------
    dict
        Event row with timestamp, event_type, and appropriate placeholder columns
    """
    placeholder_info = get_meanings_placeholders()
    placeholders = placeholder_info['placeholders']
    event_to_placeholder = placeholder_info['event_to_placeholder']
    
    # Initialize row with all placeholder columns as None
    row = {
        'timestamp': timestamp,
        'event_type': event_type,
    }
    for placeholder in placeholders:
        row[placeholder] = None
    
    # Fill in the appropriate placeholder column if this event type has one
    if event_type in event_to_placeholder and description is not None:
        placeholder_name = event_to_placeholder[event_type]
        row[placeholder_name] = description
    
    return row


def get_omitted_events_table(loader: BehaviorPickleLoader) -> pd.DataFrame:
    """
    Create an events table for omitted flashes with timestamp, event_type, and placeholder columns
    
    Parameters
    ----------
    loader : BehaviorPickleLoader
        The loader object containing the experimental data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, event_type, and placeholder columns from MEANINGS
    """
    
    # Get the necessary data
    stimulus_parameters = loader.get_stimulus_parameters()
    timing = loader.get_timing_data()
    
    # Extract timing info
    intervals_ms = timing['intervals_ms']
    cumulative_times = get_interval_time_in_sec(intervals_ms)
    
    # Extract stimulus parameters
    omitted_flashes = stimulus_parameters['omitted_flashes']
    
    # Create list to store event rows
    events = []
    
    # Add omitted flash events by converting frame indices to timestamps
    for omitted_frame_idx in omitted_flashes:
        if omitted_frame_idx < len(cumulative_times):
            omitted_timestamp = cumulative_times[omitted_frame_idx]
            events.append(_create_event_row(omitted_timestamp, 'omitted_flash'))
    
    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.sort_values('timestamp').reset_index(drop=True)
    
    return events_df

def get_licks_and_rewards_events_table(loader: BehaviorPickleLoader) -> pd.DataFrame:
    """Pull the lick and reward events from pickle file with proper lick classification.
    
    Lick classification logic:
    - `lick_correct_action`: First lick within the response_window after a stimulus change
    - `lick`: Additional licks after the initial correct lick (during response window or 
      reward consumption period of ~3.5 seconds)
    - `lick_incorrect_action`: Lick before the response_window starts (early_response), 
      which causes the trial to abort
    
    Parameters
    ----------
    loader : BehaviorPickleLoader
        Loader object containing experimental data

    Return
    ------
    pd.DataFrame
        DataFrame with columns: timestamp, event_type, and placeholder columns from MEANINGS
    """
    events = []
    
    # Get metadata for response window parameters
    metadata = loader.get_experiment_metadata()
    response_window = metadata.get('response_window', [0.15, 0.75])
    response_window_start = response_window[0] if response_window else 0.15
    response_window_end = response_window[1] if response_window and len(response_window) > 1 else 0.75
    
    # Reward consumption period (time after reward where licks don't abort)
    reward_consumption_period = 3.5  # seconds
    
    # Get trial log which contains per-trial lick classification info
    stimulus_params = loader.get_stimulus_parameters()
    trial_log = stimulus_params.get('trial_log', [])
    
    # Track all licks we've classified to avoid duplicates
    classified_lick_times = set()
    
    for trial in trial_log:
        trial_licks = trial.get('licks', [])
        trial_rewards = trial.get('rewards', [])
        stimulus_changes = trial.get('stimulus_changes', [])
        trial_events = trial.get('events', [])
        
        # Get change time if there was a stimulus change
        change_time = None
        if stimulus_changes:
            # stimulus_changes format: [(('im000', 'im000'), ('im031', 'im031'), time, frame)]
            change_time = stimulus_changes[0][2]
        
        # Check for early_response/abort events
        has_early_response = any(
            event[0] == 'early_response' for event in trial_events
        )
        
        # Get reward time if any
        reward_time = None
        if trial_rewards:
            # rewards format: [(volume, time, frame)]
            reward_time = trial_rewards[0][1]
        
        # Track if we've seen the first correct lick in this trial
        first_correct_lick_seen = False
        
        for lick_time, lick_frame in trial_licks:
            # Skip if we've already classified this lick (avoid duplicates)
            if lick_time in classified_lick_times:
                continue
            classified_lick_times.add(lick_time)
            
            # Determine lick type based on timing
            if change_time is not None:
                time_since_change = lick_time - change_time
                
                # Check if lick is within response window
                if response_window_start <= time_since_change <= response_window_end:
                    if not first_correct_lick_seen:
                        # First lick in response window = correct action
                        events.append(_create_event_row(lick_time, 'lick_correct_action'))
                        first_correct_lick_seen = True
                    else:
                        # Subsequent licks in response window
                        events.append(_create_event_row(lick_time, 'lick'))
                elif time_since_change < response_window_start:
                    # Lick before response window starts (but after change)
                    # This shouldn't typically happen as early_response is before change
                    events.append(_create_event_row(lick_time, 'lick_incorrect_action'))
                else:
                    # Lick after response window
                    # Check if within reward consumption period
                    if reward_time is not None:
                        time_since_reward = lick_time - reward_time
                        if 0 <= time_since_reward <= reward_consumption_period:
                            # Lick during reward consumption - normal lick
                            events.append(_create_event_row(lick_time, 'lick'))
                        else:
                            # Lick outside consumption period
                            events.append(_create_event_row(lick_time, 'lick'))
                    else:
                        events.append(_create_event_row(lick_time, 'lick'))
            else:
                # No stimulus change in this trial
                if has_early_response:
                    # This trial was aborted due to early lick
                    events.append(_create_event_row(lick_time, 'lick_incorrect_action'))
                else:
                    # Regular lick (no change context)
                    events.append(_create_event_row(lick_time, 'lick'))
        
        # Add reward events
        for reward_entry in trial_rewards:
            if len(reward_entry) >= 2:
                reward_timestamp = reward_entry[1]  # Second element is timestamp
                events.append(_create_event_row(reward_timestamp, 'reward'))
    
    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.sort_values('timestamp').reset_index(drop=True)
    
    return events_df

def _get_stimulus_epoch(set_log, current_set_index, start_frame, n_frames):
    """
    Get the frame range for which a stimulus setting is active.
    Adapted from legacy_change_detection.py
    
    Parameters
    ----------
    set_log : list
        List of set_log entries
    current_set_index : int
        Index of current set_log entry
    start_frame : int
        Frame number where stimulus was set
    n_frames : int
        Total number of frames
        
    Returns
    -------
    tuple
        (start_frame, end_frame) - end frame is non-inclusive
    """
    try:
        next_set_event = set_log[current_set_index + 1]  # attr_name, attr_value, time, frame
    except IndexError:  # assume this is the last set event
        next_set_event = (None, None, None, n_frames, )

    return (start_frame, next_set_event[3])  # end frame isn't inclusive


def _get_draw_epochs(draw_log, start_frame, stop_frame):
    """
    Find epochs where stimulus was actually drawn within a time period.
    Adapted from legacy_change_detection.py
    
    Parameters
    ----------
    draw_log : list
        Binary list where 1 indicates frame was drawn
    start_frame : int
        Start of search period (inclusive)
    stop_frame : int
        End of search period (non-inclusive)
        
    Returns
    -------
    list
        List of (epoch_start, epoch_end) tuples for each draw period
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
                (current_frame - epoch_length - 1, current_frame - 1, )
            )

    return draw_epochs


def _resolve_image_category(change_log, frame):
    """
    Determine the image category at a given frame.
    Adapted from legacy_change_detection.py
    
    Parameters
    ----------
    change_log : list
        List of change events
    frame : int
        Frame number to check
        
    Returns
    -------
    str
        Image category at this frame
    """
    def unpack_change_log(change):
        (from_category, from_name), (to_category, to_name, ), time, frame = change
        return dict(
            frame=frame,
            time=time,
            from_category=from_category,
            to_category=to_category,
            from_name=from_name,
            to_name=to_name,
        )
    
    for change in (unpack_change_log(c) for c in change_log):
        if frame < change['frame']:
            return change['from_category']

    return change['to_category']


def get_image_events_table(loader: BehaviorPickleLoader) -> pd.DataFrame:
    """
    Create image events table using the legacy algorithm with corrected timing.
    
    This properly handles:
    - set_log: defines what stimulus is active and provides ABSOLUTE timestamps
    - draw_log: defines when stimuli were actually displayed
    - Correctly pairs onset/offset events for each draw epoch
    - Uses set_log timestamps as the authoritative timing source
    
    Parameters
    ----------
    loader : BehaviorPickleLoader
        The loader object containing the experimental data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, event_type, and placeholder columns from MEANINGS
    """
    # Get data
    stimulus_parameters = loader.get_stimulus_parameters()
    timing = loader.get_timing_data()
    set_log = stimulus_parameters['set_log']
    draw_log = stimulus_parameters['draw_log']
    
    # Build frame-based time array from intervals
    intervals_ms = timing['intervals_ms']
    frame_times = get_interval_time_in_sec(intervals_ms)
    n_frames = len(frame_times)
    
    if not set_log or not draw_log:
        placeholder_info = get_meanings_placeholders()
        columns = ['timestamp', 'event_type'] + sorted(placeholder_info['placeholders'])
        return pd.DataFrame(columns=columns)
    
    # Calculate timing offset between frame-based times and absolute experiment times
    # The set_log contains absolute experiment timestamps
    # The frame_times start at 0 when the display frames begin
    first_set_time = set_log[0][2]  # Absolute timestamp from set_log
    first_set_frame = set_log[0][3]  # Frame index
    timing_offset = first_set_time - frame_times[first_set_frame]
    
    events = []
    
    # Iterate through set_log entries (each time stimulus parameters change)
    for idx, (attr_name, attr_value, _time, frame) in enumerate(set_log):
        orientation = attr_value if attr_name.lower() == "ori" else np.nan
        image_name = attr_value if attr_name.lower() == "image" else np.nan
        
        # Get the time period this stimulus setting is active
        stimulus_epoch = _get_stimulus_epoch(set_log, idx, frame, n_frames)
        
        # Find when stimulus was actually drawn during this period
        draw_epochs = _get_draw_epochs(draw_log, *stimulus_epoch)
        
        for epoch_start, epoch_end in draw_epochs:
            # Visual stimulus doesn't actually change until start of following frame
            epoch_start += 1
            epoch_end += 1
            
            # Bounds check
            if epoch_start >= len(frame_times) or epoch_end >= len(frame_times):
                continue
            
            # Determine stimulus type and description
            if not pd.isna(image_name):
                stim_type = "image"
                description = image_name
            elif not pd.isna(orientation):
                stim_type = "grating"
                description = f"orientation_{orientation}"
            else:
                continue
            
            # Convert frame-based times to absolute experiment times using offset
            onset_time = frame_times[epoch_start] + timing_offset
            offset_time = frame_times[epoch_end] + timing_offset
            
            # Add onset event
            events.append(_create_event_row(onset_time, f'{stim_type}_onset', description))
            
            # Add offset event
            events.append(_create_event_row(offset_time, f'{stim_type}_offset', description))
    
    # Create DataFrame
    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.sort_values('timestamp').reset_index(drop=True)
    
    return events_df


def get_fingerprint_events_table(loader: BehaviorPickleLoader) -> pd.DataFrame:
    """
    Create an events table for fingerprint movie presentations
    
    Parameters
    ----------
    loader : BehaviorPickleLoader
        The loader object containing the experimental data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, event_type, and placeholder columns from MEANINGS
    """
    
    # Get the necessary data
    metadata = loader.get_experiment_metadata()
    stimulus_parameters = loader.get_stimulus_parameters()
    timing = loader.get_timing_data()
    
    if 'epilogue' not in metadata or metadata['epilogue']['name'] != 'fingerprint':
        return pd.DataFrame()  # No fingerprint data
    
    epilogue = metadata['epilogue']
    runs = epilogue['params']['runs']
    
    # Calculate timing
    intervals_ms = timing['intervals_ms']
    cumulative_times = get_interval_time_in_sec(intervals_ms)
    total_duration = cumulative_times[-1]
    
    # Find when fingerprint starts (after last image)
    set_log = stimulus_parameters['set_log']
    last_image_time = set_log[-1][2] if set_log else 0
    
    # Estimate fingerprint start (add small buffer after last image)
    fingerprint_start = last_image_time + 1.0  # 1 second buffer
    fingerprint_duration = total_duration - fingerprint_start
    duration_per_run = fingerprint_duration / runs
    
    events = []
    
    for run in range(runs):
        run_start = fingerprint_start + (run * duration_per_run)
        run_end = run_start + duration_per_run
        
        # Fingerprint onset - uses movie_name placeholder
        events.append(_create_event_row(run_start, 'fingerprint_onset', f"run_{run+1}"))
        
        # Fingerprint offset - uses movie_name placeholder
        events.append(_create_event_row(run_end, 'fingerprint_offset', f"run_{run+1}"))
    
    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.sort_values('timestamp').reset_index(drop=True)
    
    return events_df


def get_trial_statistics(loader: BehaviorPickleLoader) -> dict:
    """
    Get trial-level statistics from the experiment.
    
    Parameters
    ----------
    loader : BehaviorPickleLoader
        The loader object containing the experimental data
        
    Returns
    -------
    dict
        Dictionary containing trial statistics:
        - total_trials: Total number of trials
        - successful_trials: Number of successful trials (hit)
        - aborted_trials: Number of aborted trials (early lick)
        - missed_trials: Number of missed trials (no response in window)
        - catch_trials: Number of catch trials
        - auto_reward_trials: Number of auto-reward trials
        - failure_repeats: Max number of times a trial can be repeated
        - response_window: Response window parameters [start, end] in seconds
    """
    metadata = loader.get_experiment_metadata()
    stimulus_params = loader.get_stimulus_parameters()
    trial_log = stimulus_params.get('trial_log', [])
    
    # Count trial outcomes
    successful = 0
    aborted = 0
    missed = 0
    catch_trials = 0
    auto_reward_trials = 0
    
    for trial in trial_log:
        trial_events = trial.get('events', [])
        trial_params = trial.get('trial_params', {})
        
        # Check for specific outcomes
        has_abort = any(event[0] == 'abort' for event in trial_events)
        has_miss = any(event[0] == 'miss' for event in trial_events)
        is_successful = trial.get('success', False)
        
        if has_abort:
            aborted += 1
        elif has_miss:
            missed += 1
        elif is_successful:
            successful += 1
        
        # Count trial types
        if trial_params.get('catch', False):
            catch_trials += 1
        if trial_params.get('auto_reward', False):
            auto_reward_trials += 1
    
    return {
        'total_trials': len(trial_log),
        'successful_trials': successful,
        'aborted_trials': aborted,
        'missed_trials': missed,
        'catch_trials': catch_trials,
        'auto_reward_trials': auto_reward_trials,
        'failure_repeats': metadata.get('failure_repeats', 5),
        'response_window': metadata.get('response_window', [0.15, 0.75]),
        'timeout_duration': metadata.get('timeout_duration', 0.3),
    }


def get_epoch_events_table(loader: BehaviorPickleLoader) -> pd.DataFrame:
    """
    Create an events table for experiment epochs (blocks).
    
    Identifies three main epochs:
    1. warm_up - Initial trials with auto_reward=True
    2. change_detection - Main task trials
    3. fingerprint - Movie presentation at the end (if present)
    
    Parameters
    ----------
    loader : BehaviorPickleLoader
        The loader object containing the experimental data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, event_type, epoch_number, and other placeholder columns
    """
    metadata = loader.get_experiment_metadata()
    stimulus_params = loader.get_stimulus_parameters()
    timing = loader.get_timing_data()
    trial_log = stimulus_params.get('trial_log', [])
    
    if not trial_log:
        placeholder_info = get_meanings_placeholders()
        columns = ['timestamp', 'event_type'] + sorted(placeholder_info['placeholders'])
        return pd.DataFrame(columns=columns)
    
    events = []
    
    # Calculate total session duration
    intervals_ms = timing['intervals_ms']
    cumulative_times = get_interval_time_in_sec(intervals_ms)
    total_duration = cumulative_times[-1] if len(cumulative_times) > 0 else 0
    
    # Find warm-up trials (auto_reward=True)
    warmup_end_idx = 0
    for i, trial in enumerate(trial_log):
        if not trial.get('trial_params', {}).get('auto_reward', False):
            warmup_end_idx = i
            break
    else:
        # All trials are warm-up (unlikely but handle it)
        warmup_end_idx = len(trial_log)
    
    # Get timestamps from trial events
    def get_trial_start_time(trial):
        for event in trial.get('events', []):
            if event[0] == 'trial_start':
                return event[2]
        return None
    
    def get_trial_end_time(trial):
        for event in trial.get('events', []):
            if event[0] == 'trial_end':
                return event[2]
        return None
    
    # Epoch 1: Warm-up
    if warmup_end_idx > 0:
        warmup_start = get_trial_start_time(trial_log[0])
        warmup_end = get_trial_end_time(trial_log[warmup_end_idx - 1])
        
        if warmup_start is not None:
            events.append(_create_event_row(warmup_start, 'epoch_start', 'warm_up'))
        if warmup_end is not None:
            events.append(_create_event_row(warmup_end, 'epoch_end', 'warm_up'))
    
    # Epoch 2: Change Detection (main task)
    if warmup_end_idx < len(trial_log):
        task_start = get_trial_start_time(trial_log[warmup_end_idx])
        task_end = get_trial_end_time(trial_log[-1])
        
        if task_start is not None:
            events.append(_create_event_row(task_start, 'epoch_start', 'change_detection'))
        if task_end is not None:
            events.append(_create_event_row(task_end, 'epoch_end', 'change_detection'))
    
    # Epoch 3: Fingerprint (if present)
    if 'epilogue' in metadata and metadata['epilogue'] is not None:
        epilogue = metadata['epilogue']
        if epilogue.get('name') == 'fingerprint':
            # Fingerprint starts after last trial ends
            last_trial_end = get_trial_end_time(trial_log[-1])
            if last_trial_end is not None:
                fingerprint_start = last_trial_end + 1.0  # Small buffer
                fingerprint_end = total_duration
                
                events.append(_create_event_row(fingerprint_start, 'epoch_start', 'fingerprint'))
                events.append(_create_event_row(fingerprint_end, 'epoch_end', 'fingerprint'))
    
    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.sort_values('timestamp').reset_index(drop=True)
    
    return events_df


def get_complete_events_table(loader: BehaviorPickleLoader) -> pd.DataFrame:
    """
    Create a complete events table combining all event types
    
    Parameters
    ----------
    loader : BehaviorPickleLoader
        The loader object containing the experimental data
        
    Returns
    -------
    pd.DataFrame
        Complete DataFrame with columns: timestamp, event_type, and placeholder columns from MEANINGS
        (e.g., image_name, movie_name, epoch_number)
    """
    
    # Get all event tables
    image_events = get_image_events_table(loader)
    omitted_events = get_omitted_events_table(loader)
    licks_rewards_events = get_licks_and_rewards_events_table(loader)
    fingerprint_events = get_fingerprint_events_table(loader)
    epoch_events = get_epoch_events_table(loader)
    
    # Combine all events
    all_events = []
    
    for df in [image_events, omitted_events, licks_rewards_events, fingerprint_events, epoch_events]:
        if len(df) > 0:
            all_events.append(df)
    
    if all_events:
        complete_events_df = pd.concat(all_events, ignore_index=True)
        complete_events_df = complete_events_df.sort_values('timestamp').reset_index(drop=True)
        return complete_events_df
    else:
        placeholder_info = get_meanings_placeholders()
        columns = ['timestamp', 'event_type'] + sorted(placeholder_info['placeholders'])
        return pd.DataFrame(columns=columns)


def get_event_type_values(loader: BehaviorPickleLoader) -> dict:
    """
    Extract all unique description values for each event type that has dynamic descriptions.
    
    This collects all possible values that can fill in the '#' placeholder in MEANINGS
    for event types like image_onset, grating_onset, fingerprint_onset, etc.
    
    Parameters
    ----------
    loader : BehaviorPickleLoader
        The loader object containing the experimental data
        
    Returns
    -------
    dict
        Dictionary mapping event_type to sorted list of unique description values.
        Only includes event types that have dynamic descriptions (those with '#' in MEANINGS).
    """
    # Get stimulus parameters
    stimulus_parameters = loader.get_stimulus_parameters()
    metadata = loader.get_experiment_metadata()
    set_log = stimulus_parameters.get('set_log', [])
    
    # Initialize collections for each event type with dynamic descriptions
    event_values = {
        'image_onset': set(),
        'image_offset': set(),
        'grating_onset': set(),
        'grating_offset': set(),
        'fingerprint_onset': set(),
        'fingerprint_offset': set(),
    }
    
    # Extract image and grating names from set_log
    for attr_name, attr_value, _time, _frame in set_log:
        if attr_name.lower() == "image":
            event_values['image_onset'].add(attr_value)
            event_values['image_offset'].add(attr_value)
        elif attr_name.lower() == "ori":
            description = f"orientation_{attr_value}"
            event_values['grating_onset'].add(description)
            event_values['grating_offset'].add(description)
    
    # Extract fingerprint run values
    if 'epilogue' in metadata and metadata['epilogue'].get('name') == 'fingerprint':
        runs = metadata['epilogue']['params'].get('runs', 0)
        for run in range(runs):
            run_desc = f"run_{run + 1}"
            event_values['fingerprint_onset'].add(run_desc)
            event_values['fingerprint_offset'].add(run_desc)
    
    # Convert sets to sorted lists and filter out empty ones
    return {
        event_type: sorted(list(values))
        for event_type, values in event_values.items()
        if len(values) > 0
    }


def create_event_type_meanings_tables():
    """
    Create a MeaningsTable containing the event types, descriptions, and HED tag meanings.
    
    The MeaningsTable contains:
    - value: The event type string (e.g., 'image_onset', 'lick')
    - meaning: The HED tag annotation for that event type
    - event_description: Human-readable description of the event type
    
    Returns
    -------
    MeaningsTable
        MeaningsTable containing event types with their HED tags and descriptions.
    """
    meanings_table = MeaningsTable(
        name="change_detection_events_meanings",
        description="Meanings table for change detection event types containing HED tag annotations and descriptions."
    )
    
    # Collect all data first, then add the column with data
    # (MeaningsTable requires columns with data to be added after rows, or data provided upfront)
    event_types = []
    hed_tags = []
    descriptions = []
    
    for event_type, meaning_data in MEANINGS.items():
        event_types.append(event_type)
        hed_tags.append(meaning_data.get('HED', ''))
        descriptions.append(meaning_data.get('Description', ''))
    
    # Add rows with value and meaning (built-in columns)
    for event_type, hed_tag in zip(event_types, hed_tags):
        meanings_table.add_row(
            value=event_type, 
            meaning=hed_tag,
        )
    
    # Add event_description column with data after rows are added
    # Note: 'description' is reserved as a table parameter, so we use 'event_description'
    meanings_table.add_column(
        name="event_description",
        description="Human-readable description of the event type",
        data=descriptions
    )
    
    return meanings_table


def export_hed_sidecar(loader: BehaviorPickleLoader = None, output_path: str = None) -> dict:
    """
    Export the MEANINGS dictionary to a HED-compliant JSON sidecar format.
    
    The JSON sidecar follows the BIDS/HED specification format for events.json files,
    with categorical columns having "Levels" for descriptions and "HED" for annotations.
    
    Parameters
    ----------
    loader : BehaviorPickleLoader, optional
        The loader object to extract actual values for Levels. If None, Levels will be empty.
    output_path : str, optional
        Path to write the JSON sidecar file. If None, only returns the dict.
        
    Returns
    -------
    dict
        HED-compliant JSON sidecar dictionary
    """
    import json
    
    # Build the event_type categorical column structure
    event_type_levels = {}
    event_type_hed = {}
    
    for event_type, meaning_data in MEANINGS.items():
        event_type_levels[event_type] = meaning_data.get('Description', '')
        event_type_hed[event_type] = meaning_data.get('HED', '')
    
    sidecar = {
        "event_type": {
            "Description": "Type of behavioral event in the change detection task",
            "Levels": event_type_levels,
            "HED": event_type_hed
        }
    }
    
    # Get placeholder columns info
    placeholder_info = get_meanings_placeholders()
    
    # Get actual values from loader if provided
    if loader is not None:
        event_values = get_event_type_values(loader)
    else:
        event_values = {}
    
    # Extract unique values for each placeholder from the data
    # image_name values come from image_onset/image_offset events
    image_names = set()
    if 'image_onset' in event_values:
        image_names.update(event_values['image_onset'])
    if 'image_offset' in event_values:
        image_names.update(event_values['image_offset'])
    
    # movie_name values come from fingerprint_onset/fingerprint_offset events  
    movie_names = set()
    if 'fingerprint_onset' in event_values:
        movie_names.update(event_values['fingerprint_onset'])
    if 'fingerprint_offset' in event_values:
        movie_names.update(event_values['fingerprint_offset'])
    
    # epoch_number values - these are fixed epoch names
    epoch_names = {'warm_up', 'change_detection', 'fingerprint'}
    
    # Build placeholder column entries with Levels
    placeholder_config = {
        'image_name': {
            "Description": "Name identifier of the image stimulus presented",
            "Levels": {name: f"Image stimulus: {name}" for name in sorted(image_names)},
            "HED": {name: f"(Photograph, Label/{name})" for name in sorted(image_names)} if image_names else "(Photograph, Label/#)"
        },
        'movie_name': {
            "Description": "Name identifier of the movie/fingerprint stimulus",
            "Levels": {name: f"Movie stimulus: {name}" for name in sorted(movie_names)},
            "HED": {name: f"(Movie, Label/{name})" for name in sorted(movie_names)} if movie_names else "(Movie, Label/#)"
        },
        'epoch_number': {
            "Description": "Name of the experimental epoch or block",
            "Levels": {
                "warm_up": "Initial warm-up period with auto-reward trials",
                "change_detection": "Main change detection task trials",
                "fingerprint": "Fingerprint movie presentation epoch"
            },
            "HED": {
                "warm_up": "(Time-block, Label/warm_up)",
                "change_detection": "(Time-block, Label/change_detection)",
                "fingerprint": "(Time-block, Label/fingerprint)"
            }
        }
    }
    
    for placeholder in sorted(placeholder_info['placeholders']):
        if placeholder in placeholder_config:
            sidecar[placeholder] = placeholder_config[placeholder]
        else:
            sidecar[placeholder] = {
                "Description": f"Value for {placeholder} placeholder in HED tags",
                "Levels": {},
                "HED": "Label/#"
            }
    
    # Write to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(sidecar, f, indent=2)
        print(f"HED sidecar written to: {output_path}")
    
    return sidecar


def create_nwb_events_table(
    loader: BehaviorPickleLoader,
    monitor_delay: float,
    name: str = "change_detection_events",
    description: str = "Change detection behavioral events with HED annotations"
) -> tuple:
    """
    Create an NWB EventsTable from change detection data with monitor delay correction.
    
    This creates a structured EventsTable with:
    - An 'event_type' column of type CategoricalVectorData linked to a MeaningsTable
    - Dynamic columns of type HedValueVector for each placeholder (e.g., 'image_name', 'movie_name', 'epoch_number')
      that get substituted into the HED tag templates
    
    Also creates a MeaningsTable containing:
    - value: The event type string
    - meaning: The HED tag annotation
    - event_description: Human-readable description
    
    Parameters
    ----------
    loader : BehaviorPickleLoader
        The loader object containing the experimental data
    monitor_delay : float
        Monitor delay in seconds to add to timestamps for alignment
    name : str, optional
        Name for the EventsTable, by default "change_detection_events"
    description : str, optional
        Description for the EventsTable
        
    Returns
    -------
    tuple
        (EventsTable, MeaningsTable) where:
        - EventsTable: NWB EventsTable object with corrected timestamps and properly typed columns
        - MeaningsTable: Table mapping event types to their HED tags and descriptions
    """
    # Get the pandas events table
    events_df = get_complete_events_table(loader)
    
    if len(events_df) == 0:
        raise ValueError("No events found in loader data")
    
    # Apply monitor delay correction
    events_df['timestamp'] = events_df['timestamp'] + monitor_delay
    
    # Create MeaningsTable from MEANINGS dictionary
    meanings_table = create_event_type_meanings_tables()
    
    # Create the event_type column as CategoricalVectorData linked to meanings table
    event_type_col = CategoricalVectorData(
        name="event_type",
        description="Type of behavioral event (e.g., image_onset, lick, reward)",
        meanings=meanings_table
    )
    
    # Create the EventsTable with the meanings_tables parameter
    # This ensures the MeaningsTable is stored within the EventsTable
    nwb_events_table = EventsTable(
        name=name,
        description=description,
        columns=[event_type_col],
        meanings_tables=[meanings_table]
    )
    
    # Get placeholder info and add HedValueVector columns for each placeholder
    # HedValueVector is used because these values get substituted into HED tag placeholders
    # Each HedValueVector requires a 'hed' parameter - the HED tag template with # placeholder
    placeholder_info = get_meanings_placeholders()
    placeholder_config = {
        'image_name': {
            "description": "Name of the image stimulus (filled for image_onset/image_offset events)",
            "hed": "Photograph, Label/#"
        },
        'movie_name': {
            "description": "Name of the movie/fingerprint stimulus (filled for fingerprint_onset/fingerprint_offset events)",
            "hed": "Movie, Label/#"
        },
        'epoch_number': {
            "description": "Epoch number identifier (filled for epoch_start/epoch_end events)",
            "hed": "Time-block, Label/#"
        },
    }
    
    # Add placeholder columns as HedValueVector type
    for placeholder in sorted(placeholder_info['placeholders']):
        config = placeholder_config.get(placeholder, {
            "description": f"Description value for events using {placeholder} placeholder",
            "hed": "Label/#"
        })
        nwb_events_table.add_column(
            name=placeholder,
            description=config["description"],
            col_cls=HedValueVector,
            hed=config["hed"],
        )
    
    # Add rows with all data at once
    for _, row in events_df.iterrows():
        row_data = {'timestamp': row['timestamp'], 'event_type': row['event_type']}
        for placeholder in sorted(placeholder_info['placeholders']):
            value = row.get(placeholder)
            row_data[placeholder] = value if value is not None and not pd.isna(value) else ""
        nwb_events_table.add_row(**row_data)
    
    return nwb_events_table, meanings_table

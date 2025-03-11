import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
from matplotlib.gridspec import GridSpec
import tkinter as tk 
from tkinter import filedialog, messagebox
import logging
import traceback
from datetime import datetime
from scipy import signal
from scipy.stats import sem
import os
from datetime import datetime
import mne
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from edfio import Edf, EdfSignal
import re

class MuseTrialProcessor:
    def __init__(self):
        """Initialize MuseTrialProcessor with default configurations."""
        self.GUI = {
            'LAYOUT': {
                'figure_size': (15, 12),
                'left_margin': 0.15,
                'right_margin': 0.85,
                'top_margin': 0.95,
                'bottom_margin': 0.1,
                'subplot_spacing': 0.3
            },
            'CONTROLS': {
                'filter_toggles': {
                    'x': 0.88,
                    'y': 0.8,
                    'width': 0.1,
                    'height': 0.1
                },
                'scale_slider': {
                    'x': 0.88,
                    'y': 0.6,
                    'width': 0.1,
                    'height': 0.02
                },
                'prev_button': {
                    'x': 0.88,
                    'y': 0.4,
                    'width': 0.1,
                    'height': 0.04
                },
                'next_button': {
                    'x': 0.88,
                    'y': 0.3,
                    'width': 0.1,
                    'height': 0.04
                },
                'avg_button': {
                    'x': 0.88,
                    'y': 0.2,
                    'width': 0.1,
                    'height': 0.04
                }
            }
        }
        
        self.CONFIG = {
            'ICA': {
                'enabled': False,
                'n_components': 4,
                'method': 'fastica',
                'random_state': 64,
                'max_iter': 'auto',
                'fit_params': [None],
                'extended': True,
                'blink_tmin': -0.2,
                'blink_tmax': 0.5,
                'exclude': None
            },
            'PHOTODIODE': {
                'DETECTION': {
                    'THRESHOLD': 550,
                    'MIN_AMPLITUDE': 150,
                    'SLOPE_DIRECTION': 'neg',
                    'MIN_DURATION': 0.001,
                    'MAX_DURATION': 0.021,
                    'SEARCH_WINDOW': 0.5,
                    'CHANNEL': 'AUX_L',
                    'TIME_SHIFT': 0.0 
                }
            },
            'SAMPLING_RATE': 256,
            'TRIAL_WINDOW': {
                'START': -0.2,
                'END': 0.6,
                'BASELINE': (-0.2, 0)    
            },
            'CHANNELS': {
                'EEG': ['AF7', 'AF8', 'TP9', 'TP10'],
                'AUX': ['AUX_L', 'AUX_R']
            },
            'FILTERS': {
                'EEG': {
                    'enabled': False,
                    'highpass': 0.5,
                    'lowpass': 25,
                    'notch': 50,
                    'method': 'fir',
                    'design': 'firwin',
                    'window': 'hamming',
                    'phase': 'zero',
                    'order': '5s',
                    'pad': 'reflect_limited'
                },
                'AUX': {
                    'enabled': False,
                    'highpass': None,
                    'lowpass': None,
                    'notch': 50,
                    'method': 'fir',
                    'design': 'firwin',
                    'window': 'hamming',
                    'phase': 'zero',
                    'order': 'auto',
                    'pad': 'reflect_limited'
                }
            },
            'PLOT': {
                'SCALES': {
                    'EEG': {
                        'default': 20,
                        'max': 1700,
                        'min': 0
                    },
                    'AUX': {
                        'default': 1000,
                        'max': 1700,
                        'min': 0
                    }
                },
                'COLORS': {
                    'AF7': '#4169E1',
                    'AF8': '#4169E1',
                    'TP9': '#32CD32',
                    'TP10': '#32CD32',
                    'AUX_L': '#9467bd',
                    'AUX_R': '#8c564b'
                },
                'STYLES': {
                    'background': '#F5F5F7',
                    'grid_alpha': 0.15,
                    'line_width': 2,
                    'marker_alpha': 0.5,
                    'axis_label_pad': 10,
                    'title_pad': 15
                }
            },
            'VISUALIZATION': {
                'STIM_WINDOW_ALPHA': 0.2,
                'STIM_WINDOW_COLOR': 'gray'
            },
            'AVERAGING': {
                'conditions': ['highpos', 'neutral', 'Grand Average']
            }
        }
        self._setup_logging()
        self._initialize_attributes()
        self._initialize_ica()
        
    def _setup_logging(self):
        """Setup detailed logging configuration."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=f'trialSegmenter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        
        self.logger = logging.getLogger('muse_processor')
        
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        mne.set_log_level('WARNING')
        
    def _initialize_attributes(self):
        """Initialize all class attributes."""
        self.CHANNELS = self.CONFIG['CHANNELS']
        self.CHANNEL_TYPES = {ch: 'EEG' for ch in self.CHANNELS['EEG']}
        self.CHANNEL_TYPES.update({ch: 'AUX' for ch in self.CHANNELS['AUX']})
        
        self.trials_event_id = {}
        self.raw_data = None
        self.filtered_data = None
        self.trials = None
        self.timestamps = None
        self.experiment_start_time = None
        self.input_filepath = None
        self.log_data = None
        self.SF = self.CONFIG['SAMPLING_RATE']
        self.avg_stim_duration = 0.5
        
        self.current_trial = 0
        self.trial_timestamps = []
        self.current_time = 0.0
        self.current_condition = 0
        self.epoch_length = self.CONFIG['TRIAL_WINDOW']['END'] - self.CONFIG['TRIAL_WINDOW']['START']
        
        self.channel_scales = {
            ch: self.CONFIG['PLOT']['SCALES']['EEG']['default'] 
            for ch in self.CHANNELS['EEG']
        }
        self.channel_scales.update({
            ch: self.CONFIG['PLOT']['SCALES']['AUX']['default'] 
            for ch in self.CHANNELS['AUX']
        })
        
        self.electrode_states = {
            'AF7': True,
            'AF8': True,
            'TP9': False,
            'TP10': False
        }

        # Scales for different plot components
        self.y_scale_erp = self.CONFIG['PLOT']['SCALES']['EEG']['default']
        self.y_scale_pd = self.CONFIG['PLOT']['SCALES']['AUX']['default']
        
        # Trial info
        self.trial_info = None
        self.trial_info_filepath = None
        
    def _initialize_ica(self):
        """Initialize ICA-related attributes."""
        self.ica = None
        self.ica_applied = False
        
    def load_muse_data(self, filepath):
        """Load and preprocess MUSE EEG data from CSV."""
        self.logger.info(f"Loading MUSE data from {filepath}")
        self.input_filepath = filepath
        
        try:
            data = {
                'timestamp': [],
                'TP9': [], 'AF7': [], 'AF8': [], 'TP10': [],
                'AUX_L': [], 'AUX_R': []
            }
            
            first_timestamp = None
            
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split(',')
                        if len(parts) >= 8 and '/muse/eeg' in parts[1]:
                            timestamp = float(parts[0])
                            if first_timestamp is None:
                                first_timestamp = timestamp
                                self.experiment_start_time = timestamp
                            
                            data['timestamp'].append(timestamp - first_timestamp)
                            data['TP9'].append(float(parts[2]))
                            data['AF7'].append(float(parts[3]))
                            data['AF8'].append(float(parts[4]))
                            data['TP10'].append(float(parts[5]))
                            data['AUX_L'].append(float(parts[6]))
                            data['AUX_R'].append(float(parts[7]))
                            
                    except (ValueError, IndexError) as e:
                        continue
                    except Exception as e:
                        self.logger.warning(f"Unexpected error parsing line: {e}")
                        continue
            
            df_processed = pd.DataFrame(data)
            self.timestamps = df_processed['timestamp'].values
            
            ch_types = ['eeg'] * len(self.CHANNELS['EEG']) + ['misc'] * len(self.CHANNELS['AUX'])
            ch_names = self.CHANNELS['EEG'] + self.CHANNELS['AUX']
            
            info = mne.create_info(
                ch_names=ch_names,
                sfreq=self.CONFIG['SAMPLING_RATE'],
                ch_types=ch_types
            )
            
            data_array = df_processed[ch_names].to_numpy().T
            self.raw_data = mne.io.RawArray(data_array, info)
            
            # Apply filters by default
            self.filtered_data = self.raw_data.copy()
            self.apply_filters()
            
            self.logger.info(f"Successfully loaded {len(self.timestamps)} samples")
            
            # Automatically look for trial info file
            self.load_trial_info()
            
        except Exception as e:
            self.logger.error(f"Error loading MUSE data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        
    def load_trial_info(self, filepath=None):
        """Load trial information from DAT file."""
        if filepath is None:
            # If no filepath provided, try to find a DAT file in the same directory
            if self.input_filepath:
                base_dir = os.path.dirname(self.input_filepath)
                dat_files = [f for f in os.listdir(base_dir) if f.endswith('.dat')]
                if len(dat_files) > 0:
                    filepath = os.path.join(base_dir, dat_files[0])
                    self.logger.info(f"Auto-detected DAT file: {filepath}")
                else:
                    # Ask user to select a file
                    root = tk.Tk()
                    root.withdraw()
                    filepath = filedialog.askopenfilename(
                        title="Select trial information file (.dat)",
                        filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
                    )
                    root.destroy()
                    
        if not filepath:
            self.logger.warning("No trial information file selected")
            return False
            
        self.trial_info_filepath = filepath
        
        try:
            # Read the file content first to inspect
            with open(filepath, 'r') as f:
                content = f.readlines()
            
            # Skip any header lines by finding the first line with typical column names
            start_line = 0
            for i, line in enumerate(content):
                if 'Trial' in line and ('Cond' in line or 'Block' in line or 'Pic' in line):
                    start_line = i
                    break
            
            # Try multiple parsing approaches
            parsing_methods = [
                # Method 1: Tab-separated with header
                lambda: pd.read_csv(filepath, sep='\t', skiprows=start_line),
                
                # Method 2: Space-separated with fixed width
                lambda: pd.read_fwf(filepath, skiprows=start_line),
                
                # Method 3: Space-separated with variable whitespace
                lambda: pd.read_csv(filepath, sep=r'\s+', engine='python', skiprows=start_line),
                
                # Method 4: Try to infer separator
                lambda: pd.read_csv(filepath, sep=None, engine='python', skiprows=start_line)
            ]
            
            # Try each method until one works
            trial_info = None
            last_error = None
            
            for method in parsing_methods:
                try:
                    trial_info = method()
                    # Check if 'Cond' column exists
                    if 'Cond' in trial_info.columns:
                        break
                    # If not, see if there is a similar column
                    potential_columns = [col for col in trial_info.columns 
                                        if 'cond' in col.lower() or 'condition' in col.lower()]
                    if potential_columns:
                        # Rename to 'Cond'
                        trial_info = trial_info.rename(columns={potential_columns[0]: 'Cond'})
                        break
                except Exception as e:
                    last_error = e
                    continue
            
            if trial_info is None or 'Cond' not in trial_info.columns:
                if last_error:
                    raise ValueError(f"Failed to parse trial info file: {str(last_error)}")
                else:
                    raise ValueError("Could not find required 'Cond' column in trial info file")
            
            # Clean up the dataframe
            # Remove any completely empty columns
            trial_info = trial_info.dropna(axis=1, how='all')
            
            # Convert Cond to int if it's numeric
            if trial_info['Cond'].dtype == 'object':
                try:
                    trial_info['Cond'] = trial_info['Cond'].astype(int)
                except:
                    self.logger.warning("Could not convert Cond column to integer")
            
            self.trial_info = trial_info
            self.logger.info(f"Successfully loaded {len(trial_info)} trial entries")
            
            # Log condition counts
            condition_counts = trial_info['Cond'].value_counts()
            for cond, count in condition_counts.items():
                self.logger.info(f"Condition {cond}: {count} trials")
                
            # Create mapping from condition codes to string labels
            # Default mapping is numeric code to string names
            self.condition_map = {
                1: 'highpos',
                2: 'neutral'
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading trial information: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
        
    def detect_photodiode_events(self, data):
        """Detect recurrent pattern of photodiode signal in AUX channel."""
        pd_idx = data.ch_names.index(self.CONFIG['PHOTODIODE']['DETECTION']['CHANNEL'])
        pd_data = data.get_data()[pd_idx]
        SF = self.CONFIG['SAMPLING_RATE']
        found_events = []  # Initialize empty list for events
        
        # Diagnostic information
        self.logger.info(f"Photodiode channel range: {np.min(pd_data):.1f} to {np.max(pd_data):.1f}")
        self.logger.info(f"Standard deviation: {np.std(pd_data):.1f}")
        
        # Calculate and show threshold levels
        high_threshold = 1000
        low_threshold = 100
        oscillation_mean_min = 800
        oscillation_mean_max = 1000
        oscillation_std_min = 200
        oscillation_std_max = 400
        
        self.logger.info(f"Detection parameters:")
        self.logger.info(f"  High threshold: {high_threshold} uV")
        self.logger.info(f"  Low threshold: {low_threshold} uV")
        self.logger.info(f"  Oscillation amplitude range: {oscillation_mean_min}-{oscillation_mean_max} uV")
        self.logger.info(f"  Oscillation variability range: {oscillation_std_min}-{oscillation_std_max} uV")
        
        # Define minimal sample sizes
        min_hold = int(0.025 * SF)  # 25ms minimum hold time
        check_window = int(0.01 * SF)  # 10ms for quick checks
        
        def verify_high_hold(start_idx):
            """Verify signal holds above 1000μV for at least 25ms"""
            if start_idx + min_hold >= len(pd_data):
                return False
            segment = pd_data[start_idx:start_idx+min_hold]
            return np.all(segment > high_threshold)
        
        def verify_zero_hold(start_idx):
            """Verify signal stays near zero for at least 25ms"""
            if start_idx + min_hold >= len(pd_data):
                return False
            segment = pd_data[start_idx:start_idx+min_hold]
            return np.all(segment < low_threshold)
        
        def find_constant_oscillation(start_idx):
            """Find and verify constant oscillation pattern"""
            if start_idx + int(0.5 * SF) >= len(pd_data):
                return None
                    
            # Look ahead up to 0.5s
            for i in range(start_idx, start_idx + int(0.5 * SF), check_window):
                window = pd_data[i:i+int(0.2*SF)]  # 200ms window
                
                if len(window) < 0.1*SF:  # Skip if window is too short
                    continue
                    
                # Check for stable oscillation
                window_mean = np.mean(window)
                window_std = np.std(window)
                peak_count = len(signal.find_peaks(window)[0])
                
                # More detailed oscillation check
                oscillation_valid = (
                    oscillation_mean_min < window_mean < oscillation_mean_max and 
                    oscillation_std_min < window_std < oscillation_std_max and
                    peak_count > 8  # Multiple peaks
                )
                        
                if oscillation_valid:
                    return i
                        
            return None
        
        i = min_hold
        segment_count = 0
        
        # Add search limits for diagnostics
        search_end = min(len(pd_data) - min_hold, int(400 * SF))  # Search first 400 seconds or less
        
        while i < search_end:
            try:
                segment_count += 1
                # Skip frequent logging
                if segment_count % 1000 == 0:
                    self.logger.debug(f"Processed {segment_count} segments, currently at {i/SF:.1f}s")
                    
                # 1. Look for sharp rise above threshold
                if pd_data[i] > high_threshold and verify_high_hold(i):
                    # Found high trigger
                    
                    # 2. Look for transition to oscillation
                    osc_start = find_constant_oscillation(i + min_hold)
                    if osc_start is not None:
                        
                        # 3. Look for end of oscillation block
                        j = osc_start + int(0.2 * SF)
                        while j < min(len(pd_data) - min_hold, osc_start + int(2 * SF)):
                            # Find sharp drop to zero
                            if pd_data[j] < low_threshold and verify_zero_hold(j):
                                # Found complete valid pattern
                                found_events.append(osc_start)
                                self.logger.info(f"Found complete pattern at {osc_start/SF:.3f}s")
                                i = j + min_hold
                                break
                            j += check_window
                        if j >= min(len(pd_data) - min_hold, osc_start + int(2 * SF)):
                            # Couldn't find end of oscillation, move past the oscillation start
                            i = osc_start + int(0.2 * SF)
                        continue
                    
                i += check_window
                    
            except Exception as e:
                self.logger.warning(f"Error at {i}: {str(e)}")
                i += check_window
        
        if not found_events:  # Check if list is empty
            self.logger.warning("No valid photodiode patterns found in the data segment analyzed")
            self.logger.info(f"Search covered {search_end/SF:.1f} seconds of data")
            return np.array([])  # Return empty array
                
        self.logger.info(f"Found {len(found_events)} valid patterns in {search_end/SF:.1f}s of data")
        
        # Add detection rate metrics
        if len(found_events) > 1:
            intervals = np.diff(found_events) / SF
            self.logger.info(f"Average interval between patterns: {np.mean(intervals):.2f}s (std: {np.std(intervals):.2f}s)")
            if len(intervals) > 0:
                self.logger.info(f"Interval range: {np.min(intervals):.2f}s to {np.max(intervals):.2f}s")
        
        return np.array(found_events)
    
    def create_trials(self):
        """Create epochs aligned to oscillation start, independent of trial info."""
        self.logger.info("Creating trials from valid patterns...")
        
        try:
            # Get trigger events - this is the primary source of trial detection
            raw_events = self.detect_photodiode_events(
                self.filtered_data if hasattr(self, 'filtered_data') and self.CONFIG['FILTERS']['EEG']['enabled'] else self.raw_data
            )
            
            if len(raw_events) == 0:
                raise ValueError("No valid patterns found in photodiode signal")
                
            # Store trial timestamps for reference
            self.trial_timestamps = []
            
            # Calculate and log the actual timestamps for each event
            self.logger.info("========== TRIAL ZERO TIMEPOINTS ==========")
            self.logger.info("Trial #  |  Sample Index  |  Time (s)  |  Absolute Timestamp")
            self.logger.info("--------------------------------------------------")
            
            for i, event_sample in enumerate(raw_events):
                # Convert sample index to seconds from start
                event_time_seconds = event_sample / self.CONFIG['SAMPLING_RATE']
                
                # Get the timestamp from timestamps array if available
                timestamp_value = None
                if hasattr(self, 'timestamps') and len(self.timestamps) > 0:
                    if event_sample < len(self.timestamps):
                        timestamp_value = self.timestamps[event_sample]
                    else:
                        # Estimate timestamp if index is out of bounds
                        timestamp_value = event_time_seconds
                else:
                    timestamp_value = event_time_seconds
                
                # Calculate absolute timestamp
                absolute_timestamp = None
                if self.experiment_start_time is not None and timestamp_value is not None:
                    absolute_timestamp = self.experiment_start_time + timestamp_value
                    timestamp_dt = datetime.fromtimestamp(absolute_timestamp)
                    formatted_timestamp = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                else:
                    formatted_timestamp = "N/A"
                
                # Store for future reference
                self.trial_timestamps.append({
                    'trial_num': i+1,
                    'sample_index': int(event_sample),
                    'time_seconds': event_time_seconds,
                    'absolute_timestamp': absolute_timestamp,
                    'formatted_timestamp': formatted_timestamp
                })
                
                # Log the timestamp information
                self.logger.info(f"{i+1:6d}  |  {int(event_sample):12d}  |  {event_time_seconds:9.3f}  |  {formatted_timestamp}")
            
            self.logger.info("==============================================")
            
            # Save the timestamps to a CSV file for reference
            self._save_trial_timestamps_to_csv()
                    
            # Always create basic events first
            events = np.zeros((len(raw_events), 3), dtype=int)
            events[:, 0] = raw_events  # Sample indices
            events[:, 2] = 1          # Default event ID
            
            # Create basic trials first
            self.trials = mne.Epochs(
                self.filtered_data if hasattr(self, 'filtered_data') and self.CONFIG['FILTERS']['EEG']['enabled'] else self.raw_data,
                events,
                event_id={'Stimulus': 1},  # Default event ID mapping
                tmin=self.CONFIG['TRIAL_WINDOW']['START'],
                tmax=self.CONFIG['TRIAL_WINDOW']['END'],
                baseline=self.CONFIG['TRIAL_WINDOW']['BASELINE'],
                preload=True,
                reject=None,
                verbose=False
            )
            
            self.logger.info(f"Created {len(self.trials)} trials based on photodiode detection")
            
            # If trial_info exists, apply condition mapping after trials are created
            if hasattr(self, 'trial_info') and self.trial_info is not None:
                try:
                    self.logger.info("Applying trial condition information...")
                    
                    if len(self.trial_info) >= len(raw_events):
                        # Create condition mapping
                        condition_map = {
                            1: 'highpos',
                            2: 'neutral'
                        }
                        
                        # Create new events with condition codes
                        condition_events = events.copy()
                        condition_events[:, 2] = self.trial_info['Cond'].values[:len(raw_events)]
                        
                        # Update trials with condition information
                        self.trials_event_id = {
                            'highpos': 1,
                            'neutral': 2
                        }
                        
                        # Create new epochs with condition information
                        self.trials = mne.Epochs(
                            self.filtered_data if hasattr(self, 'filtered_data') and self.CONFIG['FILTERS']['EEG']['enabled'] else self.raw_data,
                            condition_events,
                            event_id=self.trials_event_id,
                            tmin=self.CONFIG['TRIAL_WINDOW']['START'],
                            tmax=self.CONFIG['TRIAL_WINDOW']['END'],
                            baseline=self.CONFIG['TRIAL_WINDOW']['BASELINE'],
                            preload=True,
                            reject=None,
                            verbose=False
                        )
                        
                        # Log condition for each trial with all timestamp information
                        self.logger.info("========== TRIALS WITH CONDITIONS ==========")
                        self.logger.info("Trial #  |  Condition  |  Sample Index  |  Time (s)  |  Absolute Timestamp")
                        self.logger.info("----------------------------------------------------------------------------------")

                        for i, timestamp_data in enumerate(self.trial_timestamps):
                            if i < len(self.trial_info):
                                condition = self.trial_info['Cond'].values[i]
                                condition_name = condition_map.get(condition, f"Unknown ({condition})")
                                sample_index = timestamp_data['sample_index']
                                time_seconds = timestamp_data['time_seconds']
                                formatted_timestamp = timestamp_data['formatted_timestamp']
                                
                                self.logger.info(f"{i+1:6d}  |  {condition_name:10s}  |  {sample_index:12d}  |  {time_seconds:9.3f}  |  {formatted_timestamp}")

                        self.logger.info("==============================================")
                        
                        self.logger.info(f"Successfully applied condition information to {len(self.trials)} trials")
                    else:
                        self.logger.warning(f"Trial info length ({len(self.trial_info)}) does not match detected trials ({len(raw_events)})")
                        
                except Exception as e:
                    self.logger.error(f"Error applying trial conditions: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    # Continue with basic trials if condition mapping fails
                    self.logger.info("Continuing with basic trial structure")
                
            # Run this only for initial loading, not when updating filter/ICA changes
            if not hasattr(self, '_initial_save_complete') or not self._initial_save_complete:
                self.save_trials_to_edf()
                self._initial_save_complete = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating trials: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

        
    def _save_trial_timestamps_to_csv(self):
        """Save the trial timestamps to a CSV file for reference."""
        try:
            if not self.trial_timestamps:
                self.logger.warning("No trial timestamps to save")
                return
                
            # Create CSV file path
            base_dir = os.path.dirname(self.input_filepath) if hasattr(self, 'input_filepath') else '.'
            input_name = os.path.basename(self.input_filepath).split('.')[0] if hasattr(self, 'input_filepath') else 'unknown'
            timestamp = datetime.now().strftime("%Y%m%d")
            
            output_file = os.path.join(base_dir, f"{input_name}_trial_timestamps_{timestamp}.csv")
            
            # Create DataFrame from trial_timestamps
            df = pd.DataFrame(self.trial_timestamps)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            self.logger.info(f"Trial timestamps saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving trial timestamps to CSV: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def save_trials_to_edf(self):
        """Save individual trials as EDF files and create a combined EDF file with improved folder checking."""
        try:
            if not hasattr(self, 'trials') or self.trials is None:
                self.logger.warning("No trials available to save")
                return
                
            # Create directory name based on input file
            base_dir = os.path.dirname(self.input_filepath) if hasattr(self, 'input_filepath') else '.'
            input_name = os.path.basename(self.input_filepath).split('.')[0] if hasattr(self, 'input_filepath') else 'unknown'
            
            # Check for existing trial folders for this file
            existing_folders = []
            for folder in os.listdir(base_dir):
                # Look for folders starting with "trials_" and containing the input filename
                if folder.startswith('trials_') and input_name in folder:
                    folder_path = os.path.join(base_dir, folder)
                    if os.path.isdir(folder_path):
                        # Check if folder contains valid trial data
                        if any(f.endswith('.edf') for f in os.listdir(folder_path)):
                            existing_folders.append(folder_path)
            
            if existing_folders:
                self.logger.info(f"Found existing trial folders for {input_name}:")
                for folder in existing_folders:
                    self.logger.info(f"  {folder}")
                self.logger.info("Skipping trial export as data is already saved.")
                return
                
            # If no existing trials found, proceed with saving
            timestamp = datetime.now().strftime("%Y%m%d")
            save_dir = os.path.join(base_dir, f"trials_{input_name}_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)
            
            self.logger.info(f"Saving trials to directory: {save_dir}")
            
            # Get condition labels if available
            if hasattr(self, 'trials') and hasattr(self.trials, 'events') and hasattr(self.trials, 'event_id'):
                event_id_inverse = {v: k for k, v in self.trials.event_id.items()}
                condition_labels = [event_id_inverse.get(evt[2], 'unk') for evt in self.trials.events]
            else:
                condition_labels = ['unk'] * len(self.trials)
            
            # Prepare signals for combined EDF file
            combined_signals = []
            
            # Save each trial
            for i, trial_data in enumerate(self.trials.get_data()):
                signals = []
                
                # Process each channel
                for ch_idx, ch_name in enumerate(self.trials.ch_names):
                    # Get channel data - keep original values but round to 1 decimal place
                    signal_data = np.round(trial_data[ch_idx], 1)
                    
                    # Calculate expected number of samples for 1.5 second duration
                    expected_samples = int(1.5 * self.CONFIG['SAMPLING_RATE'])
                    
                    # Pad or trim signal data to match expected samples
                    if len(signal_data) < expected_samples:
                        # Pad with zeros
                        pad_width = expected_samples - len(signal_data)
                        signal_data = np.pad(signal_data, (0, pad_width), mode='constant')
                    elif len(signal_data) > expected_samples:
                        # Trim excess samples
                        signal_data = signal_data[:expected_samples]
                    
                    edf_signal = EdfSignal(signal_data, sampling_frequency=self.CONFIG['SAMPLING_RATE'], label=ch_name)
                    signals.append(edf_signal)
                    
                    # Create truncated labels for combined file to fit 16-char limit
                    # Format: ChName_T# (e.g., AUX_L_T10)
                    short_cond = condition_labels[i][:3].lower()  # first 3 chars of condition
                    combined_label = f"{ch_name}_T{i+1}"
                    
                    # Ensure label is 16 characters or less
                    combined_signal = EdfSignal(
                        signal_data, 
                        sampling_frequency=self.CONFIG['SAMPLING_RATE'], 
                        label=combined_label
                    )
                    combined_signals.append(combined_signal)
                
                # Create individual trial EDF
                edf = Edf(signals, data_record_duration=1.5)  # Set data record duration to 1.5 seconds
                
                # Include condition in filename if available
                cond_label = condition_labels[i]
                file_path = os.path.join(save_dir, f'trial_{i+1:03d}_{cond_label}.edf')
                
                try:
                    edf.write(file_path)
                    # Only log every 10 trials to reduce console spam
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Saved {i + 1}/{len(self.trials)} trials")
                except Exception as e:
                    self.logger.error(f"Error saving trial {i+1}: {str(e)}")
            
            # Create combined EDF file for all trials
            combined_edf = Edf(combined_signals, data_record_duration=1.5)
            combined_file_path = os.path.join(save_dir, f'{input_name}_all_trials.edf')
            try:
                combined_edf.write(combined_file_path)
                self.logger.info(f"Saved combined EDF file: {combined_file_path}")
            except Exception as e:
                self.logger.error(f"Error saving combined EDF file: {str(e)}")
            
            # Save metadata
            metadata_file = os.path.join(save_dir, 'trial_metadata.txt')
            with open(metadata_file, 'w') as f:
                f.write(f"Original file: {self.input_filepath}\n")
                f.write(f"Number of trials: {len(self.trials)}\n")
                f.write(f"Sampling rate: {self.CONFIG['SAMPLING_RATE']} Hz\n")
                f.write(f"Trial window: {self.CONFIG['TRIAL_WINDOW']['START']}s to {self.CONFIG['TRIAL_WINDOW']['END']}s\n")
                f.write(f"EEG channels: {', '.join(self.CHANNELS['EEG'])}\n")
                f.write(f"AUX channels: {', '.join(self.CHANNELS['AUX'])}\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Data unit: microvolts (uV)\n")
                
                # Add condition summary
                f.write("\nCondition summary:\n")
                condition_counts = pd.Series(condition_labels).value_counts()
                for cond, count in condition_counts.items():
                    f.write(f"{cond}: {count} trials\n")
                    
                # Add trial timestamps
                f.write("\nTrial Zero Timepoints:\n")
                f.write("Trial #  |  Sample Index  |  Time (s)  |  Absolute Timestamp\n")
                f.write("--------------------------------------------------\n")
                
                for ts_data in self.trial_timestamps:
                    f.write(f"{ts_data['trial_num']:6d}  |  {ts_data['sample_index']:12d}  |  {ts_data['time_seconds']:9.3f}  |  {ts_data['formatted_timestamp']}\n")
                    
            self.logger.info(f"Successfully saved {len(self.trials)} trials to {save_dir}")
                
        except Exception as e:
            self.logger.error(f"Error saving trials to EDF: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def apply_filters(self):
        """Apply filters to data and recreate trials"""
        if not hasattr(self, 'raw_data'):
            self.logger.error("No data loaded")
            return False
            
        try:
            # Always create a fresh copy from raw data
            self.filtered_data = self.raw_data.copy()
            applied_any_filter = False
            
            # EEG filtering
            if self.CONFIG['FILTERS']['EEG']['enabled']:
                self.logger.info("Applying EEG filters...")
                
            # Apply high-pass if specified
            if self.CONFIG['FILTERS']['EEG']['highpass'] is not None:
                self.logger.info(f"Applying highpass filter at {self.CONFIG['FILTERS']['EEG']['highpass']} Hz")
                self.filtered_data.filter(
                    l_freq=self.CONFIG['FILTERS']['EEG']['highpass'],
                    h_freq=None,
                    picks=self.CHANNELS['EEG'],
                    method=self.CONFIG['FILTERS']['EEG']['method'],
                    phase=self.CONFIG['FILTERS']['EEG']['phase'],
                    fir_design=self.CONFIG['FILTERS']['EEG']['design'],
                    verbose=True
                )
                applied_any_filter = True
                
            # Apply low-pass if specified
            if self.CONFIG['FILTERS']['EEG']['lowpass'] is not None:
                self.logger.info(f"Applying lowpass filter at {self.CONFIG['FILTERS']['EEG']['lowpass']} Hz")
                self.filtered_data.filter(
                    l_freq=None,
                    h_freq=self.CONFIG['FILTERS']['EEG']['lowpass'],
                    picks=self.CHANNELS['EEG'],
                    method=self.CONFIG['FILTERS']['EEG']['method'],
                    phase=self.CONFIG['FILTERS']['EEG']['phase'],
                    fir_design=self.CONFIG['FILTERS']['EEG']['design'],
                    verbose=True
                )
                applied_any_filter = True
                    
                # Apply notch filter if specified
                if self.CONFIG['FILTERS']['EEG']['notch'] is not None:
                    self.logger.info(f"Applying notch filter at {self.CONFIG['FILTERS']['EEG']['notch']} Hz")
                    self.filtered_data.notch_filter(
                        freqs=self.CONFIG['FILTERS']['EEG']['notch'],
                        picks=self.CHANNELS['EEG'],
                        method=self.CONFIG['FILTERS']['EEG']['method'],
                        phase=self.CONFIG['FILTERS']['EEG']['phase'],
                        verbose=True
                    )
                    applied_any_filter = True
                    
            # AUX filtering
            if self.CONFIG['FILTERS']['AUX']['enabled'] and len(self.CHANNELS['AUX']) > 0:
                self.logger.info("Applying AUX filters...")
                
                # Apply high-pass if specified
                if self.CONFIG['FILTERS']['AUX']['highpass'] is not None:
                    self.filtered_data.filter(
                        l_freq=self.CONFIG['FILTERS']['AUX']['highpass'],
                        h_freq=None,
                        picks=self.CHANNELS['AUX'],
                        method=self.CONFIG['FILTERS']['AUX']['method'],
                        phase=self.CONFIG['FILTERS']['AUX']['phase'],
                        verbose=True
                    )
                    applied_any_filter = True
                    
                # Apply low-pass if specified  
                if self.CONFIG['FILTERS']['AUX']['lowpass'] is not None:
                    self.filtered_data.filter(
                        l_freq=None,
                        h_freq=self.CONFIG['FILTERS']['AUX']['lowpass'],
                        picks=self.CHANNELS['AUX'],
                        method=self.CONFIG['FILTERS']['AUX']['method'],
                        phase=self.CONFIG['FILTERS']['AUX']['phase'],
                        verbose=True
                    )
                    applied_any_filter = True
                    
                # Apply notch filter if specified
                if self.CONFIG['FILTERS']['AUX']['notch'] is not None:
                    self.filtered_data.notch_filter(
                        freqs=self.CONFIG['FILTERS']['AUX']['notch'],
                        picks=self.CHANNELS['AUX'],
                        method=self.CONFIG['FILTERS']['AUX']['method'],
                        phase=self.CONFIG['FILTERS']['AUX']['phase'],
                        verbose=True
                    )
                    applied_any_filter = True
                    
            if applied_any_filter:
                self.logger.info("Filters successfully applied")
            else:
                self.logger.info("No filters were applied (none enabled or specified)")
                
            # Recreate trials if they already exist
            if hasattr(self, 'trials') and self.trials is not None:
                self.create_trials()
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying filters: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Restore original data if filtering fails
            if hasattr(self, 'raw_data'):
                self.filtered_data = self.raw_data.copy()
            return False
        
    def setup_ica(self):
        """Setup ICA with component exclusion"""
        try:
            # Require EEG filters to be enabled
            if not self.CONFIG['FILTERS']['EEG']['enabled']:
                self.logger.warning("ICA requires EEG filters to be enabled first")
                return False
                    
            # Use filtered data
            if hasattr(self, 'filtered_data') and self.filtered_data is not None:
                data_to_use = self.filtered_data
            else:
                self.logger.warning("No filtered data available")
                return False
                    
            # Add standard 10-20 montage for visualization
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                data_to_use.set_montage(montage)
            except Exception as e:
                self.logger.warning(f"Could not set montage: {e}")
            
            # Initialize ICA correctly based on method
            if self.CONFIG['ICA']['method'] == 'infomax':
                self.ica = mne.preprocessing.ICA(
                    n_components=self.CONFIG['ICA']['n_components'],
                    method=self.CONFIG['ICA']['method'],
                    random_state=self.CONFIG['ICA']['random_state'],
                    fit_params=dict(extended=self.CONFIG['ICA']['extended'])
                )
            else:
                # For fastica and other methods that don't use 'extended'
                self.ica = mne.preprocessing.ICA(
                    n_components=self.CONFIG['ICA']['n_components'],
                    method=self.CONFIG['ICA']['method'],
                    random_state=self.CONFIG['ICA']['random_state']
                )
                    
            # Fit ICA on EEG channels only
            self.logger.info(f"Fitting ICA on filtered data...")
            self.ica.fit(data_to_use, picks=self.CHANNELS['EEG'])
            
            self.manually_exclude_ica_components(self.CONFIG['ICA']['exclude'])
            
            # Show custom visualization
            self.visualize_ica_components()
            plt.show(block=False)
            
            return True
                
        except Exception as e:
            self.logger.error(f"Error setting up ICA: {e}")
            self.logger.error(traceback.format_exc())
            return False
        
    def apply_ica(self):
        """Apply ICA to the data and update signals across trials."""
        if self.ica is None or not self.CONFIG['ICA']['enabled']:
            return
                
        try:
            # Use filtered data
            if hasattr(self, 'filtered_data') and self.filtered_data is not None:
                data_to_use = self.filtered_data
            else:
                self.logger.warning("No filtered data available")
                return
                
            # Apply ICA to a copy of the data
            cleaned_data = data_to_use.copy()
            
            # Ensure correct exclusions set
            if not hasattr(self.ica, 'exclude'):
                self.ica.exclude = []
                
            self.logger.info(f"Applying ICA with components {self.ica.exclude} excluded")
            self.ica.apply(cleaned_data)
            
            # Create only one component visualization to prevent duplicates
            self.visualize_ica_components()
            
            # Show ICA overlay (original vs. cleaned)
            overlay_fig = self.ica.plot_overlay(data_to_use, exclude=self.ica.exclude, show=False)
            self.overlay_fig = overlay_fig
            
            # Use comprehensive before/after visualization 
            before_after_fig = self._visualize_ica_before_after(data_to_use)
            self.before_after_fig = before_after_fig
                
            # Apply ICA to filtered data
            self.filtered_data = cleaned_data
            self.ica_applied = True
                
            # Recreate trials to maintain consistency
            if hasattr(self, 'trials'):
                self.create_trials()
            
            # Show all figures without blocking execution
            plt.show(block=False)
                    
        except Exception as e:
            self.logger.error(f"Error applying ICA: {e}")
            self.logger.error(traceback.format_exc())
            
    def manually_exclude_ica_components(self, component_indices):
        """Exclude components selected in the config"""
        if not hasattr(self, 'ica') or self.ica is None:
            self.logger.warning("ICA must be fit before components can be excluded")
            return False
            
        try:
            # Add the specified components to the exclusion list
            if not hasattr(self.ica, 'exclude'):
                self.ica.exclude = []
                
            # Convert to list of integers and remove duplicates
            exclude_list = list(set([int(idx) for idx in component_indices]))
            
            # Validate that all indices are within range
            valid_indices = [idx for idx in exclude_list if 0 <= idx < self.ica.n_components_]
            if len(valid_indices) != len(exclude_list):
                invalid = [idx for idx in exclude_list if idx not in valid_indices]
                self.logger.warning(f"Invalid component indices ignored: {invalid}")
                
            # Set the exclusion list
            self.ica.exclude = valid_indices
            self.logger.info(f"Marked ICA components {valid_indices} for exclusion")
            
            # If ICA has already been applied, reapply it with the new exclusions
            if hasattr(self, 'ica_applied') and self.ica_applied:
                self.logger.info("Reapplying ICA with updated exclusions")
                self.apply_ica()
            
            # Update any open visualizations
            if hasattr(self, 'ica_components_fig') and plt.fignum_exists(self.ica_components_fig.number):
                self.visualize_ica_components()
                
            return True
                
        except Exception as e:
            self.logger.error(f"Error excluding ICA components: {e}")
            self.logger.error(traceback.format_exc())
            return False
            
    def toggle_ica(self, label=None):
        """Toggle ICA with EEG filter requirement"""
        try:
            if not self.CONFIG['FILTERS']['EEG']['enabled']:
                self.logger.warning("ICA requires EEG filters to be enabled first")
                return
                
            self.CONFIG['ICA']['enabled'] = not self.CONFIG['ICA']['enabled']
            
            if self.CONFIG['ICA']['enabled']:
                if self.ica is None:
                    success = self.setup_ica()
                    if not success:
                        self.CONFIG['ICA']['enabled'] = False
                        return
                        
                if not self.ica_applied:
                    self.apply_ica()
            else:
                if self.ica_applied:
                    self.filtered_data = self.raw_data.copy()
                    self.apply_filters()
                    self.ica_applied = False
                    if hasattr(self, 'trials'):
                        self.create_trials()
            
            self.update_plot()
            
        except Exception as e:
            print(f"Error toggling ICA: {str(e)}")
        
    def visualize_ica_components_with_timeseries(self):
        """Visualize continous ICA data as interactive plots"""
        if not hasattr(self, 'ica') or self.ica is None:
            self.logger.warning("ICA must be fit before visualization")
            return None
            
        try:
            # Check if there is a figure open - this prevents duplicates
            if hasattr(self, 'ica_components_fig') and plt.fignum_exists(self.ica_components_fig.number):
                plt.close(self.ica_components_fig)
            
            # Get data and dimensions
            n_components = self.ica.n_components_
            data_to_use = self.filtered_data if hasattr(self, 'filtered_data') else self.raw_data
            
            # Get sources from ICA - these are the component time series
            sources = self.ica.get_sources(data_to_use)
            source_data = sources.get_data()
            
            # Get time values for x-axis
            times = np.arange(source_data.shape[1]) / self.CONFIG['SAMPLING_RATE']
            
            # Create figure with n_components rows
            fig = plt.figure(figsize=(15, 2.5 * n_components))
            fig.suptitle("ICA Components with Time Series", fontsize=16, y=0.98)
            
            # Use fixed width for topography, centered in left column
            left_column_width = 2.0  # Width in inches for left column
            right_column_width = 13.0  # Width in inches for right column
            
            # Calculate ratio for GridSpec
            total_width = left_column_width + right_column_width
            topo_ratio = left_column_width / total_width
            time_ratio = right_column_width / total_width
            
            # Create GridSpec with proper ratio
            gs = GridSpec(n_components, 2, width_ratios=[topo_ratio, time_ratio])
            
            # Get info object for plotting topographies
            info = data_to_use.info
            
            # Store time series axes for later reference
            time_axes = []
            
            # Initialize navigation state
            self.ica_nav = {
                'time_window': [0, min(60, times[-1])],  # Start with 60-second window
                'y_scales': [1.0] * n_components,        # Y-scale factor for each component
                'full_times': times,                     # Store full time array 
                'full_data': source_data,                # Store full source data
                'current_time_idx': 0                    # Current starting index
            }
            
            # Create subplot for each component
            for idx in range(n_components):
                # Create a dedicated topo axes that's centered in the left column
                ax_topo = fig.add_subplot(gs[idx, 0])
                
                # Get component topography map
                component_data = self.ica.get_components()[:, idx]
                
                # Set aspect equal for proper circular shape and center the plot
                ax_topo.set_aspect('equal')
                
                # Remove all axis elements to position topography in center
                ax_topo.axis('off')
                
                # Plot the topography with parameters
                try:
                    mne.viz.plot_topomap(
                        component_data, 
                        info,
                        axes=ax_topo, 
                        show=False,
                        cmap='RdBu_r',
                        outlines='head',
                        contours=6,
                        image_interp='cubic',
                        extrapolate='head'
                    )
                except Exception as e:
                    self.logger.warning(f"Error plotting topography with full params: {e}")
                    # Try with even simpler parameters if first attempt fails
                    try:
                        mne.viz.plot_topomap(
                            component_data, 
                            info,
                            axes=ax_topo, 
                            show=False
                        )
                    except Exception as e2:
                        self.logger.error(f"Failed to plot topography with simple params: {e2}")
                        # Leave an empty plot if no topography found
                        ax_topo.text(0.5, 0.5, f"ICA{idx:03d}", 
                                    ha='center', va='center', transform=ax_topo.transAxes)
                
                # Mark component if excluded - only on topographies, not duplicated on y-axis
                is_excluded = idx in self.ica.exclude if hasattr(self.ica, 'exclude') else False
                title_color = 'red' if is_excluded else 'black'
                fontweight = 'bold' if is_excluded else 'normal'
                
                # Position the ICA label text at the top, centered horizontally
                ax_topo.text(0.5, 1.05, f'ICA{idx:03d}', 
                            color=title_color, 
                            fontweight=fontweight, 
                            ha='center', 
                            va='bottom', 
                            transform=ax_topo.transAxes,
                            fontsize=10)
                
                # Plot time series on the right
                ax_time = fig.add_subplot(gs[idx, 1])
                time_axes.append(ax_time)
                
                # Get visible time window
                t_start, t_end = self.ica_nav['time_window']
                start_idx = np.argmax(times >= t_start) if t_start < times[-1] else 0
                end_idx = np.argmax(times >= t_end) if t_end < times[-1] else len(times)-1
                
                # Extract data for visible window
                visible_times = times[start_idx:end_idx]
                visible_data = source_data[idx, start_idx:end_idx]
                
                # Plot the data for the current component
                line_color = 'red' if is_excluded else 'steelblue'
                ax_time.plot(visible_times, visible_data, color=line_color, linewidth=0.8)
                
                # Style time series plot
                ax_time.set_xlim([t_start, t_end])
                
                # Calculate appropriate y-limits
                if len(visible_data) > 0:
                    # Start with data range in the visible window
                    data_range = visible_data.max() - visible_data.min()
                    if data_range == 0:  # Handle flat line case
                        data_range = 1.0
                    
                    mean_val = np.mean(visible_data)
                    y_min = mean_val - (data_range/2) * self.ica_nav['y_scales'][idx] * 1.1
                    y_max = mean_val + (data_range/2) * self.ica_nav['y_scales'][idx] * 1.1
                    ax_time.set_ylim([y_min, y_max])
                
                # Remove duplicated ICA label, use empty string for ylabel
                ax_time.set_ylabel("", rotation=0, ha='right', va='center')
                
                # Remove top and right spines
                ax_time.spines['top'].set_visible(False)
                ax_time.spines['right'].set_visible(False)
                
                # Only show x-axis for the last component
                if idx < n_components - 1:
                    ax_time.set_xticklabels([])
                else:
                    ax_time.set_xlabel('Time (s)')
            
            # Add time window slider at the bottom - with virtually unlimited maximum
            slider_ax = plt.axes([0.2, 0.01, 0.65, 0.02])
            
            # Calculate max time for slider - use the entire recording length
            max_window = times[-1]  # No arbitrary limit - use full recording length
            
            time_slider = plt.Slider(
                slider_ax, 'Window Width (s)', 
                1, max_window, 
                valinit=self.ica_nav['time_window'][1] - self.ica_nav['time_window'][0],
                valstep=None  # Allow continuous values
            )
            
            # Store references for interactivity
            self.ica_nav['time_axes'] = time_axes
            self.ica_nav['time_slider'] = time_slider
            
            # Connect the slider to the update function (defined at class level)
            time_slider.on_changed(self._update_time_window_slider)
            
            # Connect keyboard events for navigation
            self.ica_key_event_id = fig.canvas.mpl_connect('key_press_event', self._on_ica_key_press)
            
            # Connect mouse wheel events for zooming the y-axis
            self.ica_scroll_event_id = fig.canvas.mpl_connect('scroll_event', self._on_ica_scroll)
            
            # Adjust layout
            plt.tight_layout()
            fig.subplots_adjust(top=0.95, bottom=0.07, hspace=0.4)
            
            # Store figure reference
            self.ica_components_fig = fig
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error in ICA visualization: {e}")
            self.logger.error(traceback.format_exc())
            return None
        
    def _visualize_component_scores(self, scores):
        """Visualize component scores with proper discrete x-axis."""
        try:
            # Create a new figure
            fig = plt.figure(figsize=(10, 6))
            
            # Case-check for no component scores
            if len(scores) == 0:
                self.logger.warning("No scores to visualize")
                plt.text(0.5, 0.5, "No component scores available", 
                    ha='center', va='center', transform=plt.gca().transAxes)
                plt.tight_layout()
                self.score_fig = fig
                return fig
            
            # Ensure scores is a proper numpy array of floats
            scores_array = np.zeros(len(scores))
            for i in range(len(scores)):
                try:
                    if hasattr(scores[i], 'item'):
                        scores_array[i] = float(scores[i].item())
                    elif isinstance(scores[i], (list, tuple)) and len(scores[i]) > 0:
                        scores_array[i] = float(scores[i][0])
                    else:
                        scores_array[i] = float(scores[i])
                except (TypeError, ValueError, IndexError):
                    scores_array[i] = 0.0
            
            # Use integer component indices for x-axis
            component_indices = list(range(len(scores_array)))
            
            # Create bar plot with specific x positions
            plt.bar(component_indices, scores_array, color='blue', width=0.7)
            
            # Highlight excluded components
            if hasattr(self.ica, 'exclude') and self.ica.exclude:
                for idx in self.ica.exclude:
                    if idx < len(scores_array):
                        plt.bar(int(idx), scores_array[int(idx)], color='red', width=0.7)
            
            # Set x-ticks explicitly to component indices
            plt.xticks(component_indices, [f'ICA{i:03d}' for i in component_indices])
            plt.tick_params(axis='x', rotation=45 if len(component_indices) > 10 else 0)
            
            # Label axes
            plt.xlabel('ICA Component')
            plt.ylabel('Correlation Score')
            plt.title('ICA Component Correlation with EOG')
            plt.grid(True, alpha=0.3)
            
            # Add threshold line if applicable
            if len(scores_array) > 0 and not np.all(np.isnan(scores_array)):
                threshold_val = np.percentile(scores_array[~np.isnan(scores_array)], 75)
                threshold = float(threshold_val)
                plt.axhline(y=threshold, color='r', linestyle='--', 
                        label=f'Threshold (75th percentile = {threshold:.3f})')
                plt.legend()
                
            plt.tight_layout()
            
            # Store figure reference
            self.score_fig = fig
            
            # Show figure without blocking
            plt.show(block=False)
            
            # Return figure
            return fig
            
        except Exception as e:
            self.logger.warning(f"Error visualizing component scores: {e}")
            self.logger.error(traceback.format_exc())
            return None
            
    def visualize_ica_components(self):
        """Create topography maps for each IC"""
        if not hasattr(self, 'ica') or self.ica is None:
            self.logger.warning("ICA must be fit before visualization")
            return
                
        try:
            # Create custom visualization
            fig = self.visualize_ica_components_with_timeseries()
            
            # Store figure reference to prevent garbage collection
            self.ica_components_fig = fig
            
            # Show figure without blocking
            plt.show(block=False)
            
            return fig
                    
        except Exception as e:
            self.logger.error(f"Error in ICA visualization: {e}")
            self.logger.error(traceback.format_exc())      
            
    def _visualize_ica_before_after(self, data):
        """Create a comprehensive before/after visualization for each channel."""
        try:
            # Get channel data
            eeg_data = data.get_data(picks=self.CHANNELS['EEG'])
            
            # Apply ICA cleaning to get "after" data
            cleaned_data = data.copy()
            cleaned_data = self.ica.apply(cleaned_data)
            cleaned_eeg = cleaned_data.get_data(picks=self.CHANNELS['EEG'])
            
            # Create figure
            fig = plt.figure(figsize=(14, 10))
            plt.suptitle(f"ICA Artifact Removal ({len(self.ica.exclude)} components excluded)", 
                        fontsize=16, y=0.98)
            
            # Compute time vector with potential downsampling
            sfreq = data.info['sfreq']
            n_samples = eeg_data.shape[1]
            times = np.arange(n_samples) / sfreq
            
            # Downsample for plotting if needed
            if n_samples > 10000:
                downsample_factor = n_samples // 10000
                times = times[::downsample_factor]
                eeg_data = eeg_data[:, ::downsample_factor]
                cleaned_eeg = cleaned_eeg[:, ::downsample_factor]
            
            # Plot each channel
            n_channels = len(self.CHANNELS['EEG'])
            for i, ch_name in enumerate(self.CHANNELS['EEG']):
                ax = plt.subplot(n_channels + 1, 1, i + 1)
                ch_idx = self.CHANNELS['EEG'].index(ch_name)
                
                # Plot before and after with distinct colors
                ax.plot(times, eeg_data[ch_idx], color='#E63946', linewidth=1, label='Before ICA', alpha=0.8)
                ax.plot(times, cleaned_eeg[ch_idx], color='#1D3557', linewidth=1, label='After ICA', alpha=0.9)
                
                # Style the plot
                ax.set_title(f"Channel {ch_name}", fontsize=12, loc='left')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Only add legend to first plot
                if i == 0:
                    ax.legend(loc='upper right')
                    
                # Only show x-axis for bottom plot
                if i < n_channels - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
            
            # Add a difference plot at the bottom
            ax = plt.subplot(n_channels + 1, 1, n_channels + 1)
            
            # Calculate the average absolute difference across channels
            diff_data = np.mean(np.abs(eeg_data - cleaned_eeg), axis=0)
            ax.plot(times, diff_data, color='#6a0dad', linewidth=1.2)
            ax.set_title('Average Absolute Difference (Artifact Strength)', fontsize=12, loc='left')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('Time (s)')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Store the figure reference
            self.before_after_fig = fig
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating before/after visualization: {e}")
            self.logger.error(traceback.format_exc())
            return None
        
    def _on_ica_scroll(self, event):
        """Handle mouse wheel scrolling to zoom y-axis of the component under cursor."""
        if not hasattr(self, 'ica_nav') or not event.inaxes:
            return
            
        # Check if the scroll event happened over a time series axis
        if event.inaxes in self.ica_nav['time_axes']:
            # Get the component index
            comp_idx = self.ica_nav['time_axes'].index(event.inaxes)
            
            # Get current scale factor
            scale = self.ica_nav['y_scales'][comp_idx]
            
            # Adjust scale based on scroll direction
            if event.button == 'up':
                # Zoom in (make signal appear larger)
                self.ica_nav['y_scales'][comp_idx] = scale * 0.8
            elif event.button == 'down':
                # Zoom out (make signal appear smaller)
                self.ica_nav['y_scales'][comp_idx] = scale * 1.2
                
            # Update only the affected component
            self._update_single_component_display(comp_idx)
            
    def _update_single_component_display(self, comp_idx):
        """Update a single component's display after y-scale change."""
        if not hasattr(self, 'ica_nav') or comp_idx >= len(self.ica_nav['time_axes']):
            return
            
        # Get the data
        times = self.ica_nav['full_times']
        source_data = self.ica_nav['full_data']
        t_start, t_end = self.ica_nav['time_window']
        
        # Get the axis to update
        ax = self.ica_nav['time_axes'][comp_idx]
        
        # Find indices for the visible time window
        start_idx = np.argmax(times >= t_start) if t_start < times[-1] else 0
        end_idx = np.argmax(times >= t_end) if t_end < times[-1] else len(times)-1
        
        # Extract data for visible window
        visible_data = source_data[comp_idx, start_idx:end_idx]
        
        # Update y-limits based on new scale factor
        if len(visible_data) > 0:
            data_range = visible_data.max() - visible_data.min()
            if data_range == 0:  # Handle flat line case
                data_range = 1.0
                
            mean_val = np.mean(visible_data)
            y_min = mean_val - (data_range/2) * self.ica_nav['y_scales'][comp_idx] * 1.1
            y_max = mean_val + (data_range/2) * self.ica_nav['y_scales'][comp_idx] * 1.1
            ax.set_ylim([y_min, y_max])
    
        # Redraw just this axis for efficiency
        if hasattr(self, 'ica_components_fig'):
            self.ica_components_fig.canvas.draw_idle()
            
    def _update_time_window_slider(self, val):
        """Update the time window based on slider value."""
        if not hasattr(self, 'ica_nav'):
            return
            
        times = self.ica_nav['full_times']
        window_width = val
        current_center = np.mean(self.ica_nav['time_window'])
        
        # Ensure the window stays within bounds
        new_start = max(0, current_center - window_width/2)
        new_end = min(times[-1], new_start + window_width)
        
        # Adjust start if end hit the boundary
        if new_end == times[-1]:
            new_start = max(0, new_end - window_width)
        
        self.ica_nav['time_window'] = [new_start, new_end]
        self._update_ica_time_display()
        
    def _on_ica_key_press(self, event):
        """Handle keyboard navigation in the ICA visualization."""
        if not hasattr(self, 'ica_nav'):
            return
            
        times = self.ica_nav['full_times']
        t_start, t_end = self.ica_nav['time_window']
        window_width = t_end - t_start
        
        # Get the focused component if the event happened on a specific axis
        focused_comp = None
        if hasattr(event, 'inaxes') and event.inaxes in self.ica_nav['time_axes']:
            focused_comp = self.ica_nav['time_axes'].index(event.inaxes)
        
        # Handle keyboard input
        if event.key == 'right':
            # Move window forward
            step = window_width * 0.2  # Move by 20% of window
            new_start = min(times[-1] - window_width, t_start + step)
            new_end = new_start + window_width
            self.ica_nav['time_window'] = [new_start, new_end]
            self._update_ica_time_display()
        
        elif event.key == 'left':
            # Move window backward
            step = window_width * 0.2  # Move by 20% of window
            new_start = max(0, t_start - step)
            new_end = new_start + window_width
            self.ica_nav['time_window'] = [new_start, new_end]
            self._update_ica_time_display()
        
        elif event.key == 'up':
            # Zoom in y-axis (reduce scale)
            if focused_comp is not None:
                # Scale only the focused component
                self.ica_nav['y_scales'][focused_comp] *= 0.8
                self._update_single_component_display(focused_comp)
            else:
                # Scale all components
                self.ica_nav['y_scales'] = [scale * 0.8 for scale in self.ica_nav['y_scales']]
                self._update_ica_time_display()
        
        elif event.key == 'down':
            # Zoom out y-axis (increase scale)
            if focused_comp is not None:
                # Scale only the focused component
                self.ica_nav['y_scales'][focused_comp] *= 1.2
                self._update_single_component_display(focused_comp)
            else:
                # Scale all components
                self.ica_nav['y_scales'] = [scale * 1.2 for scale in self.ica_nav['y_scales']]
                self._update_ica_time_display()
                
    def _update_ica_time_display(self):
        """Update the ICA time series display after navigation."""
        if not hasattr(self, 'ica_nav'):
            return
            
        times = self.ica_nav['full_times']
        source_data = self.ica_nav['full_data']
        time_axes = self.ica_nav['time_axes']
        t_start, t_end = self.ica_nav['time_window']
        
        # Find indices for the visible time window
        start_idx = np.argmax(times >= t_start) if t_start < times[-1] else 0
        end_idx = np.argmax(times >= t_end) if t_end < times[-1] else len(times)-1
        
        # Update each time series plot
        for idx, ax in enumerate(time_axes):
            # Clear the current axis
            ax.clear()
            
            # Extract data for visible window
            visible_times = times[start_idx:end_idx]
            if idx < source_data.shape[0]:
                visible_data = source_data[idx, start_idx:end_idx]
                
                # Plot the data
                is_excluded = idx in self.ica.exclude if hasattr(self.ica, 'exclude') else False
                line_color = 'red' if is_excluded else 'steelblue'
                ax.plot(visible_times, visible_data, color=line_color, linewidth=0.8)
                
                # Set x-limits to the current time window
                ax.set_xlim([t_start, t_end])
                
                # Calculate appropriate y-limits based on scale factor
                if len(visible_data) > 0:
                    data_range = visible_data.max() - visible_data.min()
                    if data_range == 0:  # Handle flat line case
                        data_range = 1.0
                    
                    mean_val = np.mean(visible_data)
                    y_min = mean_val - (data_range/2) * self.ica_nav['y_scales'][idx] * 1.1
                    y_max = mean_val + (data_range/2) * self.ica_nav['y_scales'][idx] * 1.1
                    ax.set_ylim([y_min, y_max])
            
            # Style the plot excluding the y-axis label to avoid duplication
            ax.set_ylabel("", rotation=0, ha='right', va='center')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Only show x-axis for the last component
            if idx < len(time_axes) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (s)')
        
        # Redraw the figure
        if hasattr(self, 'ica_components_fig'):
            self.ica_components_fig.canvas.draw_idle()

    def setup_gui(self):
        """Setup the visualization GUI."""
        self.fig = plt.figure(figsize=self.GUI['LAYOUT']['figure_size'])
        self.fig.patch.set_facecolor(self.CONFIG['PLOT']['STYLES']['background'])
        
        gs = GridSpec(len(self.CHANNELS['EEG'] + self.CHANNELS['AUX']), 1)
        self.axes = {}
        
        for i, ch in enumerate(self.CHANNELS['EEG'] + self.CHANNELS['AUX']):
            ax = self.fig.add_subplot(gs[i, 0])
            ax.set_facecolor('white')
            ax.grid(True, alpha=self.CONFIG['PLOT']['STYLES']['grid_alpha'])
            
            self.axes[ch] = ax
            ax.set_ylabel(
                f'{ch}\n(μV)',
                color=self.CONFIG['PLOT']['COLORS'][ch],
                labelpad=self.CONFIG['PLOT']['STYLES']['axis_label_pad']
            )
            
            if i == len(self.CHANNELS['EEG'] + self.CHANNELS['AUX']) - 1:
                ax.set_xlabel('Time (ms)')
        
        self.setup_controls()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.update_plot()

    def setup_controls(self):
        """Setup GUI controls with ICA toggle unchecked by default."""
        ctrl_cfg = self.GUI['CONTROLS']
        
        # Filter toggles with correct initial states
        filter_pos = ctrl_cfg['filter_toggles']
        filter_ax = plt.axes([
            filter_pos['x'], filter_pos['y'],
            filter_pos['width'], filter_pos['height']
        ])
        
        # Add ICA to the filter check buttons - now disabled by default
        self.filter_checks = CheckButtons(
            filter_ax,
            ['EEG Filters', 'AUX Filters', 'ICA'],
            [self.CONFIG['FILTERS']['EEG']['enabled'],
            self.CONFIG['FILTERS']['AUX']['enabled'],
            False]  # ICA always starts disabled
        )
        
        # Scale slider
        slider_pos = ctrl_cfg['scale_slider']
        scale_ax = plt.axes([
            slider_pos['x'], slider_pos['y'],
            slider_pos['width'], slider_pos['height']
        ])
        self.scale_slider = Slider(
            scale_ax, 'Scale (μV)',
            self.CONFIG['PLOT']['SCALES']['EEG']['min'],
            self.CONFIG['PLOT']['SCALES']['EEG']['max'],
            valinit=self.CONFIG['PLOT']['SCALES']['EEG']['default']
        )
        
        # Navigation buttons
        self.prev_button = Button(
            plt.axes([ctrl_cfg['prev_button']['x'], ctrl_cfg['prev_button']['y'],
                    ctrl_cfg['prev_button']['width'], ctrl_cfg['prev_button']['height']]),
            '◀ Previous'
        )
        self.next_button = Button(
            plt.axes([ctrl_cfg['next_button']['x'], ctrl_cfg['next_button']['y'],
                    ctrl_cfg['next_button']['width'], ctrl_cfg['next_button']['height']]),
            'Next ▶'
        )
        self.avg_button = Button(
            plt.axes([ctrl_cfg['avg_button']['x'], ctrl_cfg['avg_button']['y'],
                    ctrl_cfg['avg_button']['width'], ctrl_cfg['avg_button']['height']]),
            'Show Averages'
        )
        
        # Connect callbacks
        self.filter_checks.on_clicked(self.toggle_filters)
        self.scale_slider.on_changed(self.update_plot)
        self.prev_button.on_clicked(lambda x: self.navigate_trials('prev'))
        self.next_button.on_clicked(lambda x: self.navigate_trials('next'))
        self.avg_button.on_clicked(lambda x: self.setup_averages_view())
        
        # Add power button
        self.power_button = Button(
            plt.axes([0.88, 0.1, 0.1, 0.04]),
            'Show Power'
        )
        self.power_button.on_clicked(lambda x: self.calculate_and_plot_global_power())
        
        self.psd_button = Button(
            plt.axes([0.88, 0.05, 0.1, 0.04]),
            'Show PSD'
        )
        self.psd_button.on_clicked(lambda x: self.calculate_and_plot_psd())
        
    def update_plot(self):
        """Update the plot with current trial data."""
        if self.trials is None or len(self.trials) == 0:
            self.logger.warning("No trials available to plot")
            for ax in self.axes.values():
                ax.clear()
                ax.text(0.5, 0.5, 'No valid trials available',
                    ha='center', va='center')
            plt.draw()
            return
                    
        try:
            # Get current trial data
            current_data = self.trials.get_data()[self.current_trial]
            times = self.trials.times * 1000  # Convert to milliseconds
            
            # Update title
            trial_num = self.current_trial + 1
            title = f"Trial {trial_num}/{len(self.trials)}"
            self.fig.suptitle(title, y=0.98)
            
            # Default stimulus window - at time 0 (original trigger point)
            stim_start = -1800  # Start at -1800ms (shifted trigger point)
            stim_end = -1300    # End 500ms after trigger
            
            # Update each channel plot
            for ch in self.CHANNELS['EEG'] + self.CHANNELS['AUX']:
                ax = self.axes[ch]
                ax.clear()
                
                ch_idx = self.trials.ch_names.index(ch)
                ch_data = current_data[ch_idx]
                
                # Skip if all data is NaN
                if np.all(np.isnan(ch_data)):
                    ax.text(0.5, 0.5, f'No valid data for {ch}',
                        ha='center', va='center')
                    continue
                
                # Plot stimulus window
                ax.axvspan(stim_start, stim_end,
                        alpha=self.CONFIG['VISUALIZATION']['STIM_WINDOW_ALPHA'],
                        color=self.CONFIG['VISUALIZATION']['STIM_WINDOW_COLOR'],
                        zorder=1)
                
                # Plot the data
                valid_data = ch_data[~np.isnan(ch_data)]
                if len(valid_data) > 0:
                    ax.plot(times, ch_data,
                        color=self.CONFIG['PLOT']['COLORS'][ch],
                        linewidth=self.CONFIG['PLOT']['STYLES']['line_width'],
                        zorder=2)
                    
                    # Add vertical line at original trigger (now at -1800ms)
                    ax.axvline(x=-1800, color='r', linestyle='--',
                            alpha=self.CONFIG['PLOT']['STYLES']['marker_alpha'])
                    
                    # Set axis limits
                    if ch in self.CHANNELS['EEG']:
                        data_range = self.channel_scales[ch]
                    else:  # AUX
                        data_range = self.channel_scales[ch]
                    
                    # Calculate valid min/max for y-axis limits
                    valid_min = np.nanmin(ch_data)
                    valid_max = np.nanmax(ch_data)
                    if not (np.isnan(valid_min) or np.isnan(valid_max)):
                        y_min = valid_min - data_range * 0.1
                        y_max = valid_max + data_range * 0.1
                        ax.set_ylim(y_min, y_max)
                    else:
                        # Fallback limits if data is invalid
                        ax.set_ylim(-data_range/2, data_range/2)
                
                # Set x-axis limits with adjusted time window
                buffer = (times[-1] - times[0]) * 0.01  # 1% buffer  
                ax.set_xlim(times[0] - buffer, times[-1] + buffer)
                
                # Style
                ax.grid(True, alpha=self.CONFIG['PLOT']['STYLES']['grid_alpha'])
                ax.set_ylabel(
                    f'{ch}\n(μV)',
                    color=self.CONFIG['PLOT']['COLORS'][ch],
                    labelpad=self.CONFIG['PLOT']['STYLES']['axis_label_pad']
                )
                
                if ch == list(self.axes.keys())[-1]:
                    ax.set_xlabel('Time (ms)')
                        
            plt.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating plot: {str(e)}")
            self.logger.error(traceback.format_exc())

    def calculate_and_plot_psd(self):
        """Create a figure showing PSD with toggleable electrodes."""
        if not hasattr(self, 'trials') or self.trials is None:
            self.logger.warning("No trials available for PSD calculation")
            return
            
        # Create new figure with control panel space
        self.fig_psd = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 1)
        self.ax_psd = self.fig_psd.add_subplot(gs[0])
        
        # Add control panel
        ctrl_width = 0.2
        main_width = 0.98 - ctrl_width
        
        # Setup checkboxes
        check_height = 0.15
        current_y = 0.8
        self.fig_psd.text(
            main_width + 0.01, current_y,
            'Select Electrodes',
            horizontalalignment='left'
        )
        
        # Add electrode checkboxes
        self.psd_electrode_checks = CheckButtons(
            plt.axes([main_width + 0.01, current_y - check_height, ctrl_width-0.02, check_height]),
            ['AF7', 'AF8', 'TP9', 'TP10'],
            [self.electrode_states[ch] for ch in ['AF7', 'AF8', 'TP9', 'TP10']]
        )
        
        # Connect callback
        self.psd_electrode_checks.on_clicked(self.update_psd_plot)
        
        # Adjust layout
        self.fig_psd.subplots_adjust(right=main_width)
        
        # Initial plot
        self.plot_psd_data()
        plt.show()

    def plot_psd_data(self):
        """Update PSD plot based on selected electrodes."""
        try:
            self.ax_psd.clear()
            
            # Calculate frequency parameters
            data = self.trials.get_data()
            nfft = int(2 ** np.ceil(np.log2(data.shape[-1])))
            freqs = fftfreq(nfft, 1/self.CONFIG['SAMPLING_RATE'])
            positive_freqs = freqs[freqs >= 0]  # Only positive frequencies
            
            # Color scheme
            colors = {
                'AF7': '#4169E1',  # Royal Blue
                'AF8': '#4169E1',  # Royal Blue
                'TP9': '#32CD32',  # Lime Green
                'TP10': '#32CD32', # Lime Green
                'AF7/AF8': '#4169E1',
                'TP9/TP10': '#32CD32'
            }
            
            pairs = [('AF7', 'AF8'), ('TP9', 'TP10')]
            plotted_in_pair = set()
            
            # First plot paired electrodes
            for pair in pairs:
                if all(self.electrode_states[ch] for ch in pair):
                    pair_data = []
                    for ch in pair:
                        ch_idx = self.trials.ch_names.index(ch)
                        ch_data = self.trials.get_data()[:, ch_idx, :]
                        
                        # Calculate PSD for each trial
                        trial_psds = []
                        for trial in ch_data:
                            # Apply Hanning window
                            windowed = trial * signal.windows.hann(len(trial))
                            # Calculate FFT
                            fft_vals = fft(windowed, n=nfft)
                            # Calculate power
                            psd = np.abs(fft_vals[:len(positive_freqs)]) ** 2
                            trial_psds.append(psd)
                        
                        pair_data.append(np.array(trial_psds))
                    
                    # Average across channels then trials
                    avg_data = np.mean(np.stack(pair_data), axis=0)
                    psd_mean = np.mean(avg_data, axis=0)
                    psd_sem = sem(avg_data, axis=0, nan_policy='omit')
                    
                    # Plot
                    pair_name = f"{pair[0]}/{pair[1]}"
                    self.ax_psd.plot(positive_freqs, psd_mean,
                                color=colors[pair_name],
                                label=pair_name,
                                linewidth=2)
                    self.ax_psd.fill_between(positive_freqs,
                                        psd_mean - psd_sem,
                                        psd_mean + psd_sem,
                                        color=colors[pair_name],
                                        alpha=0.2)
                    plotted_in_pair.update(pair)
            
            # Then plot individual electrodes not in pairs
            for ch in self.electrode_states:
                if self.electrode_states[ch] and ch not in plotted_in_pair:
                    ch_idx = self.trials.ch_names.index(ch)
                    ch_data = self.trials.get_data()[:, ch_idx, :]
                    
                    # Calculate PSD for each trial
                    trial_psds = []
                    for trial in ch_data:
                        windowed = trial * signal.windows.hann(len(trial))
                        fft_vals = fft(windowed, n=nfft)
                        psd = np.abs(fft_vals[:len(positive_freqs)]) ** 2
                        trial_psds.append(psd)
                    
                    # Calculate mean and SEM
                    psd_mean = np.mean(trial_psds, axis=0)
                    psd_sem = sem(trial_psds, axis=0, nan_policy='omit')
                    
                    # Plot
                    self.ax_psd.plot(positive_freqs, psd_mean,
                                color=colors[ch],
                                label=ch,
                                linewidth=2,
                                linestyle='--' if 'TP' in ch else '-')
                    self.ax_psd.fill_between(positive_freqs,
                                        psd_mean - psd_sem,
                                        psd_mean + psd_sem,
                                        color=colors[ch],
                                        alpha=0.2)
            
            # Style plot
            self.ax_psd.set_xlim(0, 50)  # Show frequencies up to 50 Hz
            self.ax_psd.set_yscale('log')  # Use log scale for better visualization
            self.ax_psd.grid(True, alpha=0.15)
            self.ax_psd.spines['top'].set_visible(False)
            self.ax_psd.spines['right'].set_visible(False)
            
            # Set labels
            self.ax_psd.set_xlabel('Frequency (Hz)')
            self.ax_psd.set_ylabel('Power (μV²/Hz)')
            self.ax_psd.set_title(f'Power Spectral Density (n={len(self.trials)} trials)')
            
            if len(self.ax_psd.lines) > 0:
                self.ax_psd.legend(loc='upper right', frameon=False)
            
            self.fig_psd.canvas.draw_idle()
            
        except Exception as e:
            self.logger.error(f"Error plotting PSD data: {str(e)}")
            self.logger.error(traceback.format_exc())

    def update_psd_plot(self, label):
        """Handle electrode toggling for PSD plot."""
        self.electrode_states[label] = not self.electrode_states[label]
        self.plot_psd_data()

    def calculate_and_plot_global_power(self):
        """Create a figure showing global power with toggleable electrodes."""
        if not hasattr(self, 'trials') or self.trials is None:
            self.logger.warning("No trials available for power calculation")
            return
            
        # Create new figure with control panel space
        self.fig_power = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 1)
        self.ax_power = self.fig_power.add_subplot(gs[0])
        
        # Add control panel
        ctrl_width = 0.2
        main_width = 0.98 - ctrl_width
        
        # Setup checkboxes
        check_height = 0.15
        current_y = 0.8
        self.fig_power.text(
            main_width + 0.01, current_y,
            'Select Electrodes',
            horizontalalignment='left'
        )
        
        # Add electrode checkboxes
        self.power_electrode_checks = CheckButtons(
            plt.axes([main_width + 0.01, current_y - check_height, ctrl_width-0.02, check_height]),
            ['AF7', 'AF8', 'TP9', 'TP10'],
            [self.electrode_states[ch] for ch in ['AF7', 'AF8', 'TP9', 'TP10']]
        )
        
        # Connect callback
        self.power_electrode_checks.on_clicked(self.update_power_plot)
        
        # Adjust layout
        self.fig_power.subplots_adjust(right=main_width)
        
        # Initial plot
        self.plot_power_data()
        plt.show()
        
    def plot_power_data(self):
        """Update power plot based on selected electrodes."""
        try:
            self.ax_power.clear()
            times = self.trials.times * 1000
            
            pairs = [('AF7', 'AF8'), ('TP9', 'TP10')]
            colors = {
                'AF7/AF8': '#4169E1',  # Royal Blue
                'TP9/TP10': '#32CD32',  # Lime Green
                'AF7': '#4169E1',       # Royal Blue
                'AF8': '#4169E1',       # Royal Blue
                'TP9': '#32CD32',       # Lime Green
                'TP10': '#32CD32'       # Lime Green
            }
            
            plotted_in_pair = set()
            
            # First plot paired electrodes
            for pair in pairs:
                if all(self.electrode_states[ch] for ch in pair):
                    pair_data = []
                    for ch in pair:
                        ch_idx = self.trials.ch_names.index(ch)
                        ch_data = self.trials.get_data()[:, ch_idx, :]
                        
                        # Baseline correction
                        baseline_window = np.logical_and(times >= -200, times <= 0)
                        baseline_mean = np.nanmean(ch_data[:, baseline_window], axis=1)
                        ch_data = ch_data - baseline_mean[:, np.newaxis]
                        pair_data.append(ch_data)
                    
                    # Average pair data and calculate power
                    avg_data = np.mean(np.stack(pair_data), axis=0)
                    power = avg_data ** 2
                    
                    # Calculate mean and SEM
                    power_mean = np.nanmean(power, axis=0)
                    power_sem = sem(power, axis=0, nan_policy='omit')
                    
                    # Plot
                    pair_name = f"{pair[0]}/{pair[1]}"
                    self.ax_power.plot(times, power_mean, 
                                    color=colors[pair_name],
                                    label=pair_name,
                                    linewidth=2)
                    self.ax_power.fill_between(times, 
                                            power_mean - power_sem,
                                            power_mean + power_sem,
                                            color=colors[pair_name],
                                            alpha=0.2)
                    plotted_in_pair.update(pair)
            
            # Then plot individual electrodes not in pairs
            for ch in self.electrode_states:
                if self.electrode_states[ch] and ch not in plotted_in_pair:
                    ch_idx = self.trials.ch_names.index(ch)
                    ch_data = self.trials.get_data()[:, ch_idx, :]
                    
                    # Baseline correction
                    baseline_window = self.CONFIG['TRIAL_WINDOW']['BASELINE']
                    baseline_mean = np.nanmean(ch_data[:, baseline_window], axis=1)
                    ch_data = ch_data - baseline_mean[:, np.newaxis]
                    
                    # Calculate power
                    power = ch_data ** 2
                    
                    # Calculate mean and SEM
                    power_mean = np.nanmean(power, axis=0)
                    power_sem = sem(power, axis=0, nan_policy='omit')
                    
                    # Plot
                    self.ax_power.plot(times, power_mean,
                                    color=colors[ch],
                                    label=ch,
                                    linewidth=2,
                                    linestyle='--')
                    self.ax_power.fill_between(times,
                                            power_mean - power_sem,
                                            power_mean + power_sem,
                                            color=colors[ch],
                                            alpha=0.2)
            
            # Style plot
            self.ax_power.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            self.ax_power.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            self.ax_power.grid(True, alpha=0.15)
            self.ax_power.spines['top'].set_visible(False)
            self.ax_power.spines['right'].set_visible(False)
            
            # Set labels
            self.ax_power.set_xlabel('Time (ms)')
            self.ax_power.set_ylabel('Global Power (μV²)')
            self.ax_power.set_title(f'Global Power Average (n={len(self.trials)} trials)')
            
            if len(self.ax_power.lines) > 0:
                self.ax_power.legend(loc='upper right', frameon=False)
                
                # Set y-limits based on data
                ydata = np.concatenate([line.get_ydata() for line in self.ax_power.lines])
                ymax = np.nanmax(ydata) * 1.1
                ymin = np.nanmin(ydata) * 0.9
                if ymin > 0:
                    ymin = 0
                self.ax_power.set_ylim(ymin, ymax)
            
            self.fig_power.canvas.draw_idle()
            
        except Exception as e:
            self.logger.error(f"Error plotting power data: {str(e)}")
            self.logger.error(traceback.format_exc())
            
    def update_power_plot(self, label):
        """Handle electrode toggling for power plot."""
        self.electrode_states[label] = not self.electrode_states[label]
        self.plot_power_data()

    def toggle_filters(self, label):
        """Toggle filters and update display."""
        filter_changed = False
        
        # Handle different filter types
        if label == 'EEG Filters':
            old_state = self.CONFIG['FILTERS']['EEG']['enabled']
            self.CONFIG['FILTERS']['EEG']['enabled'] = not old_state
            self.logger.info(f"EEG filters {'enabled' if self.CONFIG['FILTERS']['EEG']['enabled'] else 'disabled'}")
            filter_changed = old_state != self.CONFIG['FILTERS']['EEG']['enabled']
            
        elif label == 'AUX Filters':
            old_state = self.CONFIG['FILTERS']['AUX']['enabled']
            self.CONFIG['FILTERS']['AUX']['enabled'] = not old_state
            self.logger.info(f"AUX filters {'enabled' if self.CONFIG['FILTERS']['AUX']['enabled'] else 'disabled'}")
            filter_changed = old_state != self.CONFIG['FILTERS']['AUX']['enabled']
            
        elif label == 'ICA':
            # Use the toggle_ica method to handle ICA changes
            self.toggle_ica(label)
            # Already handled the update in toggle_ica, so return early
            return
        
        # Apply filters if filter settings changed
        if filter_changed:
            success = self.apply_filters()
            self.logger.info(f"Filters applied: {success}")
        
        # Update the plot
        self.update_plot()

    def zoom_in_erp_plot(self, event):
        """Zoom in on ERP plot while maintaining center"""
        if not hasattr(self, 'ax_avg') or len(self.ax_avg.lines) == 0:
            return
            
        # Get current y limits
        ymin, ymax = self.ax_avg.get_ylim()
        center = (ymin + ymax) / 2
        
        # Calculate new range (zoom in by 20%)
        current_range = ymax - ymin
        new_range = current_range * 0.8  # 20% zoom in
        
        # Set new limits centered around middle
        half_range = new_range / 2
        self.ax_avg.set_ylim(center - half_range, center + half_range)
        self.fig_avg.canvas.draw_idle()

    def zoom_out_erp_plot(self, event):
        """Zoom out on ERP plot"""
        if not hasattr(self, 'ax_avg') or len(self.ax_avg.lines) == 0:
            return
            
        # Get current y limits
        ymin, ymax = self.ax_avg.get_ylim()
        center = (ymin + ymax) / 2
        
        # Calculate new range (zoom out by 20%)
        current_range = ymax - ymin
        new_range = current_range * 1.2  # 20% zoom out
        
        # Set new limits centered around middle
        half_range = new_range / 2
        self.ax_avg.set_ylim(center - half_range, center + half_range)
        self.fig_avg.canvas.draw_idle()

    def zoom_in_pd_plot(self, event):
        """Zoom in on PD plot"""
        self.y_scale_pd = max(
            self.CONFIG['PLOT']['SCALES']['AUX']['min'],
            self.y_scale_pd * 0.8
        )
        self.update_averages_plot()

    def zoom_out_pd_plot(self, event):
        """Zoom out on PD plot"""
        self.y_scale_pd = min(
            self.CONFIG['PLOT']['SCALES']['AUX']['max'],
            self.y_scale_pd * 1.2
        )
        self.update_averages_plot()
        
    def _plot_empty_condition(self, times, condition_name):
        """Plot empty condition with message when no data available."""
        self.ax_avg.text(0.5, 0.5, f"No data available for condition: {condition_name}",
                    ha='center', va='center', transform=self.ax_avg.transAxes,
                    fontsize=12, color='gray')
        self.ax_pd.text(0.5, 0.5, "No photodiode data",
                    ha='center', va='center', transform=self.ax_pd.transAxes,
                    fontsize=12, color='gray')
        
        # Set reasonable axis limits
        self.ax_avg.set_xlim(times[0], times[-1])
        self.ax_avg.set_ylim(-1, 1)
        self.ax_pd.set_xlim(times[0], times[-1])
        self.ax_pd.set_ylim(-1, 1)

    def _plot_condition_data(self, epochs, times, condition, color=None, line_style='-'):
        """Plot condition data with unique styling for each electrode-condition pair."""
        # Define distinct colors for individual electrodes
        electrode_colors = {
            'AF7': '#FF1493',    # Deep Pink
            'AF8': '#4169E1',    # Royal Blue
            'TP9': '#32CD32',    # Lime Green
            'TP10': '#FF4500',   # Orange Red
        }
        
        # Define pair color palettes for different conditions
        pair_color_palettes = {
            'highpos': {'AF7/AF8': '#8A2BE2', 'TP9/TP10': '#DC143C'},     # Blue Violet, Crimson
            'neutral': {'AF7/AF8': '#FF69B4', 'TP9/TP10': '#FF4500'},     # Hot Pink, Orange Red
            'lowpos': {'AF7/AF8': '#00CED1', 'TP9/TP10': '#2E8B57'},      # Dark Turquoise, Sea Green
            'highneg': {'AF7/AF8': '#4682B4', 'TP9/TP10': '#800080'},     # Steel Blue, Purple
            'lowneg': {'AF7/AF8': '#DAA520', 'TP9/TP10': '#20B2AA'}       # Goldenrod, Light Sea Green
        }
        
        # Get photodiode data
        if self.CONFIG['PHOTODIODE']['DETECTION']['CHANNEL'] in epochs.ch_names:
            pd_idx = epochs.ch_names.index(self.CONFIG['PHOTODIODE']['DETECTION']['CHANNEL'])
            pd_data = epochs.get_data()[:, pd_idx, :]
            
            if len(pd_data) > 0 and not np.all(np.isnan(pd_data)):
                pd_avg = np.nanmean(pd_data, axis=0)
                pd_sem = sem(pd_data, axis=0, nan_policy='omit')
                
                self.ax_pd.plot(times, pd_avg, color=color, 
                            label=f"PD ({condition})",
                            linewidth=self.CONFIG['PLOT']['STYLES']['line_width'],
                            linestyle=line_style)
                self.ax_pd.fill_between(times, pd_avg - pd_sem, pd_avg + pd_sem,
                                    color=color, alpha=0.2)
        
        # Process selected electrodes
        selected_channels = [ch for ch, state in self.electrode_states.items() if state]
        if not selected_channels:
            self.logger.warning("No electrodes selected for averaging")
            return
            
        # Handle electrode pairs separately
        pair_channels = {
            'AF7/AF8': ['AF7', 'AF8'],
            'TP9/TP10': ['TP9', 'TP10']
        }
        
        for pair_name, pair_electrodes in pair_channels.items():
            if all(ch in selected_channels for ch in pair_electrodes):
                # Find indices for the pair
                pair_indices = [epochs.ch_names.index(ch) for ch in pair_electrodes]
                
                # Get data for selected pair
                pair_data = epochs.get_data()[:, pair_indices, :]
                
                # Average across the pair first
                avg_pair_data = np.nanmean(pair_data, axis=1)
                
                # Calculate average and standard error
                grand_avg = np.nanmean(avg_pair_data, axis=0)
                grand_stderr = sem(avg_pair_data, axis=0, nan_policy='omit')
                
                # Use condition-specific pair color
                pair_color = pair_color_palettes.get(condition, {}).get(pair_name, 'gray')
                
                # Plot pair average
                self.ax_avg.plot(times, grand_avg, color=pair_color,
                            label=f"{pair_name} ({condition})",
                            linewidth=self.CONFIG['PLOT']['STYLES']['line_width'],
                            linestyle=line_style)
                self.ax_avg.fill_between(times, grand_avg - grand_stderr, grand_avg + grand_stderr,
                                    color=pair_color, alpha=0.2)
        
        # Plot individual electrodes if not part of a pair
        for ch in selected_channels:
            if ch not in ['AF7', 'AF8', 'TP9', 'TP10']:
                if ch in epochs.ch_names:
                    ch_idx = epochs.ch_names.index(ch)
                    ch_data = epochs.get_data()[:, ch_idx, :]
                    
                    if len(ch_data) > 0 and not np.all(np.isnan(ch_data)):
                        ch_avg = np.nanmean(ch_data, axis=0)
                        ch_stderr = sem(ch_data, axis=0, nan_policy='omit')
                        
                        ch_color = electrode_colors.get(ch, 'gray')
                        
                        self.ax_avg.plot(times, ch_avg, color=ch_color,
                                    label=f"{ch} ({condition})",
                                    linewidth=self.CONFIG['PLOT']['STYLES']['line_width'],
                                    linestyle=line_style)
                        self.ax_avg.fill_between(times, ch_avg - ch_stderr, ch_avg + ch_stderr,
                                            color=ch_color, alpha=0.2)


    def setup_averages_view(self):
        """Setup interactive averages view with checkbox condition selection."""
        if self.trials is None:
            self.logger.warning("No trials available for averaging")
            messagebox.showwarning("No Data", "No trials available for averaging")
            return
        
        # Create new figure
        self.fig_avg = plt.figure(figsize=self.GUI['LAYOUT']['figure_size'])
        gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Create subplots
        self.ax_avg = self.fig_avg.add_subplot(gs[0])  # ERP plot
        self.ax_pd = self.fig_avg.add_subplot(gs[1])   # Photodiode plot
        
        # Get conditions
        if hasattr(self, 'trials') and hasattr(self.trials, 'event_id'):
            available_conditions = list(self.trials.event_id.keys())
            self.logger.info(f"Found {len(available_conditions)} conditions: {available_conditions}")
        else:
            self.logger.warning("No condition information found - using 'All Trials'")
            available_conditions = ['All Trials']
            
        # Setup control panel
        ctrl_width = 0.2
        main_width = 0.98 - ctrl_width
        
        # Condition selection header
        current_y = 0.92
        self.fig_avg.text(
            main_width + 0.01 + (ctrl_width-0.02)/2,
            current_y,
            'Select Conditions',
            horizontalalignment='center',
            fontsize=11,
            fontweight='bold'
        )
        current_y -= 0.05
        
        # Initialize condition states dictionary for checkboxes
        if not hasattr(self, 'condition_states'):
            self.condition_states = {cond: True for cond in available_conditions}
        else:
            # Make sure all available conditions are included
            for cond in available_conditions:
                if cond not in self.condition_states:
                    self.condition_states[cond] = True
        
        # Add condition checkboxes
        condition_ax = plt.axes([main_width + 0.01, current_y - 0.15, ctrl_width-0.02, 0.15])
        self.condition_checks = CheckButtons(
            condition_ax,
            available_conditions,
            [self.condition_states.get(cond, True) for cond in available_conditions]
        )
        self.condition_checks.on_clicked(self.toggle_condition)
        current_y -= 0.2
        
        # Electrode selection header
        self.fig_avg.text(
            main_width + 0.01 + (ctrl_width-0.02)/2,
            current_y,
            'Select Electrodes',
            horizontalalignment='center',
            fontsize=11, 
            fontweight='bold'
        )
        current_y -= 0.05
        
        # Initialize electrode states if needed
        if not hasattr(self, 'electrode_states'):
            self.electrode_states = {
                'AF7': True,
                'AF8': True,
                'TP9': False, 
                'TP10': False
            }
        
        # Electrode checkboxes
        electrode_ax = plt.axes([main_width + 0.01, current_y - 0.15, ctrl_width-0.02, 0.15])
        self.electrode_checks = CheckButtons(
            electrode_ax,
            ['AF7', 'AF8', 'TP9', 'TP10'],
            [self.electrode_states[ch] for ch in ['AF7', 'AF8', 'TP9', 'TP10']]
        )
        self.electrode_checks.on_clicked(self.toggle_electrode)
        current_y -= 0.2
        
        # Scale controls
        btn_height = 0.04
        btn_spacing = 0.01
        
        self.zoom_in_erp = Button(
            plt.axes([main_width + 0.01, current_y, ctrl_width-0.02, btn_height]),
            'ERP Zoom +'
        )
        current_y -= (btn_height + btn_spacing)
        
        self.zoom_out_erp = Button(
            plt.axes([main_width + 0.01, current_y, ctrl_width-0.02, btn_height]),
            'ERP Zoom -'
        )
        current_y -= (btn_height + btn_spacing*2)
        
        self.zoom_in_pd = Button(
            plt.axes([main_width + 0.01, current_y, ctrl_width-0.02, btn_height]),
            'PD Zoom +'
        )
        current_y -= (btn_height + btn_spacing)
        
        self.zoom_out_pd = Button(
            plt.axes([main_width + 0.01, current_y, ctrl_width-0.02, btn_height]),
            'PD Zoom -'
        )
        current_y -= (btn_height + btn_spacing*3)
        
        # Export button
        self.export_avg_button = Button(
            plt.axes([main_width + 0.01, current_y, ctrl_width-0.02, btn_height]),
            'Export Averages'
        )
        self.export_avg_button.on_clicked(self.export_averages)
        
        # Connect callbacks
        self.zoom_in_erp.on_clicked(self.zoom_in_erp_plot)
        self.zoom_out_erp.on_clicked(self.zoom_out_erp_plot)
        self.zoom_in_pd.on_clicked(self.zoom_in_pd_plot)
        self.zoom_out_pd.on_clicked(self.zoom_out_pd_plot)
        
        # Connect keyboard and scroll events
        self.fig_avg.canvas.mpl_connect('key_press_event', self.on_key_press_avg)
        self.fig_avg.canvas.mpl_connect('scroll_event', self.on_scroll_avg)
        
        # Style figure
        self.fig_avg.patch.set_facecolor(self.CONFIG['PLOT']['STYLES']['background'])
        self.ax_avg.set_facecolor('white')
        self.ax_pd.set_facecolor('white')
        
        # Adjust layout and set default scales
        self.fig_avg.subplots_adjust(right=main_width)
        if not hasattr(self, 'y_scale_erp'):
            self.y_scale_erp = self.CONFIG['PLOT']['SCALES']['EEG']['default']
        if not hasattr(self, 'y_scale_pd'):
            self.y_scale_pd = self.CONFIG['PLOT']['SCALES']['AUX']['default']
        
        # Update averages plot based on current selections
        self.update_averages_plot()
        plt.show()
        
    def export_averages(self, event):
        """Export averages to CSV files."""
        try:
            if not hasattr(self, 'trials') or self.trials is None:
                self.logger.warning("No trials to export")
                return
                
            # Create export directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.dirname(self.input_filepath) if hasattr(self, 'input_filepath') else '.'
            export_dir = os.path.join(base_dir, f"averages_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)
            
            times = self.trials.times * 1000  # Convert to ms
            
            # Export each selected condition
            selected_conditions = [cond for cond, state in self.condition_states.items() if state]
            if not selected_conditions:
                messagebox.showwarning("Export Warning", "No conditions selected for export")
                return
                
            # Export each condition
            for condition in selected_conditions:
                if condition == 'All Trials':
                    self._export_condition_average(self.trials, times, 'All_Trials', export_dir)
                else:
                    try:
                        cond_trials = self.trials[condition]
                        self._export_condition_average(cond_trials, times, condition, export_dir)
                    except Exception as e:
                        self.logger.warning(f"Could not export {condition}: {str(e)}")
            
            # Export grand average if multiple conditions selected
            if len(selected_conditions) > 1:
                # Just export the current view as grand average
                self._export_condition_average(self.trials, times, 'Grand_Average', export_dir)
                
            self.logger.info(f"Averages exported to {export_dir}")
            messagebox.showinfo("Export Complete", f"Averages exported to {export_dir}")
            
        except Exception as e:
            self.logger.error(f"Error exporting averages: {str(e)}")
            self.logger.error(traceback.format_exc())
            messagebox.showerror("Export Error", str(e))
            
    def _export_condition_average(self, epochs, times, condition_name, export_dir):
        """Export average data for a specific condition to CSV."""
        # Calculate average for each channel
        data_dict = {'time_ms': times}
        
        # EEG channels
        for ch in self.CHANNELS['EEG']:
            if ch in epochs.ch_names:
                ch_idx = epochs.ch_names.index(ch)
                ch_data = epochs.get_data()[:, ch_idx, :]
                avg = np.nanmean(ch_data, axis=0)
                sem_values = sem(ch_data, axis=0, nan_policy='omit')
                data_dict[f'{ch}_avg'] = avg
                data_dict[f'{ch}_sem'] = sem_values
                
        # AUX channels
        for ch in self.CHANNELS['AUX']:
            if ch in epochs.ch_names:
                ch_idx = epochs.ch_names.index(ch)
                ch_data = epochs.get_data()[:, ch_idx, :]
                avg = np.nanmean(ch_data, axis=0)
                sem_values = sem(ch_data, axis=0, nan_policy='omit')
                data_dict[f'{ch}_avg'] = avg
                data_dict[f'{ch}_sem'] = sem_values
        
        # Create DataFrame and save
        df = pd.DataFrame(data_dict)
        safe_name = condition_name.replace('/', '_').replace(' ', '_')
        file_path = os.path.join(export_dir, f"{safe_name}_avg_n{len(epochs)}.csv")
        df.to_csv(file_path, index=False)
        
        # Save metadata
        with open(os.path.join(export_dir, f"{safe_name}_info.txt"), 'w') as f:
            f.write(f"Condition: {condition_name}\n")
            f.write(f"Number of trials: {len(epochs)}\n")
            f.write(f"Sampling rate: {self.CONFIG['SAMPLING_RATE']} Hz\n")
            f.write(f"Trial window: {self.CONFIG['TRIAL_WINDOW']['START']}s to {self.CONFIG['TRIAL_WINDOW']['END']}s\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    def toggle_electrode(self, label):
        """Handle individual electrode toggling."""
        self.electrode_states[label] = not self.electrode_states[label]
        self.update_averages_plot()

    def toggle_condition(self, label):
        """Toggle visibility of a condition."""
        self.condition_states[label] = not self.condition_states[label]
        self.update_averages_plot()

    def update_averages_plot(self):
        """Update the averages plot based on selected conditions and electrodes."""
        # Clear the plots
        self.ax_avg.clear()
        self.ax_pd.clear()
        
        # Get time values
        times = self.trials.times * 1000  # Convert to milliseconds
        
        try:
            # Check if any conditions are selected
            selected_conditions = [cond for cond, state in self.condition_states.items() if state]
            if not selected_conditions:
                self._plot_empty_condition(times, "No conditions selected")
                self._style_average_plots()
                self.fig_avg.canvas.draw_idle()
                return
                
            # Define colors for conditions
            condition_colors = {
                'highpos': '#1f77b4',  # blue
                'neutral': '#ff7f0e',  # orange
                'lowpos': '#2ca02c',   # green
                'highneg': '#d62728',  # red
                'lowneg': '#9467bd',   # purple
                'All Trials': 'black',
                'Grand Average': 'black'
            }
            
            # Plot each selected condition
            total_trials = 0
            legends_added = False
            
            for condition in selected_conditions:
                try:
                    if condition == 'Grand Average':
                        # Plot all trials
                        self._plot_condition_data(self.trials, times, 'Grand Average', 
                                            color=condition_colors.get('Grand Average', 'black'))
                        total_trials = len(self.trials)
                    else:
                        # Plot specific condition
                        cond_trials = self.trials[condition]
                        if len(cond_trials) > 0:
                            self._plot_condition_data(cond_trials, times, condition,
                                                color=condition_colors.get(condition, None))
                            total_trials += len(cond_trials)
                            legends_added = True
                        else:
                            self.logger.warning(f"No trials found for condition {condition}")
                except Exception as e:
                    self.logger.warning(f"Error plotting condition {condition}: {str(e)}")
            
            # Add title with trial count
            if len(selected_conditions) == 1:
                condition = selected_conditions[0]
                title = f"{condition} - {total_trials} trials"
            else:
                title = f"Multiple Conditions - {total_trials} total trials"
            
            self.ax_avg.set_title(title)
            self._style_average_plots()
            
        except Exception as e:
            self.logger.error(f"Error updating averages plot: {str(e)}")
            self.logger.error(traceback.format_exc())
            self._plot_empty_condition(times, f"Error: {str(e)}")
        
        self.fig_avg.canvas.draw_idle()

    def _style_average_plots(self):
        """Apply styling to average plots."""
            
        for ax in [self.ax_avg, self.ax_pd]:
            # Add vertical line at stimulus onset
            ax.axvline(x=0, color='black', linestyle='--',
                    alpha=self.CONFIG['PLOT']['STYLES']['marker_alpha'])
            
            # Add horizontal zero line
            ax.axhline(y=0, color='black', linestyle='-',
                    alpha=0.3, zorder=1)
                    
            # Set standard axis limits
            ax.set_xlim(
                self.CONFIG['TRIAL_WINDOW']['START'] * 1000,
                self.CONFIG['TRIAL_WINDOW']['END'] * 1000
            )
            
            # Add grid
            ax.grid(True, alpha=self.CONFIG['PLOT']['STYLES']['grid_alpha'])
            
            # Hide top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Set labels
        self.ax_avg.set_ylabel('Amplitude (μV)')
        self.ax_pd.set_ylabel('Photodiode (μV)')
        self.ax_pd.set_xlabel('Time (ms)')
        
        # Add legends if there are any labeled artists
        if len(self.ax_avg.get_legend_handles_labels()[0]) > 0:
            self.ax_avg.legend(loc='upper right', frameon=True, fontsize=9)
        if len(self.ax_pd.get_legend_handles_labels()[0]) > 0:
            self.ax_pd.legend(loc='upper right', frameon=True, fontsize=9)
        
        # Auto-scale for visibility
        scale_erp = self.y_scale_erp if hasattr(self, 'y_scale_erp') else 5
        scale_pd = self.y_scale_pd if hasattr(self, 'y_scale_pd') else 1000
        
        # Only auto-scale if there are lines to measure
        if len(self.ax_avg.lines) > 0:
            ydata = np.concatenate([line.get_ydata() for line in self.ax_avg.lines])
            if not np.all(np.isnan(ydata)):
                ymin, ymax = np.nanmin(ydata), np.nanmax(ydata)
                if np.isfinite(ymin) and np.isfinite(ymax):
                    ymean = (ymin + ymax) / 2
                    self.ax_avg.set_ylim(ymean - scale_erp/2, ymean + scale_erp/2)
                
        if len(self.ax_pd.lines) > 0:
            ydata = np.concatenate([line.get_ydata() for line in self.ax_pd.lines])
            if not np.all(np.isnan(ydata)):
                ymin, ymax = np.nanmin(ydata), np.nanmax(ydata)
                if np.isfinite(ymin) and np.isfinite(ymax):
                    ymean = (ymin + ymax) / 2
                    self.ax_pd.set_ylim(ymean - scale_pd/2, ymean + scale_pd/2)
        
    def navigate_conditions(self, direction):
        """Navigate between conditions in average view."""
        conditions = self.CONFIG['AVERAGING']['conditions']
        if direction == 'prev':
            self.current_condition = (self.current_condition - 1) % len(conditions)
        elif direction == 'next':
            self.current_condition = (self.current_condition + 1) % len(conditions)
        
        self.update_averages_plot()

    def navigate_trials(self, direction):
        """Navigate between trials."""
        if direction == 'prev' and self.current_trial > 0:
            self.current_trial -= 1
        elif direction == 'next' and self.current_trial < len(self.trials) - 1:
            self.current_trial += 1
            
        self.update_plot()

    # Event handlers
    def on_scroll(self, event):
        """Handle scroll events."""
        if event.inaxes:
            ch = event.inaxes.get_ylabel().split('\n')[0]
            ch_type = 'EEG' if ch in self.CHANNELS['EEG'] else 'AUX'
            scales = self.CONFIG['PLOT']['SCALES'][ch_type]
            
            current_scale = self.channel_scales[ch]
            
            if event.button == 'up':
                new_scale = current_scale / 1.2
            else:
                new_scale = current_scale * 1.2
            
            if ch_type == 'EEG':
                self.channel_scales[ch] = np.clip(new_scale, scales['min'], scales['max'])
            else:  # AUX
                min_scale = (scales['max'] - scales['min']) / 10
                max_scale = scales['max'] - scales['min']
                self.channel_scales[ch] = np.clip(new_scale, min_scale, max_scale)
            
            self.update_plot()

    def on_scroll_avg(self, event):
        """Handle scroll events in average view."""
        if event.inaxes:
            if event.inaxes == self.ax_avg:
                if event.button == 'up':
                    self.zoom_in_erp_plot(None)
                else:
                    self.zoom_out_erp_plot(None)
            elif event.inaxes == self.ax_pd:
                if event.button == 'up':
                    self.zoom_in_pd_plot(None)
                else:
                    self.zoom_out_pd_plot(None)

    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'left':
            self.navigate_trials('prev')
        elif event.key == 'right':
            self.navigate_trials('next')
        elif event.key == 'up':
            self.scale_slider.set_val(min(
                self.CONFIG['PLOT']['SCALES']['EEG']['max'],
                self.scale * 1.2
            ))
        elif event.key == 'down':
            self.scale_slider.set_val(max(
                self.CONFIG['PLOT']['SCALES']['EEG']['min'],
                self.scale / 1.2
            ))

    def on_key_press_avg(self, event):
        """Handle keyboard events in average view."""
        if event.key == 'left':
            self.navigate_conditions('prev')
        elif event.key == 'right':
            self.navigate_conditions('next')
        elif event.key == 'up':
            if event.inaxes == self.ax_avg:
                self.zoom_in_erp_plot(None)
            elif event.inaxes == self.ax_pd:
                self.zoom_in_pd_plot(None)
        elif event.key == 'down':
            if event.inaxes == self.ax_avg:
                self.zoom_out_erp_plot(None)
            elif event.inaxes == self.ax_pd:
                self.zoom_out_pd_plot(None)

def main():
    """Main entry point for CSV-only application."""
    root = tk.Tk()
    root.withdraw()
    
    try:
        processor = MuseTrialProcessor()
        
        muse_file = filedialog.askopenfilename(
            title="Select MUSE data file",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not muse_file:
            print("No file selected. Exiting...")
            return
            
        # Process data
        processor.load_muse_data(muse_file)
        processor.create_trials()
        processor.setup_gui()
        
        plt.show()
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        messagebox.showerror("Error", str(e))
        raise

if __name__ == "__main__":
    main()
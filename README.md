This repository contains tools for stimulus presentation and EEG data analysis used in Internship I project of VU Amsterdam,
completed by Michal Trajdos, under the supervision of Dr. Markus Junghöfer.

## Stimulus Applications

### EmoScenes/ShamScenes Setup

1. Download the relevant `.apk` installation file.
2. Connect your tablet to a desktop computer.
3. On the tablet, select "File Transfer" mode to enable file access.
4. Transfer the `.apk` file to your chosen directory on the tablet.
   - When prompted with the Android security message, select "Continue without scanning" (this occurs due to unrecognized external .apk)
5. Navigate to the transferred `.apk` file and tap to install the application.
6. Launch the installed application.
7. Tap once on the black screen to initialize stimulus display.
8. Tap again to terminate the stimulation.
   - The stimulation log file will be saved in the Downloads folder of your tablet

## Segmentation and analysis

### Requirements

- Python 3.6+
- MNE-Python
- NumPy, Pandas, Matplotlib
- SciPy
- edfio (optional - in case you do not need to save trials as in .edf format, you can remove save_trials_to_edf function from the script)

## Installation

### Python Environment
Ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

If pip command is not available, you need to install pip first. The requirements file includes:

```
mne>=1.8.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.9.0
scipy>=1.14.0
edfio>=0.4.4
```
### Usage

1. Ensure your Python environment has all required packages installed.
2. Verify that filenames of your Muse `.csv` data file and log file match:
   - Desktop script: uses `.dat` log files
   - Tablet script: uses `.log` or `.txt` log files
   - **Note**: Without matching names between the Muse data and the log, the script will rely solely on photodiode signal detection, which is less robust and won't assign experimental conditions to trials
3. Execute the relevant analysis script:
   ```bash
   python MuseTrialAnalysisForDesktopStimulation.py
   # or
   python3 MuseTrialAnalysisForTabletStimulation.py
   ```
4. When prompted, select the `.csv` data file.
5. Wait for segmentation to complete.
6. Once processing is finished, several interactive plots will be generated:
   - Navigate between trials using left/right arrow keys
   - Zoom subplots by scrolling up/down with the mouse wheel
   - Access additional analyses via the UI buttons

### Configuration

Both analysis scripts contain a configurable `CONFIG` dictionary that can be modified.

**Note**: PD detection method (detect_photodiode_events) for the desktop-displayed stimuli is still a work in progress, therefore has limited config compatibility.

```python
self.CONFIG = {
    'ICA': {
        'enabled': False,
        'n_components': 4,
        'method': 'fastica',
        # Additional ICA parameters...
    },
    'PHOTODIODE': {
        'DETECTION': {
            'THRESHOLD': 550,           # Signal threshold for detection
            'MIN_AMPLITUDE': 150,       # Minimum amplitude change
            'CHANNEL': 'AUX_L',         # Photodiode input channel
            # Additional detection parameters...
        }
    },
    'SAMPLING_RATE': 256,
    'TRIAL_WINDOW': {
        'START': -0.2,                  # Time window start (seconds)
        'END': 0.6,                     # Time window end (seconds)
        'BASELINE': (-0.2, 0)           # Baseline correction period
    },
    # Additional configuration parameters...
}
```

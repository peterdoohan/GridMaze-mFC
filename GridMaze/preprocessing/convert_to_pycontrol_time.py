""" """
#%% Imports
import os
import numpy as np
from .rsync import Rsync_aligner
from . import pycontrol_data_import as di

#%% Global variables

# %% Preprocessing functions
def get_spike_pytimes(session_dir):
    '''
    Converts kilsort sample number spike times to pycontrol spike times (spike_pytimes)
        INPUT:
            - kilosort_path: filepath to the /Phy folder in the kilosorted session of interest
            - sync_path: filepath to the timestamps.npy in the raw ephys file
                !NOTE! Haven't decided where this file will be sorted on disk, manualy added to /raw_data/ephys for the moment'
            - pycontrol_path: filepath to the associated raw pycontrol session file (.txt)

        OUTPUT:
            - spike_pytimes: times of spikes recorded in the session reported in seconds of pycontrol time.
                returns nan values where pycontrol was not running while ephys was being recorded (i.e. start and end of session)
        NOTES:
            - cambridge neurotech probe sample rate = 30000 samples/s which has been hard coded into the function
    '''
    spike_sample_numbers = np.load(os.path.join(session_dir.phy_path, 'spike_times.npy'))
    rsync_sample_numbers = np.load(session_dir.ephys_timestamps_path)[::2]
    pycontrol_sync_pulse_times = di.Session(session_dir.pycontrol_path).times['rsync']/1000 #seconds
    spike_pytimes = Rsync_aligner(rsync_sample_numbers, pycontrol_sync_pulse_times, units_A=1/30000, units_B=1).A_to_B(spike_sample_numbers)
    return spike_pytimes.reshape(len(spike_pytimes),)

# %%

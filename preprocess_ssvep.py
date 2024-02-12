import mne
import numpy as np
import os
from scipy.signal import filtfilt, butter,lfilter



def filter_bandpass(signal, lowcut, highcut, fs, order=4, filttype='forward-backward'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if filttype == 'forward':
        filtered = lfilter(b, a, signal, axis=-1)
    elif filttype == 'forward-backward':
        filtered = filtfilt(b, a, signal, axis=-1)
    else:
        raise ValueError("Unknown filttype:", filttype)    
    return filtered    




class ExoSkeleton():

    def __init__(self,tmin,delta_t,resampling=256):
        
        self.data_path = "/ssvep_moabb/"
        self.freqs = [13,17,21]
        self.nb_trials_per_session = 32
        self.sampling = 256
        self.event_code_fif = [1,2,3,4]
        self.lowcut = 1
        self.highcut = self.freqs[-1]*2+1
        self.channels = ['Oz','O1','O2','PO3','POz','PO7','PO8','PO4']
        self.names=['resting','stim13','stim21','stim17']
        
        self.resampling = resampling        
        self.tmin = tmin
        self.delta_t = delta_t
        assert self.tmin>=0, "The starting time tmin must be positive"
        assert self.delta_t>=0, "The duration delta_t must be positive"
        assert self.tmin+self.delta_t<= 5, "Each trial has a maximum duration of 5 seconds"
        
        
    def SsvepLoading(self):
        """
        Outputs:
            subj_list : a list of subjects ["subject1",...]
            records : a dictionnary of subjects and their associated sessions
        """
        subj_list = os.listdir(self.data_path)
        records = {s: [] for s in range(len(subj_list))}
        for s in range(len(subj_list)):
            subj = subj_list[s]
            record_all = os.listdir(self.data_path+subj+'/')
            for file in record_all:
                if file[len(file)-8:]=="_raw.fif":
                    records[s].append(file[:len(file)-8])
        return subj_list,records


    def load_all_data(self,records,subj_list):
        data = { subject : None for subject in range(len(subj_list))}
        for subject in range(len(subj_list)):
            trials,labels = self.extract_trials(records,subj_list, subject)
            data[subject] = [trials,labels]
        return data




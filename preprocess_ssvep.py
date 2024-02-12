import numpy as np
from scipy.signal import filtfilt, butter,lfilter
from moabb.datasets import SSVEPExo
from moabb.paradigms import SSVEP


def filter_bandpass(signal, lowcut, highcut, fs, order=4, filttype='forward-backward'):
    """
    filters a signal using a bandpass filter

    Parameters
    ----------
    signal : array_like
        The array of data to be filtered.
    lowcut : float
        The cutoff frequency (Hz) for the high pass filter
    highcut : float
        The cutoff frequency (Hz) for the low pass filter
    fs : float
        The sampling frequency (Hz).
    order : int, optional
        The order of the filter. The default is 4.
    filttype : str, optional
        The method of the filter. The default is 'forward-backward'.


    Returns
    -------
    filtered : ndarray
        The filtered output with the same shape as `signal`.

    """
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
    """
    Base dataset class for 
    SSVEP dataset from E. Kalunga PhD in University of Versailles [1].
    
    References
    ----------
    .. [1] Emmanuel K. Kalunga, Sylvain Chevallier, Quentin Barthelemy. "Online
           SSVEP-based BCI using Riemannian Geometry". Neurocomputing, 2016.
           arXiv report: https://arxiv.org/abs/1501.03227
    """

    def __init__(self,tmin,delta_t,resampling=256):
        """
        Parameters
        ----------
        tmin : float
            Offset of the trial.
        delta_t : float
            Duration of the trial.
        resampling : float, optional
            Resampling frequency (Hz). The default is 256.

        Returns
        -------
        None.

        """
        
        self.freqs = [13,17,21]
        self.lowcut = 1
        self.highcut = self.freqs[-1]*2+1
        self.classes = {"rest":1,"13":2,"21":3,"17":4}
        
        self.resampling = resampling        
        self.tmin = tmin
        self.delta_t = delta_t
        assert self.tmin>=0, "The starting time tmin must be positive"
        assert self.delta_t>=0, "The duration delta_t must be positive"
        assert self.tmin+self.delta_t<= 5, "Each trial has a maximum duration of 5 seconds"
        
    
    def transform_labels(self,labels):
        """
        transforms `rest` to 1, `13` to 2, `21` to 3 and `17` to 4.

        Parameters
        ----------
        labels : array_like
            names of the classes.

        Returns
        -------
        array
            labels after transformation.

        """
        labels_=[self.classes[x] for x in labels]
        return np.asarray(labels_)
        

    def load_all_data(self):
        """
        loads the data (trials + labels) for all the subjects of the dataset
        
        Returns
        -------
        dict
            dictionnary containing the trials and the labels of each subject.
        """
        dataset = SSVEPExo()
        subj_list = dataset.subject_list
        
        paradigm = SSVEP(fmin=self.lowcut,fmax=self.highcut,tmin=self.tmin,tmax=self.tmin+self.delta_t,resample=self.resampling)
        X, labels, meta = paradigm.get_data(dataset=dataset)
        labels = self.transform_labels(labels)
        
        records = {subject:None for subject in subj_list}
        n_trials= 0
        for subject in subj_list:
            n_trials_subject = len(meta[meta["subject"]==subject])
            records[subject] = X[n_trials:n_trials+n_trials_subject],labels[n_trials:n_trials+n_trials_subject]
            n_trials += n_trials_subject
        return records
    
    
    

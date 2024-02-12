# On elliptical and Inverse elliptical Wishart distributions: Review, new results, and applications

This code reproduces the numerical results about fitting real EEG data with Wishart and t-Wishart distributions

To get the figures and the p-values of the statistical tests provided in the paper, please run "main.py"

The repository contains:
| Name             | Description   |
| -------------    |:-------------:|
| main             | Plot figures of fitting provided in the paper         |
| preprocess_ssvep | Load the ExoSkeleton dataset, filter the SSVEP recordings, and cut them into trials     | 
| tWishart         | Draw random samples from the t-Wishart distribution and derive the MLE for the center parameter given a degree of freedom     |  
| manifold         | Framework for Riemannian optimization needed to compute the MLE of t-Wishart samples: manifold of the center parameter   |    
| fitting          | Compute the empirical cumulative density function (cdf) of EEG samples and the cdfs of the fitted Wishart and t-Wishart samples and yield the Kolmogorov-Smirnov statistical tests for the Wishart and t-Wishart distributions     |   

## Requirements: 
numpy - scipy - matplotlib - moabb - pymanopt - mne - tqdm - joblib


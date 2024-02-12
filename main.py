from preprocess_ssvep import ExoSkeleton
from fitting import eeg_covs, fit_eeg
import matplotlib.pyplot as plt

tmin = 2 #offset for trials 
delta_t = 3 #considered duration for trials
resampling = 256 #resampling frequency
selected_subjs = [12] #list of selected subjects from the twelve subjects of the dataset
quantity_to_fit = "norm" #change into "det" or "trace" if you want to perform the fitting of the determinant or the trace, respectively
df = {1:24, 2:20, 3:28, 4:41} #degrees of freedom for each class
n_jobs = -1

#1. Load the dataset
data = ExoSkeleton(tmin,delta_t,resampling).load_all_data()

#2. Extract covariance matrices 
cov_data = eeg_covs(data,tmin,delta_t,resampling)

#3. Fitting
xs,eeg_cdf,wishart_cdf,t_wishart_cdf,tests_W , tests_tW = fit_eeg(quantity_to_fit,cov_data,selected_subjs,delta_t,resampling, df=df,n_jobs=n_jobs)

#4. Plot results of the fitting for each class
for k in range(1,len(xs)+1):
    plt.plot(xs[k],eeg_cdf[k],label="eeg samples",linestyle="solid")
    plt.plot(xs[k],t_wishart_cdf[k],label="$t$-Wishart samples of df="+str(df[k]),linestyle="dotted")
    plt.plot(xs[k],wishart_cdf[k],label="Wishart samples",linestyle="dashdot")

    plt.legend(loc="best",fontsize=7) 

    plt.title("The empirical CDF of the "+quantity_to_fit+" for class "+str(k)
              +"\n KS p-value of Wishart fitting="+str(tests_W[k][1])
              +"\n KS p-value of t-Wishart fitting="+str(tests_tW[k][1]))

    plt.show()

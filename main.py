from fitting import eeg_covs, fit_eeg
import matplotlib.pyplot as plt

tmin = 2
delta_t = 3 
resampling = 256
selected_subjs = [11]
quantity_to_fit = "norm" #other options: "det" for determinant and "trace" for trace fittings
df = {1:24, 2:20, 3:28, 4:41}
n_jobs = -1


 
cov_data = eeg_covs(tmin,delta_t,resampling)
 
xs,eeg_cdf,wishart_cdf,t_wishart_cdf,tests_W , tests_tW = fit_eeg(quantity_to_fit,cov_data,selected_subjs,delta_t,resampling, df=df,n_jobs=n_jobs)


for k in range(1,len(xs)+1):
    plt.plot(xs[k],eeg_cdf[k],label="eeg samples",linestyle="solid")
    plt.plot(xs[k],t_wishart_cdf[k],label="$t$-Wishart samples of df="+str(df[k]),linestyle="dotted")
    plt.plot(xs[k],wishart_cdf[k],label="Wishart samples",linestyle="dashdot")

    plt.legend(loc="best",fontsize=7) 

    plt.title("The empirical CDF of the "+quantity_to_fit+" for class "+str(k)
              +"\n KS p-value of Wishart fitting="+str(tests_W[k][1])
              +"\n KS p-value of t-Wishart fitting="+str(tests_tW[k][1]))

    plt.show()

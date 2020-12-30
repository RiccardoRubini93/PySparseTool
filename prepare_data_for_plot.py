import numpy as np 
import shutil

shutil.copy('Regression_Results/alphas','plotting_data/alpha.txt')

info = np.genfromtxt('Regression_Results/info.txt')

n_samples  = int(info[0])
n_modes    = int(info[1])
n_features = int(info[2])
n_alpha    = int(info[3])

#assemble beta matrix

coeff = np.genfromtxt('Regression_Results/coeffs_000') 

for i in range(1,n_modes):

	coeff_ = np.genfromtxt('Regression_Results/coeffs_%03d' % i)	
	coeff = np.concatenate((coeff,coeff_),axis=1)

#save matrix beta

np.savetxt('plotting_data/beta.txt',coeff.T,delimiter=' ')
np.savetxt('plotting_data/eta.txt',coeff[0,:].reshape((n_features,n_modes),order='F'),delimiter=' ')

#compute mean square error

MSE_full = np.genfromtxt('Regression_Results/MSE_full_000').reshape(n_alpha,1)

for i in range(1,n_modes):

	MSE_full_ = np.genfromtxt('Regression_Results/MSE_full_%03d' % i).reshape(n_alpha,1)	
	MSE_full = np.concatenate((MSE_full,MSE_full_),axis=1)

##########################

#''' TO BE FIXED 

MSE_full_rel = np.genfromtxt('Regression_Results/MSE_full_rel_000').reshape(n_alpha,1)

for i in range(1,n_modes):

	MSE_full_rel_ = np.genfromtxt('Regression_Results/MSE_full_rel_%03d' % i).reshape(n_alpha,1)	
	MSE_full_rel = np.concatenate((MSE_full,MSE_full_),axis=1)
#'''
##########################

MSE_mean = np.genfromtxt('Regression_Results/MSE_mean_000').reshape(n_alpha,1) 

for i in range(1,n_modes):

	MSE_mean_ = np.genfromtxt('Regression_Results/MSE_mean_%03d' % i).reshape(n_alpha,1)	
	MSE_mean = np.concatenate((MSE_mean,MSE_mean_),axis=1)

##########################

MSE_std = np.genfromtxt('Regression_Results/MSE_std_000').reshape(n_alpha,1) 

for i in range(1,n_modes):

	MSE_std_ = np.genfromtxt('Regression_Results/MSE_std_%03d' % i).reshape(n_alpha,1)	
	MSE_std = np.concatenate((MSE_std,MSE_std_),axis=1)

##########################

nnz = np.genfromtxt('Regression_Results/nnz_000').reshape(n_alpha,1) 

for i in range(1,n_modes):

	nnz_ = np.genfromtxt('Regression_Results/nnz_%03d' % i).reshape(n_alpha,1)	
	nnz  = np.concatenate((nnz,nnz_),axis=1)

MSE = np.mean(MSE_full,axis=1)
MSE_m = np.mean(MSE_mean,axis=1)
MSE_s = np.mean(MSE_std,axis=1)
MSE_r = np.mean(MSE_full_rel,axis=1)

#define sparsity

Sp = list()

for i in range(n_alpha):

	Sp.append(np.sum(nnz[i,:])/np.sum(nnz[0,:]))

np.savetxt('plotting_data/L2.txt',MSE,delimiter=' ')
np.savetxt('plotting_data/L2_mean.txt',MSE_m,delimiter=' ')
np.savetxt('plotting_data/L2_std.txt',MSE_s,delimiter=' ')
np.savetxt('plotting_data/density.txt',Sp,delimiter=' ')
np.savetxt('plotting_data/L2_rel.txt',MSE_r,delimiter=' ')

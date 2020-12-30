from mpi4py import MPI
import os, sys
from itertools import repeat
import numpy as np
import multiprocessing
from sklearn import linear_model
from utils.collections import *
import shutil
import pdb

def get_info():

	Theta = np.loadtxt('Regression_Data/Theta.txt')
	dadt  = np.loadtxt('Regression_Data/a_dot.txt')
	nsamples, nfeatures = Theta.shape
	nn = dadt.shape[1]

	info = list()
	info.append(nsamples)
	info.append(nn)
	info.append(nfeatures)
	info.append(len(alphas))

	return info

def work(i, kfolds, alphas):
	""" Perform regression on the i-th state variable """
	# load data
	Theta = np.loadtxt('Regression_Data/Theta.txt')
	dadt  = np.loadtxt('Regression_Data/a_dot.txt')
	nsamples, nfeatures = Theta.shape
	nn = dadt.shape[1]
 
	# average mean square error across the folds
	MSE_mean = np.zeros(len(alphas))
	MSE_std  = np.zeros(len(alphas))
	MSE_full = np.zeros(len(alphas))
	MSE_full_rel = np.zeros(len(alphas))

	# number of nonzero coefficients
	nnz = np.zeros(len(alphas))
	comm = MPI.COMM_WORLD
	# coefficients
	coeffs = np.zeros((len(alphas), nfeatures))

	for j, alpha in enumerate(alphas):
		model = linear_model.LassoCV(cv=kfolds,
						alphas=[alpha],
						fit_intercept=False,
						max_iter=3000,
						tol=1e-4).fit(Theta, dadt[:, i])
        
	
		print('Worker %d  :: doing alpha=%.2e :: completed %.2f %%\n' % (comm.Get_rank(), model.alpha_, 100.0*float(j+1)/len(alphas)))

		sys.stdout.flush()
		# apparently this mse_path is already taking into
		# account the whole dataset, so we do not need to multiply by kfolds
		coeffs[j]   = model.coef_
		MSE_mean[j] = np.sqrt(nsamples*np.mean(model.mse_path_))
		MSE_std[j]  = np.sqrt(np.std(nsamples*model.mse_path_))

		#MSE_full_rel[j] = np.mean(((np.dot(Theta, model.coef_) - dadt[:, i])**2)/np.linalg.norm(dadt[:, i])**2)
		MSE_full_rel[j] = np.mean(np.linalg.norm(np.dot(Theta, model.coef_) - dadt[:, i])/np.linalg.norm(dadt[:, i]))		
		
		#MSE_full[j] = np.mean((np.dot(Theta, model.coef_) - dadt[:, i])**2)		
		MSE_full[j] =     np.mean(np.linalg.norm(np.dot(Theta, model.coef_) - dadt[:, i]))
		
		nnz[j] = np.count_nonzero(model.coef_)

		# save data
		try:
			#shutil.rmtree('Regression_Results')
			os.mkdir('Regression_Results')
		except OSError:
			pass

		
		np.savetxt('Regression_Results/MSE_mean_%03d' %  i, MSE_mean,delimiter=' ')
		np.savetxt('Regression_Results/MSE_std_%03d'  % i, MSE_std,delimiter=' ')
		np.savetxt('Regression_Results/MSE_full_%03d' % i, MSE_full,delimiter= ' ')
		np.savetxt('Regression_Results/MSE_full_rel_%03d' % i, MSE_full_rel,delimiter= ' ')
		np.savetxt('Regression_Results/coeffs_%03d'   % i, coeffs,delimiter = ' ')
		np.savetxt('Regression_Results/nnz_%03d'   % i, nnz,delimiter = ' ')

		print('Done i = %03d\n' % i)
	return True

# dimension of the problem
dadt = np.loadtxt('Regression_Data/a_dot.txt')
nt,N = dadt.shape
print('Number of snapshots : ' + str(nt))
print('Number of modes : ' + str(N))

# Lasso penalisation factors
alphas = np.logspace(-15, -2, 50)

# number of folds
kfolds = 10

print('Starting Cross Validation with : ' + str(kfolds) + ' folds')

# get rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# every worker will do this number of cases
Neach = N/comm.Get_size()

# do job

for j in range(rank*int(np.ceil(Neach)), (rank+1)*int(np.ceil(Neach))):
	if j >= N : 
		print('Index exceedes dimension')
		pass
	else:
		print('INDEX ' + str(j))
		print('RANK ' + str(rank))
		print()
		work(j, kfolds, alphas)
	
info = get_info()

info.append(len(alphas))

np.savetxt('Regression_Results/alphas',alphas,delimiter = ' ')
np.savetxt('Regression_Results/info.txt',info,delimiter = ' ')


#work(0, kfolds, alphas)



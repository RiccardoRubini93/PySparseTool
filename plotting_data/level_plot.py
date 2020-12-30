import numpy as np
import pylab as pl
from copy import deepcopy
import pdb

def level_plot(beta,alpha,eta):

	eta_new = np.zeros((eta.shape[0],eta.shape[1],alpha.shape[0]))	

	for i in range(alpha.shape[0]):

		#eta_new[:,:,i] = alpha[i]*(beta[:,i].reshape(eta.shape[0],eta.shape[1]))
		eta_new[:,:,i] = alpha[i]*(beta[:,i].reshape((eta.shape[0],eta.shape[1]),order='F'))		

	eta_sum = np.zeros((eta.shape[0],eta.shape[1]))

	for i in range(eta_new.shape[2]):

		eta_sum = eta_sum + alpha[i]*(abs(eta_new[:,:,i])>0)*1

	return eta_sum*np.max(alpha)/np.max(eta_sum)
		

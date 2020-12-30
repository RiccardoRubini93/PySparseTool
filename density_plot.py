import numpy as np
import pylab as pl
from plotting_data.level_plot import *
from plotting_data.reshape_matrix import *
import sys

folder  = '_90_renorm'

eta  = np.genfromtxt('plotting_data'+folder+'/eta.txt')
beta = np.genfromtxt('plotting_data'+folder+'/beta.txt')
rho  = np.genfromtxt('plotting_data'+folder+'/density.txt')
L2   = np.genfromtxt('plotting_data'+folder+'/L2.txt')

n_modes = eta.shape[1]

rho_i = np.zeros((n_modes,len(rho)))

for i in range(len(rho)):

	eta_ = beta[:,i].reshape(eta.shape)

	for j in range(n_modes):
		rho_i[j,i] = np.count_nonzero(eta_[:,j])/eta_[:,j].size

#check the number of active modes throught the sparsification

N_a = list()

for i in range(len(rho)):

	_eta = beta[:,i].reshape(eta.shape)
	
	count = 0

	for j in range(_eta.shape[1]):

		if np.count_nonzero(_eta[1+n_modes:,j]) != _eta[1+n_modes:,j].size: count +=1

	
	N_a.append(count) 
		


pl.figure(1)

for i in range(rho_i.shape[0]):

	pl.plot(rho,rho_i[i,:],lw=1,color=pl.cm.binary(i/rho_i.shape[0]))
		
	pl.xlabel(r'$\rho$',fontsize = 20)
	pl.ylabel(r'$\rho_i$',fontsize = 20)	
	pl.axvline(x=rho[0], ymin=0, ymax=1,color='k')
	pl.axvline(x=rho[-1], ymin=0, ymax=1,color='k')

pl.plot(rho,rho_i[0,:],'b-', lw = 3, label = 'mode 1')
pl.plot(rho,rho_i[-1,:],'r-',lw = 3, label = 'mode N') 
pl.xlim([0,1])
pl.ylim([0,1])
pl.legend()

pl.show() 


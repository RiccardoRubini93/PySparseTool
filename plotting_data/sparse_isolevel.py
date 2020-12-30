import numpy as np
from reshape_matrix import *
from level_plot import *

beta = np.genfromtxt('beta.txt')
print('beta Loaded...')
eta = np.genfromtxt('eta.txt')
print('eta Loaded...')
alpha = np.genfromtxt('alpha.txt')
print('alpha Loaded...')
L2 = np.genfromtxt('L2_mean.txt')
print('L2 Loaded...')
rho = np.genfromtxt('density.txt')


print('Dataset loaded...')

n_modes = eta.shape[1]

eta_sum = level_plot(beta,L2,eta)

print('Isolevels generated...')

C = constant(eta_sum,n_modes)
L = linear(eta_sum,n_modes)
Q = quadratic(eta_sum,n_modes)

print('Matrices generated...')

#pl.figure(1)

for i in range(0,10):
	
	pl.figure(i)
	pl.imshow(np.log10(abs(Q[i,:,:])+1e-6),cmap = pl.cm.plasma,interpolation='none')
	pl.colorbar()
	pl.xticks([])
	pl.yticks([])	
	pl.title('Mode : ' + str(i+1))

pl.show()

print('End plotting...')




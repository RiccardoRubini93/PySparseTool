import numpy as np

def constant(eta,n_modes):

	c = eta[0,:]
		
	return c

def linear(eta,n_modes):

	L = np.zeros((n_modes,n_modes))

	L = eta[1:n_modes+1,:]

	return L

def quadratic(eta,n_modes):

	Q = np.zeros((n_modes,n_modes,n_modes))

	s_nl = n_modes+1		
		
	eta_nl = eta[s_nl:,:]

	for n in range(n_modes):

		count = 0
		for i in range(n_modes):
			for j in range(i,n_modes):
				
				Q[n,i,j] = eta_nl[count,n]
				count = count +1

	#simmetrize the matrices

	for i in range(n_modes):
		for j in range(n_modes):
			for k in range(n_modes):

				Q[i,k,j] = Q[i,j,k]


	return Q


	



		

import numpy as np
from utils.Generate_database import *
import os
import sys

n_modes = int(sys.argv[1])

a     = np.genfromtxt('Data/a.txt')[:,0:n_modes]
a_dot = np.genfromtxt('Data/a_dot.txt')[:,0:n_modes]

print('Dataset loaded')

print('Number of snapshots = ' + str(a.shape[0]))

print('Database generation...')

Theta = generate_database(a.T,a.shape[1])

print('Database ready...')

# save data
try:
	os.mkdir('Regression_Data')
except OSError:
	pass

np.savetxt('Regression_Data/Theta.txt',Theta,delimiter = ' ')
np.savetxt('Regression_Data/a.txt',a,delimiter = ' ')
np.savetxt('Regression_Data/a_dot.txt',a_dot,delimiter = ' ')



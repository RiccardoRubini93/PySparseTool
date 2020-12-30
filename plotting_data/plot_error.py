import numpy as np
import pylab as pl

L2 	= np.genfromtxt('L2.txt')
L2_std 	= np.genfromtxt('L2_std.txt')
L2_mean = np.genfromtxt('L2_mean.txt')

rho = np.genfromtxt('density.txt')


pl.figure(1)
pl.plot(rho,L2_mean,'b--o',lw=2)
pl.plot(rho,L2_mean + L2_std,'r-',lw=1)
pl.plot(rho,L2_mean - L2_std,'r-',lw=1)

pl.figure(2)
pl.plot(rho,L2,'b--o',lw=2)
pl.plot(rho,L2 + L2_std,'r-',lw=1)
pl.plot(rho,L2 - L2_std,'r-',lw=1)


pl.show()

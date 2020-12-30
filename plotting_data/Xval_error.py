import pylab as pl
import numpy as np

L2 = np.genfromtxt('L2.txt')
L2_m = np.genfromtxt('L2_mean.txt')
L2_std = np.genfromtxt('L2_std.txt')
rho = np.genfromtxt('density.txt')

pl.figure(1)
pl.plot(rho,L2,'r-o',lw=3)
pl.xlabel(r"$\rho$",fontsize=30)
pl.ylabel(r"$\epsilon$",fontsize=30)

pl.figure(2)
pl.plot(rho,L2_m,'r-o',lw=3)
pl.plot(rho,L2_m + 0.5*L2_std,'b--',lw = 1)
pl.plot(rho,L2_m - 0.5*L2_std,'b--',lw = 1)
pl.xlabel(r"$\rho$",fontsize=30)
pl.ylabel(r"$\epsilon$",fontsize=30)

pl.figure(3)
pl.plot(rho,L2_m,'r-o',lw=3,label = 'X-validated')
pl.plot(rho,L2_m + 0.5*L2_std,'b--',lw = 1)
pl.plot(rho,L2_m - 0.5*L2_std,'b--',lw = 1)
pl.xlabel(r"$\rho$",fontsize=30)
pl.ylabel(r"$\epsilon$",fontsize=30)
pl.plot(rho,L2 ,'g-o',lw=3,label = 'Sheer Regression')
pl.legend()

pl.show()

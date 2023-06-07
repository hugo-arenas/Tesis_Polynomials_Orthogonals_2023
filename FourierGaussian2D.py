#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:56:55 2023

@author: proman
"""

from matplotlib import pyplot as plt
import numpy as np

#####################1 D############################

# parametros
N=101 # siempre impar
L=10



# escalas
dx=L/N
du=1/(dx*N) # averiguar porque es esto!
LF=N*du

# dominios
x=np.linspace(-L/2,L/2,N)
x = np.reshape(x, (N,1))
y = np.reshape(x, (1,N))

#u=np.linspace(-LF/2,LF/2,N)

I =np.exp(-np.pi*(x*x + y*y) )

plt.figure()
plt.imshow(I,extent=[-L/2,L/2,-L/2,L/2])

FI = np.fft.fft2(I)/N
FI = np.fft.fftshift(FI)

FI/=np.pi

plt.figure()
plt.imshow(np.absolute(FI),extent=[-LF/2,LF/2,-LF/2,LF/2])


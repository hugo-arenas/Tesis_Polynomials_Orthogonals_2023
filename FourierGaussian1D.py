#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:56:55 2023

@author: proman
"""

from matplotlib import pyplot as plt
import numpy as np

#####################1 D############################

N=31
L=6

dx=L/N
du=1/(dx*N) # averiguar porque es esto!
LF=N*du

x=np.linspace(-L/2,L/2,N)
u=np.linspace(-LF/2,LF/2,N)

I =np.exp(-np.pi*x*x)

plt.figure()
plt.plot(x,I)

FI = np.fft.fftshift(I)
FI = np.fft.fft(FI)/np.sqrt(N)
FI = np.fft.fftshift(FI)

FI/=np.sqrt(np.pi)

plt.figure()
plt.plot(u,FI)


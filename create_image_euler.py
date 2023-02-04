import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from math import e, pi
from array import *

N = 101

ini = -5


array_x = np.linspace(ini,-ini,N)

array_x = np.reshape(array_x,(N,1))

array_y = np.reshape(array_x,(1,N))

img1 = np.exp(-pi*(array_x**2 + array_y**2))


fftimg1 = np.fft.fft2(img1)*pi/N
fftimg1 = np.fft.fftshift(fftimg1)

fig = plt.figure("image vs fft")
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

im1=ax1.matshow(img1)

im2=ax2.matshow(np.abs(fftimg1))

dx = (ini*-2)/N

du = 1/(dx*N) #¿porque es esto?

Lu = N*du

plt.show()

A = img1.flatten()

u0 = -Lu/2

du = np.linspace(0,N-1,N)*1/(dx*N)

du = du + u0

duzeros = np.zeros(N)

mu = duzeros[:, np.newaxis] + du

dv = mu.flatten()

du = np.sort(dv)

def dot(weights,x,y):
  return(np.ma.sum(x*np.ma.array(weights)*y))
  

def recurrence2d(matrix,y,x):
    if y == x and y == len(matrix):
        return matrix
    else:
        if y == x:
            if x - 1 < 0:
                matrix[y+1][x] = powdot(1,matrix[y][x],matrix[y][x])*matrix[y][x]
            else:
                matrix[y+1][x] = dot(1,matrix[y][x],matrix[y][x])*matrix[y][x] - 
    



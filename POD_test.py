import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from math import e, pi
from array import *
import time

def dot(weights,x,y):
  return(np.sum(x*weights*np.conjugate(y)))

def norm(weights,x):
  return(np.sqrt(np.sum(weights*np.absolute(x)**2)))

def dot2x2(weights,x,y):
  f,c,d = y.shape
  #aux = np.ones(shape=(f,c,d),dtype=float)
  aux = np.empty((f,c,d),dtype=float)
  aux = aux + 1.0
  w = aux*weights
  mul = x*w*np.conjugate(y)
  npsum = np.sum(mul,axis=2)
  #npsum = np.reshape(npsum,(f,c,1))
  npsum = npsum[:, :, np.newaxis]
  npsum = aux*npsum

  return npsum

def norm2x2(weights,x):
  f,c,d = x.shape
  #aux = np.ones(shape=(f,c,d),dtype=float)
  aux = np.empty((f,c,d),dtype=float)
  aux = aux + 1.0
  w = aux*weights
  mul = w*np.absolute(x)**2
  npsum = np.sum(mul,axis=2)
  npsum = np.sqrt(npsum)
  #npsum = np.reshape(npsum,(f,c,1))
  npsum = npsum[:, :, np.newaxis]
  npsum = aux*npsum
  return npsum

def old_r2d(z,w,n):
    P =np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = (z**k)*np.conjugate(z**j)
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
  
    for j in range(0,n):
        for k in range(0,n):        
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
            h = 0
            for x in range(j,n):
                if (x==j):
                    h=k+1
                else:
                    h=0
                for y in range(h,n):
                    P[y,x,:] = P[y,x,:] - dot(w,P[k,j,:],P[y,x,:])*P[k,j,:]
                    P[y,x,:] = P[y,x,:]/norm(w,P[y,x,:])
                    
           
    return P

def new_r2d(z,w,n):
    P = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    M = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = (z**k)*np.conjugate(z**j)
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
    
    for j in range(0,n):
        for k in range(0,n):
            
                P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
                sub_p = np.array(P[k,j,:])
                M[k,j,:] = np.array(P[k,j,:])
                P = P - dot2x2(w,sub_p,P)*sub_p
                P = P/norm2x2(w,P)
    return M
np.set_printoptions(threshold=np.inf)

N = 31

S = 30

ini = -1.5

factor = 3

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
plt.show()

dx = (ini*2)/N
du = 1/(dx*N)
Lu = N*du

u0 = -Lu/2
du = np.linspace(u0,-u0,N)
u,v = np.meshgrid(du,du)

z = u + 1j*v

    
w = np.ones((N,N))

start_time = time.time()

P1 = old_r2d(z.flatten(), w.flatten(), S)

print(time.time() - start_time)
start_time = time.time()

P2 = new_r2d(z.flatten(), w.flatten(), S)

print(time.time() - start_time)

print(np.sum(P1-P2))
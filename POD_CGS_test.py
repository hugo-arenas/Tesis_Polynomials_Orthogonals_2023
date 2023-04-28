#import numpy as np
import cupy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from math import e, pi
from array import *
import time

def dot(weights,x,y):
  return(np.sum(x*weights*np.conjugate(y)))

def norm(weights,x):
  return(np.sqrt(np.sum(weights*np.absolute(x)**2)))

<<<<<<< HEAD
def dot2x2(weights,x,y,t):
  f,c,d = y.shape
  aux = np.ones(shape=(f,c,d),dtype=float)
  w = aux*weights
  if t==0:
    mul = y*w*np.conjugate(x)
  else:
    mul = x*w*np.conjugate(y)
=======
def dot2x2(weights,x,y):
  f,c,d = y.shape
  aux = np.ones(shape=(f,c,d),dtype=float)
  w = aux*weights
  mul = x*w*np.conjugate(y)
>>>>>>> master
  npsum = np.sum(mul,axis=2)
  npsum = np.reshape(npsum,(f,c,1))
  npsum = aux*npsum

  return npsum
  
<<<<<<< HEAD
#def dot2x2C(weights,x,y):
#  f,c,d = y.shape
#  aux = np.ones(shape=(f,c,d),dtype=float)
#  w = aux*weights
#  mul = y*w*np.conjugate(x)
#  npsum = np.sum(mul,axis=2)
#  npsum = np.reshape(npsum,(f,c,1))
#  npsum = aux*npsum
=======
def dot2x2C(weights,x,y):
  f,c,d = y.shape
  aux = np.ones(shape=(f,c,d),dtype=float)
  w = aux*weights
  mul = y*np.conjugate(x)*w
  npsum = np.sum(mul,axis=2)
  npsum = np.reshape(npsum,(f,c,1))
  npsum = aux*npsum
>>>>>>> master

  return npsum

def norm2x2(weights,x):
  f,c,d = x.shape
  aux = np.ones(shape=(f,c,d),dtype=float)
  w = aux*weights
  mul = w*np.absolute(x)**2
  npsum = np.sum(mul,axis=2)
  npsum = np.sqrt(npsum)
  npsum = np.reshape(npsum,(f,c,1))
  npsum = aux*npsum
  return npsum

def old_r2d(z,w,n):
    P =np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = (z**k)*np.conjugate(z**j)
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
  
    for j in range(0,n): 
<<<<<<< HEAD
        for k in range(0,n):
            P[k,j,:]=P[k,j,:]/norm(w,P[k,j,:])
            for x in range(0,j+1):
=======
        for k in range(0,n): 
            P[k,j,:]=P[k,j,:]/norm(w,P[k,j,:])
            for x in range(0,j+1): 
>>>>>>> master
                for y in range(0,n):
                    if (j!=x or k!=y):
                        P[k,j,:] = P[k,j,:] - dot(w,P[k,j,:],P[y,x,:])*P[y,x,:]
                    else:
                        break
                P[k,j,:]=P[k,j,:]/norm(w,P[k,j,:])
    return P

def new_r2d(z,w,n):
    P = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    M = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    D = np.zeros(z.size,dtype=np.complex128)
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = (z**k)*np.conjugate(z**j)
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
<<<<<<< HEAD
    P = P/norm2x2(w,P)
    for j in range(0,n):
        for k in range(0,n):
            if k==0 and j==0:
                P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
                D=np.array(P[k,j,:])
                M[k,j,:] = np.array(P[k,j,:])
            else:
                if (k==1 and j>0):
                    P=P/norm2x2(w,P)   
                P = P - dot2x2(w,D,P,0)*D
                P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
                if (k==0):
                    P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
                D=np.array(P[k,j,:])
                M[k,j,:] = np.array(P[k,j,:])
    return M
#np.set_printoptions(threshold=np.inf)

N = 31

S = 30
=======
    
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
            if k==0 and j==0:
                D=np.array(P[k,j,:])
                M[k,j,:] = np.array(P[k,j,:])
            else:
                if k==1 and j>0:
                    E = np.array(P[:,j:n,:])
                    E = E/norm2x2(w,E)
                    P[:,j:n,:] = np.array(E)
                    
                P = P - dot2x2C(w,D,P)*D
                P[k,j,:] =  P[k,j,:]/norm(w,P[k,j,:])
                D=np.array(P[k,j,:])
                M[k,j,:] = np.array(P[k,j,:])
    
    return M
#np.set_printoptions(threshold=np.inf)

N = 7

S = 6
>>>>>>> master

ini = -1

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

im1=ax1.matshow(np.asnumpy(img1))

im2=ax2.matshow(np.asnumpy(np.absolute(fftimg1)))
#plt.show()

dx = (ini*2)/N
du = 1/(dx*N)
Lu = N*du

u0 = -Lu/2
du = np.linspace(ini,-ini,N)
u,v = np.meshgrid(du,du)

z = u + 1j*v

    
w = np.ones((N,N))

start_time = time.time()

P1 = old_r2d(z.flatten(), w.flatten(), S)

print("Tiempo de ejecución orden N^4:", time.time() - start_time)
start_time = time.time()

P2 = new_r2d(z.flatten(), w.flatten(), S)

print("Tiempo de ejecución orden N^2 (nuevo):", time.time() - start_time)

idx = P1==P2
<<<<<<< HEAD
#print("Diferencia parcial:", np.asnumpy(np.sum((P1[:,3,:]-P2[:,3,:]),axis=1)))
print("Diferencia de resultados:", np.asnumpy(np.sum(P1-P2)))
#print("Diferencia de resultados:", idx)
=======
#print("Diferencia de resultados:", np.asnumpy(np.sum(P1-P2)))
print("Diferencia de resultados:", idx)
>>>>>>> master

plt.show()
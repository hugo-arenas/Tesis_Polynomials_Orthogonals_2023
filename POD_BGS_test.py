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

def dot2x2(weights,x,y,t):
  f,c,d = y.shape
  aux = np.ones(shape=(f,c,d),dtype=float)
  w = aux*weights
  if t==0:
    mul = y*w*np.conjugate(x)
  else:
    mul = x*w*np.conjugate(y)
  npsum = np.sum(mul,axis=2)
  npsum = np.reshape(npsum,(f,c,1))
  npsum = aux*npsum

  return npsum
  
#def dot2x2C(weights,x,y):
#  f,c,d = y.shape
#  aux = np.ones(shape=(f,c,d),dtype=float)
#  w = aux*weights
#  mul = y*w*np.conjugate(x)
#  npsum = np.sum(mul,axis=2)
#  npsum = np.reshape(npsum,(f,c,1))
#  npsum = aux*npsum

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
    
    large1 = np.array(range(0,n))
    large2 = np.ones(n-1)*(n-1)
    large = np.concatenate((large1,large2),axis=0)
    k2 = 0
    for k in large:
        if (k2!=n-1):
            l=0
        else:
            l=l+1
        k = int(k)
        k2 = k
        for j in range(l,k+1):
            no=norm(w,P[k-j+l,j,:])
            P[k-j+l,j,:]/=no
            y3 = 0
            for y in large:
                if (y3!=n-1):
                    l2=0
                else:
                    l2=l2+1
                y = int(y)
                y3 = y
                for x in range(l2,y+1):

                    if (y-x+l2 < k-j+l and x>j and y+l2==k+l) or (y+l2>k+l):
                        prod=dot(w,P[k-j+l,j,:],P[y-x+l2,x,:])
                        P[y-x+l2,x,:] -= prod*P[k-j+l,j,:]
                        no=norm(w,P[y-x+l2,x,:])
                        P[y-x+l2,x,:]/=no
            y3 = 0
            l2=0
            no = norm(w,P[k-j+l,j,:])
            P[k-j+l,j,:] = P[k-j+l,j,:]/no
            for y in large:
                if (y3!=n-1):
                    l2=0
                else:
                    l2=l2+1
                y = int(y)
                y3 = y
                for x in range(l2,y+1):
                    if (y-x+l2 > k-j+l and x<j and y+l2==k+l) or (y+l2<k+l):
                        prod=dot(w,P[k-j+l,j,:],P[y-x+l2,x,:])
                        P[k-j+l,j,:] = P[k-j+l,j,:] - prod*P[y-x+l2,x,:]
                    else:
                        break
                if y <=k:
                    P[k-j+l,j,:]=P[k-j+l,j,:]/norm(w,P[k-j+l,j,:])
    return P

def new_r2d(z,w,n):
    P = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    V = np.ones(shape=(n,n,1),dtype=int)
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = (z**k)*np.conjugate(z**j)
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
            
    large1 = np.array(range(0,n))
    large2 = np.ones(n-1)*(n-1)
    large = np.concatenate((large1,large2),axis=0)
    k2 = 0
    l = 0
    for k in large:
        if (k2!=n-1):
            l=0
        else:
            l=l+1
        k = int(k)
        k2 = k
        for j in range(l,k+1):
            no=norm(w,P[k-j+l,j,:])
            P[k-j+l,j,:] = P[k-j+l,j,:]/no          
            V[k-j+l,j,:] = 0
            sub_p = np.array(P[k-j+l,j,:])            
            dot_data = dot2x2(w,sub_p,P*V,1)          
            P = P - dot_data*sub_p         
            no_data = norm2x2(w,P)        
            no_data[(V*np.ones(z.size,dtype=int)) == 0] = 1           
            P = P/no_data
            
    V = np.ones(shape=(n,n,1),dtype=int)
    k2=0
    l=0
    p_data = np.ones(1,dtype=np.complex128)
    P = P/norm2x2(w,P)
    for k in large:
        if (k2!=n-1):
            l=0
        else:
            l=l+1
        k = int(k)
        k2 = k
        for j in range(l,k+1):
            if k==0 and j==0:
                no=norm(w,P[k-j+l,j,:])
                P[k-j+l,j,:] = P[k-j+l,j,:]/no
                V[k-j+l,j,:] = 0
                p_data = np.array(P[k-j+l,j,:])

            else:
                if (j==1+l and k>=0):
                    no_data = norm2x2(w,P)
                    no_data[(V*np.ones(z.size,dtype=int)) == 0] = 1                                   
                    P=P/no_data            
                dot_data = dot2x2(w,p_data,P*V,0)
                P = P - dot_data*p_data
                no=norm(w,P[k-j+l,j,:])
                P[k-j+l,j,:] =  P[k-j+l,j,:]/no             
                if (j==l):
                    no=norm(w,P[k-j+l,j,:])
                    P[k-j+l,j,:] = P[k-j+l,j,:]/no
                V[k-j+l,j,:] = 0
                p_data = np.array(P[k-j+l,j,:])

    return P
#np.set_printoptions(threshold=np.inf)

N = 31

S = 30

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
#print("Diferencia parcial:", np.asnumpy(np.sum((P1[:,3,:]-P2[:,3,:]),axis=1)))
print("Diferencia de resultados:", np.asnumpy(np.sum(P1-P2)))
#print("Diferencia de resultados:", idx)

plt.show()
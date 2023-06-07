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
  mul = np.zeros(1,dtype=np.complex128)
  if t==0:
    mul = y*w*np.conjugate(x)
  else:
    mul = x*w*np.conjugate(y)
  npsum = np.sum(mul,axis=2)
  npsum = np.reshape(npsum,(f,c,1))
  npsum = aux*npsum

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
        for k in range(0,n):
            if (k==n-1):
                j2 = j + 1
            else:
                j2 = j          
            no = norm(w,P[k,j,:])
            P[k,j,:] = P[k,j,:]/no
            
            for x in range(j2,n):
                if (k==n-1 and x==j2):
                    k2=0
                elif (x==j2):
                    k2=k+1
                else:
                    k2=0
                for y in range(k2,n):
                    prod=dot(w,P[k,j,:],P[y,x,:])
                    P[y,x,:] -= prod*P[k,j,:]
                    no=norm(w,P[y,x,:])
                    P[y,x,:]/=no
            no = norm(w,P[k,j,:])
            P[k,j,:] = P[k,j,:]/no    
            for x in range(0,j+1):
                for y in range(0,n):
                    if (x!=j or y!=k):
                        prod=dot(w,P[k,j,:],P[y,x,:])
                        P[k,j,:] -= prod*P[y,x,:]
                    else:
                        break
                no=norm(w,P[k,j,:])
                P[k,j,:]/=no
    P2 = np.array(P)
    #for j in range(0,n):
    #    for k in range(0,n):
        
    #        no = norm(w,P[k,j,:])
    #        P[k,j,:] = P[k,j,:]/no
            
    #        for x in range(0,j+1):
    #            for y in range(0,n):
    #                if (x!=j or y!=k):
    #                    prod=dot(w,P[k,j,:],P[y,x,:])
    #                    P[k,j,:] -= prod*P[y,x,:]
    #                else:
    #                    break
    #            no=norm(w,P[k,j,:])
    #            P[k,j,:]/=no
    return P,P2

def new_r2d(z,w,n):
    P = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    P_aux = np.ones(shape=(n,n,z.size),dtype=np.complex128)
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = (z**k)*np.conjugate(z**j)
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
           
    for j in range(0,n):
        for k in range(0,n):
            no=norm(w,P[k,j,:])
            P[k,j,:] = P[k,j,:]/no  
            sub_p = np.array(P[k,j,:]) 
            P_aux[k,j,:] = np.array(P[k,j,:])    
            dot_data = dot2x2(w,sub_p,P,1)          
            P = P - dot_data*sub_p         
            no_data = norm2x2(w,P)                 
            P = P/no_data
    P2 = np.array(P_aux)      
    P = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    k_aux=0
    j_aux=0
    P_aux = P_aux/norm2x2(w,P_aux)
    for j in range(0,n):
        for k in range(0,n):
            if k==0 and j==0:
                no=norm(w,P_aux[k,j,:])
                P_aux[k,j,:] = P_aux[k,j,:]/no
                P[k,j,:] = np.array(P_aux[k,j,:])
                k_aux=k
                j_aux=j
            else:
                if (k==1 and j>0):
                    no_data = norm2x2(w,P_aux)               
                    P_aux = P_aux/no_data
                    
                dot_data = dot2x2(w,P[k_aux,j_aux,:],P_aux,0)          
                P_aux = P_aux - dot_data*P[k_aux,j_aux,:]
                no=norm(w,P_aux[k,j,:])
                P_aux[k,j,:] = P_aux[k,j,:]/no
                if (k==0):
                    no=norm(w,P_aux[k,j,:])
                    P_aux[k,j,:] = P_aux[k,j,:]/no
                P[k,j,:] = np.array(P_aux[k,j,:])
                k_aux=k
                j_aux=j
            #k_aux=k
            #j_aux=j
    return P,P2
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

P1,P1_old = old_r2d(z.flatten(), w.flatten(), S)

print("Tiempo de ejecución orden N^4:", time.time() - start_time)
start_time = time.time()

P2,P2_old = new_r2d(z.flatten(), w.flatten(), S)

print("Tiempo de ejecución orden N^2 (nuevo):", time.time() - start_time)

idx = P1==P2
#print("Diferencia parcial:", np.asnumpy(np.sum((P1[:,3,:]-P2[:,3,:]),axis=1)))
print("Diferencia de resultados (medio paso):", np.asnumpy(np.sum(P1_old-P2_old)))
print("Diferencia de resultados:", np.asnumpy(np.sum(P1-P2)))
#print("Diferencia de resultados:", idx)

plt.show()
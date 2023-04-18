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

def recurrence2d(z,w,n):
    P =np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    P[0,0,:]=1
    P[0,0,:]=P[0,0,:]/norm(w,P[0,0,:])
    
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = (z**k)*np.conjugate(z**j)
            
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
                    #print(k,j,y,x)
                    P[y,x,:] = P[y,x,:] - dot(w,P[k,j,:],P[y,x,:])*P[k,j,:]
           
    return P
np.set_printoptions(threshold=np.inf)

N = 101

ini = -1.5


array_x = np.linspace(ini,-ini,N)

array_x = np.reshape(array_x,(N,1))

array_y = np.reshape(array_x,(1,N))

img1 = np.exp(-pi*(array_x**2 + array_y**2))

#img1, uvsample, noise_uv, uvsample_noise, uv = sim(5,101,np.ones((N,N)))

fftimg1 = np.fft.fft2(img1)*pi/N
fftimg1 = np.fft.fftshift(fftimg1)

fig = plt.figure("image vs fft")
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

im1=ax1.matshow(img1)

im2=ax2.matshow(np.abs(fftimg1))
plt.show()

dx = (ini*2)/N
du = 1/(dx*N) #¿porque es esto?
Lu = N*du

u0 = -Lu/2
du = np.linspace(u0,-u0,N)
u,v = np.meshgrid(du,du)

z = u + 1j*v

du2 =  np.linspace(u0,-u0,N)/(dx*N)
duzeros = np.zeros(N)
mu = duzeros[:, np.newaxis] + du2
dv2 = mu.flatten()
du2 = np.sort(dv2)
z2 = du2 + 1j*dv2

    
w = np.ones((N,N))

start_time = time.time()

P = recurrence2d(z.flatten(), w.flatten(), N-1)

print(time.time() - start_time)
# Polynomial correlation

K=np.arange(N-1)
J=np.arange(N-1)
K,J=np.meshgrid(K,J)
#idx=K>=J
idx=K>=J # case with diagonal
idx = np.reshape(idx,(N-1,N-1,1))
idx = np.ones((N-1,N-1,N*N))*idx
idx = idx==1
pp=P[idx]
pp =np.reshape(pp,(int(N*(N-1)/2),N*N))
#pp =np.reshape(pp,(int(N*(N-1)/2),M*M)) # case without diagonal
corr=np.dot(pp,np.conjugate(pp.T))

fig=plt.figure("corr")
im=plt.imshow(np.absolute(corr))
plt.colorbar(im)

# we exibit orthononality errors (no diagonal) bellow diagonal
cor=corr-np.diag(np.diag(corr))
fig=plt.figure()
im=plt.imshow(np.absolute(cor))
plt.colorbar(im)

# we exhibit orthogonality for whole matrix 
pp1=np.reshape(P,((N-1)*(N-1),N*N))
corr1=np.dot(pp1,np.conjugate(pp1.T))
cor1=corr1-np.diag(np.diag(corr1))
fig=plt.figure()
im=plt.imshow(np.absolute(cor1))
plt.colorbar(im)

# we exibit lack of symmetry
pp2 = np.reshape(P,((N-1),(N-1),N*N))
dpp = np.absolute(pp2 - np.transpose(pp2,axes=(1,0,2)))
dpp = np.reshape(dpp,((N-1)*(N-1),N*N))
ndpp= np.dot(dpp,np.conjugate(dpp.T))
fig=plt.figure()
im=plt.imshow(np.absolute(ndpp))
plt.colorbar(im)

# looking for different polinomials shape
P=np.reshape(P,((N-1),(N-1),N,N))

fig=plt.figure(); im=plt.imshow(np.absolute(P[2,2,:,:])); plt.colorbar(im)

fig=plt.figure(); im=plt.imshow(np.absolute(P[6,3,:,:])); plt.colorbar(im) # this exhibit symmetry diferences
fig=plt.figure(); im=plt.imshow(np.absolute(P[3,6,:,:])); plt.colorbar(im)

fig=plt.figure(); im=plt.imshow(np.real(P[25,18,:,:])); plt.colorbar(im)
fig=plt.figure(); im=plt.imshow(np.real(P[18,25,:,:])); plt.colorbar(im)

plt.show()
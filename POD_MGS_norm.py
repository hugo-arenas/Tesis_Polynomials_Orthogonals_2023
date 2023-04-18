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

def recurrence2d(z,z_target,w,n):
    P =np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    P_target = np.zeros(shape=(n,n,z_target.size),dtype=np.complex128)
    w_target = np.ones(z_target.size,dtype=float)
    
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = (z**k)*np.conjugate(z**j)
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
            P_target[k,j,:] = (z_target**k)*np.conjugate(z_target**j)
            P_target[k,j,:] = P_target[k,j,:]/norm(w_target,P_target[k,j,:])
            
            
    for j in range(0,n):
        for k in range(0,n):        
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
            P_target[k,j,:] = P_target[k,j,:]/norm(w_target,P_target[k,j,:])
            h = 0
            for x in range(j,n):
                if (x==j):
                    h=k+1
                else:
                    h=0
                for y in range(h,n):
                    P[y,x,:] = P[y,x,:] - dot(w,P[k,j,:],P[y,x,:])*P[k,j,:]
                    P[y,x,:] = P[y,x,:]/norm(w,P[y,x,:])
                    P_target[y,x,:] = P_target[y,x,:] - dot(w_target,P_target[k,j,:],P_target[y,x,:])*P_target[k,j,:]
                    P_target[y,x,:] = P_target[y,x,:]/norm(w_target,P_target[y,x,:])
                    
           
    return P, P_target
np.set_printoptions(threshold=np.inf)

N = 31

ini = -1.5

factor = 3


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
du = 1/(dx*N)
Lu = N*du

u0 = -Lu/2
du = np.linspace(u0,-u0,N)
u,v = np.meshgrid(du,du)

z = u + 1j*v

du = np.linspace(u0,-u0,N*factor)
u,v = np.meshgrid(du,du)
z_target = u + 1j*v

#du2 =  np.linspace(u0,-u0,N)/(dx*N)
#duzeros = np.zeros(N)
#mu = duzeros[:, np.newaxis] + du2
#dv2 = mu.flatten()
#du2 = np.sort(dv2)
#z2 = du2 + 1j*dv2

    
w = np.ones((N,N))

start_time = time.time()

P, P_target = recurrence2d(z.flatten(), z_target.flatten(), w.flatten(), N-1)

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
title="Polynomials Correlation half matrix below main counter diagonal"

K=np.arange(N-1)
one = np.ones(N-1)
K=np.reshape(K,(N-1,1))
one=np.reshape(one,(1,N-1))
K=K*one
J=np.arange(N-1)
J=np.reshape(J,(1,N-1))
one=np.reshape(one,(N-1,1))
J=J*one
Idx = K+J<=N-2

one=np.ones(shape=(N-1,N-1,N*N))
Idx=np.reshape(Idx,(N-1,N-1,1))
Idx=Idx*one
Idx=Idx==1
pp2=P[Idx]
pp2 = np.reshape(pp2,(int(N*(N-1)/2),N*N))
corr2=np.dot(pp2,np.conjugate(pp2.T))
cor2=corr2-np.diag(np.diag(corr2))

fig=plt.figure(title)
plt.title(title)
im=plt.imshow(np.absolute(cor2))
plt.colorbar(im)


# we exibit lack of symmetry kj <-> jk
pp2 = np.reshape(P,(N-1,N-1,N*N))
dpp = np.absolute(pp2 - np.transpose(pp2,axes=(1,0,2))) 
dpp = np.reshape(dpp,((N-1)*(N-1),N*N))
ndpp= np.dot(dpp,np.conjugate(dpp.T))
title="Correlation of differences kj - jk"
fig=plt.figure(title)
plt.title(title)
im=plt.imshow(np.absolute(ndpp))
plt.colorbar(im)

# looking for different polinomials shape
P=np.reshape(P,(N-1,N-1,N,N))
P_target=np.reshape(P_target,(N-1,N-1,factor*N,factor*N))

title="Absolute value of P_2,2"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.absolute(P[2,2,:,:])); plt.colorbar(im)

title="Absolute value of P_6,3"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.absolute(P[6,3,:,:])); plt.colorbar(im) # this exhibit symmetry diferences
title="Absolute value of P_3,6"; fig=plt.figure(title); plt.title(title);  im=plt.imshow(np.absolute(P[3,6,:,:])); plt.colorbar(im)

title="Absolute value of extrapolated P_6,3"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.absolute(P_target[6,3,:,:])); plt.colorbar(im) # this exhibit symmetry diferences
title="Absolute value of extrapolated P_3,6"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.absolute(P_target[3,6,:,:])); plt.colorbar(im)

title="Real part of P_25,18"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(P[25,18,:,:])); plt.colorbar(im)
title="Real part of P_18,25"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.real(P[18,25,:,:])); plt.colorbar(im)
title="Absolute value of extrapolated P_18,25"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.absolute(P_target[18,25,:,:])); plt.colorbar(im)

plt.show()
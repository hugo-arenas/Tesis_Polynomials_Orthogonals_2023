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

def dot2x2(weights,x,y):
  f,c,d = y.shape
  aux = np.ones(shape=(f,c,d),dtype=float)
  w = aux*weights
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
  
def recurrence2d(z,w,n,i):
    P = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    A = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    B = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    M = 0.0 + 0.0j
    f, c = i.shape
    Ig = np.zeros(shape=(f,c),dtype=np.complex128)
    std_a = np.zeros(1,dtype=np.complex128)
    iaux = np.array(i)
    D = np.zeros(z.size,dtype=np.complex128)
    
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = (z**k)*np.conjugate(z**j)
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
            
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
            sub_p = np.array(P[k,j,:])
            A[k,j,:] = np.array(P[k,j,:])
            P = P - dot2x2(w,sub_p,P)*sub_p
            P = P/norm2x2(w,P)
    '''       
    for j1 in range(0,n): # column
        for k1 in range(0,n): # row element to be substracted from the rest
        # preparing the first element from current k1,j1
        # ensure the case when we reach the bottom

            no=norm(w,A[k1,j1,:])
            A[k1,j1,:]/=no  
            # second pass
            for j2 in range(0,j1+1): #origin column
                for k2 in range(0,n): # origin row
                    #print(k1,j1,k2,j2)
                    if (j1!=j2 or k1!=k2):
                        prod=dot(w,A[k1,j1,:],A[k2,j2,:])
                        A[k1,j1,:] -= prod*A[k2,j2,:]
                    else:
                        break
                no=norm(w,A[k1,j1,:])
                A[k1,j1,:]/=no
            M = dot(w,A[k,j,:],iaux.flatten())
            Bsub = np.reshape(A,(n,n,f,c))
            Ig = Ig + M*Bsub[k,j,:,:]
            if j==0 and k == 0:
                std = np.std(i)
                std_a[0] = std
            else:
                iaux = iaux - M*Bsub[k,j,:,:]
                std = np.std(iaux)
                std_a = np.concatenate((std_a,np.array([std])),axis=0)
    '''
    for j in range(0,n):
        for k in range(0,n):
            A[k,j,:] = A[k,j,:]/norm(w,A[k,j,:])
            if k==0 and j==0:
                D=np.array(A[k,j,:])
                B[k,j,:] = np.array(A[k,j,:])
            else:
                if k==1 and j>0:
                    print(j)
                    E = np.array(A[:,j:n,:])
                    E = E/norm2x2(w,E)
                    A[:,j:n,:] = np.array(E)
                    
                A = A - dot2x2(w,D,A)*D
                A[k,j,:] =  A[k,j,:]/norm(w,A[k,j,:])
                D=np.array(A[k,j,:])
                B[k,j,:] = np.array(A[k,j,:])
            M = dot(w,B[k,j,:],iaux.flatten())
            Bsub = np.reshape(B,(n,n,f,c))
            Ig = Ig + M*Bsub[k,j,:,:]
            if j==0 and k == 0:
                std = np.std(i)
                std_a[0] = std
            else:
                iaux = iaux - M*Bsub[k,j,:,:]
                std = np.std(iaux)
                std_a = np.concatenate((std_a,np.array([std])),axis=0)
    
    return B, Ig, std_a

N = 31
S = 30

ini = -1

#factor = 3

array_x = np.linspace(ini,-ini,N)

array_x = np.reshape(array_x,(N,1))

array_y = np.reshape(array_x,(1,N))

img1 = np.exp(-pi*(array_x**2 + array_y**2))

fftimg1 = np.fft.fft2(img1)#*pi/N
fftimg1 = np.fft.fftshift(fftimg1)

fig = plt.figure("image vs fft")
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

im1=ax1.matshow(np.asnumpy(img1))

im2=ax2.matshow(np.asnumpy(np.absolute(fftimg1)))

du = np.linspace(ini,-ini,N)
u,v = np.meshgrid(du,du)
z = u + 1j*v

#u = np.reshape(np.linspace(ini,-ini,N),(N,1)) 
#v = np.reshape(np.linspace(ini,-ini,N),(1,N)) 
#z= u+1j*v

w = np.ones((N,N))

start_time = time.time()

P, Ig, std_a = recurrence2d(z.flatten(), w.flatten(), S, img1)

print(time.time() - start_time)
# Polynomial correlation

K=np.arange(S)
J=np.arange(S)
K,J=np.meshgrid(K,J)
#idx=K>=J
idx=K>=J # case with diagonal
idx = np.reshape(idx,(S,S,1))
idx = np.ones((S,S,N*N))*idx
idx = idx==1
pp=P[idx]
pp =np.reshape(pp,(int(S*(S+1)/2),N*N))
#pp =np.reshape(pp,(int(N*(N-1)/2),M*M)) # case without diagonal
corr=np.dot(pp,np.conjugate(pp.T))

fig=plt.figure("corr")
im=plt.imshow(np.asnumpy(np.absolute(corr)))
plt.colorbar(im)

'''
# we exibit orthononality errors (no diagonal) bellow diagonal
cor=corr-np.diag(np.diag(corr))
fig=plt.figure()
im=plt.imshow(np.asnumpy(np.absolute(cor)))
plt.colorbar(im)

# we exhibit orthogonality for whole matrix 
title="Polynomials Correlation half matrix below main counter diagonal"

K=np.arange(S)
one = np.ones(S)
K=np.reshape(K,(S,1))
one=np.reshape(one,(1,S))
K=K*one
J=np.arange(S)
J=np.reshape(J,(1,S))
one=np.reshape(one,(S,1))
J=J*one
Idx = K+J<=S-1

one=np.ones(shape=(S,S,N*N))
Idx=np.reshape(Idx,(S,S,1))
Idx=Idx*one
Idx=Idx==1
pp2=P[Idx]
pp2 = np.reshape(pp2,(int(S*(S+1)/2),N*N))
corr2=np.dot(pp2,np.conjugate(pp2.T))
cor2=corr2-np.diag(np.diag(corr2))

fig=plt.figure(title)
plt.title(title)
im=plt.imshow(np.asnumpy(np.absolute(cor2)))
plt.colorbar(im)


# we exibit lack of symmetry kj <-> jk
pp2 = np.reshape(P,(S,S,N*N))
dpp = np.absolute(pp2 - np.transpose(pp2,axes=(1,0,2))) 
dpp = np.reshape(dpp,((S)*(S),N*N))
ndpp= np.dot(dpp,np.conjugate(dpp.T))
title="Correlation of differences kj - jk"
fig=plt.figure(title)
plt.title(title)
im=plt.imshow(np.asnumpy(np.absolute(ndpp)))
plt.colorbar(im)

# looking for different polinomials shape
P=np.reshape(P,(S,S,N,N))
#P_target=np.reshape(P_target,(N-1,N-1,factor*N,factor*N))

#M = np.zeros(shape=(N-1,N-1),dtype=np.complex128)
#Ig = np.zeros(shape=(N,N),dtype=np.complex128)

#for x in range(0,N-1):
#    for y in range(0,N-1):
#        if x+y<N-1:
#            M[y,x] = dot(w,P[y,x,:,:],fftimg1)

#for x in range(0,N):
#    for y in range(0,N):
#        Ig[y,x] = np.sum(M*P[:,:,y,x])

#I = np.fft.fftshift(Ig)

I = np.fft.ifft2(Ig)
I = np.fft.fftshift(I)

'''

print()

residual = Ig - img1

#title="Absolute value of P_2,2"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(P[2,2,:,:]))); plt.colorbar(im)

#title="Absolute value of P_6,3"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(P[6,3,:,:]))); plt.colorbar(im)
#title="Absolute value of P_3,6"; fig=plt.figure(title); plt.title(title);  im=plt.imshow(np.asnumpy(np.absolute(P[3,6,:,:]))); plt.colorbar(im)

#title="Real part of P_25,18"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.real(P[25,18,:,:]))); plt.colorbar(im)
#title="Real part of P_18,25"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.real(P[18,25,:,:]))); plt.colorbar(im)
title="Model"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(Ig)))

plt.savefig("modelo.png")
#title="Result"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(I)))
title="Residual"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(residual)))
title="Desviation Standar"; fig=plt.figure(title); plt.title(title); plt.plot(np.asnumpy(std_a))

plt.show()


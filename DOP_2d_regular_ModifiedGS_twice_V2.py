#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 09 2023

2d DOP on regular sampling
using modified gram-schmidt
check orthogonality, recurrence, error map

@author: pablo.roman.a@usach.cl


"""

#import numpy as np
import cupy as np
from matplotlib import pyplot as plt



"""
Return a dot product with weight
"""
def dot(weights,x,y):
  return(np.sum(x*weights*np.conjugate(y)))

"""
Return the weighted norm
"""
def norm(weights,x):
  return(np.sqrt(np.sum(weights*np.absolute(x)**2)))


"""
Return the variance per location
P: polynomial matrix evaluated on given domain (axis1)
   axis0 correspond to the degre.
"""
def errorMap(P) :
  return(np.sum(np.absolute(P)**2,axis=0))



'''
Generating discrete orthornormal polynomials on 2d Hermitean realm
Z : uv sampling coordinates, considered as a complex number and is a vector
W : Weight per coordinate, considered as a real vector same size as Z
N : maximal degree for generated polynomials
'''
def DOP(Z,Z_target, W, N):
  w=np.array(W,dtype=np.double)
  z=np.array(Z) # NODES from samples
  z_target=np.array(Z_target) # NODES from target
  P =np.zeros(shape=(N,N,z.size),dtype=np.complex128)
  P_target = np.zeros(shape=(N,N,z_target.size),dtype=np.complex128) 


  ############MGS iteration##########################
  
  # loading polynomials
  for j1 in range(0,N): # column
    for k1 in range(0,N): # row
      P[k1,j1,:] = (z**k1)*(np.conjugate(z**j1))[:] # Z^k conj(Z)^j
      P_target[k1,j1,:] = (z_target**k1)*(np.conjugate(z_target**j1))[:]
      no=norm(w,P[k1,j1,:])
      P[k1,j1,:]/=no # normalizing all (***different from mgs***)
      P_target[k1,j1,:]/=no # normalizing all (***different from mgs***)
  
  # MGS 4-cycle for 2d structure
  for j1 in range(0,N): # column
    for k1 in range(0,N): # row element to be substracted from the rest
      # preparing the first element from current k1,j1
      # ensure the case when we reach the bottom
      if (k1==N-1): # at the bottom
          lowerj2=j1+1 # advance column
      else:
          lowerj2=j1 # maintain column

      no=norm(w,P[k1,j1,:])
      P[k1,j1,:]/=no  
      P_target[k1,j1,:]/=no

      # for the rest of polynomials we substract the proyection of k1,j1
      for j2 in range(lowerj2,N): #origin column
      # preparing the first element from current k1,j1
      # ensure the case when we reach the bottom
        if (k1==N-1 and j2==lowerj2 ): # at the bottom
          lowerk2=0  # reset row
        elif (j2==lowerj2): # on first column
          lowerk2=k1+1 # advance row
        else:
          lowerk2=0 # remain column reset
        for k2 in range(lowerk2,N): # origin row
          # print("updating ",k2,",",j2)
          prod=dot(w,P[k2,j2,:],P[k1,j1,:])
          P[k2,j2,:] -= prod*P[k1,j1,:]
          P_target[k2,j2,:] -= prod*P_target[k1,j1,:]
          no=norm(w,P[k2,j2,:])
          P[k2,j2,:]/=no # normalizing all (***different from mgs***)
          P_target[k2,j2,:]/=no # normalizing all (***different from mgs***)
      # for the rest of polynomials we substract the proyection of k1,j1
      # second pass
      for j2 in range(0,j1+1): #origin column
        for k2 in range(0,N): # origin row
          #print(k1,j1,k2,j2)
          if (j1!=j2 or k1!=k2):
            prod=dot(w,P[k1,j1,:],P[k2,j2,:])
            P[k1,j1,:] -= prod*P[k2,j2,:]
            P_target[k1,j1,:] -= prod*P_target[k2,j2,:]
          else:
              break
        no=norm(w,P[k1,j1,:])
        P[k1,j1,:]/=no
        P_target[k1,j1,:]/=no
        
  print(np.asnumpy(np.sum(np.sum(P,axis=2)/z.size - np.sum(P_target,axis=2)/z_target.size)))  
  return(P,P_target)


###############################################################################
###############################################################################
###############################################################################
###############################################################################



# Experiment


# parameters
s=30
N=s   # polinomial maximal order
M=s+1  # an odd number for this symmetrical case
L=1 # maximum coordinate in image
ext_factor = 1 # extrapolation factor

p = 0.02
print("parameters")
print("N: ",N)
print("M: ",M)
print("L: ",L)
print("ext_factor: ",ext_factor)


# Computing data for regression
u = np.reshape(np.linspace(-L,L,M),(M,1)) *np.ones(shape=(1,M))
v = np.reshape(np.linspace(-L,L,M),(1,M)) *np.ones(shape=(M,1))

mask = np.random.binomial(n=1,p=p,size=(M,M))

u_selected = u[mask==1]
v_selected = v[mask==1]

uv = u_selected + v_selected*1j


uv= u+1j*v

# a way to extend the domain in order to prove extrapolation is to expand resolution
u = np.reshape(np.linspace(-L,L,ext_factor*M),(ext_factor*M,1)) 
v = np.reshape(np.linspace(-L,L,ext_factor*M),(1,ext_factor*M)) 
uv_target= u+1j*v

# measurement weighting in square grid
#w =np.ones((M,M)) # for now uniform noise

w =np.ones(np.size(uv))

# DOP generation
P,P_target = DOP(uv.flatten(), uv_target.flatten(), w.flatten(), N)
print("Diferencia de suma target:", np.asnumpy(np.sum(P_target/(M*M*N*N))))
# Polynomial correlation

K=np.arange(N)
J=np.arange(N)
K,J=np.meshgrid(K,J)
#idx=K>=J
idx=K>=J # case with diagonal
idx = np.reshape(idx,(N,N,1))
idx = np.ones((N,N,M*M))*idx
idx = idx==1
pp=P_target[idx]
pp =np.reshape(pp,(int(N*(N+1)/2),M*M))
#pp =np.reshape(pp,(int(N*(N-1)/2),M*M)) # case without diagonal
corr=np.dot(pp,np.conjugate(pp.T))

title="Polynomials Correlation by pair (diagonal: equal indexes)"
fig=plt.figure(title)
plt.title(title)
im=plt.imshow(np.asnumpy(np.absolute(corr)))
plt.colorbar(im)

# we exibit orthononality errors (no diagonal) bellow diagonal
cor=corr-np.diag(np.diag(corr))
title="Polynomials Correlation below diagonal (k<=j)"
fig=plt.figure(title)
plt.title(title)
im=plt.imshow(np.asnumpy(np.absolute(cor)))
plt.colorbar(im)

# we exhibit orthogonality for whole matrix 
pp1=np.reshape(P_target,(N*N,M*M))
corr1=np.dot(pp1,np.conjugate(pp1.T))
cor1=corr1-np.diag(np.diag(corr1))
title="Polynomials Correlation full matrix with no diagonal"
fig=plt.figure(title)
plt.title(title)
im=plt.imshow(np.asnumpy(np.absolute(cor1)))
plt.colorbar(im)

# we exhibit orthogonality for whole matrix 


title="Polynomials Correlation half matrix below main counter diagonal"

K=np.arange(N)
one = np.ones(N)
K=np.reshape(K,(N,1))
one=np.reshape(one,(1,N))
K=K*one
J=np.arange(N)
J=np.reshape(J,(1,N))
one=np.reshape(one,(N,1))
J=J*one
Idx = K+J<=N-1

one=np.ones(shape=(N,N,M*M))
Idx=np.reshape(Idx,(N,N,1))
Idx=Idx*one
Idx=Idx==1
pp2=P_target[Idx]
pp2 = np.reshape(pp2,(int(N*(N+1)/2),M*M))
corr2=np.dot(pp2,np.conjugate(pp2.T))
cor2=corr2-np.diag(np.diag(corr2))

fig=plt.figure(title)
plt.title(title)
im=plt.imshow(np.asnumpy(np.absolute(cor2)))
plt.colorbar(im)


# we exibit lack of symmetry kj <-> jk
pp2 = np.reshape(P_target,(N,N,M*M))
dpp = np.absolute(pp2 - np.transpose(pp2,axes=(1,0,2))) 
dpp = np.reshape(dpp,(N*N,M*M))
ndpp= np.dot(dpp,np.conjugate(dpp.T))
title="Correlation of differences kj - jk"
fig=plt.figure(title)
plt.title(title)
im=plt.imshow(np.asnumpy(np.absolute(ndpp)))
plt.colorbar(im)

# looking for different polinomials shape
#P=np.reshape(P,(N,N,M,M))
P_target=np.reshape(P_target,(N,N,ext_factor*M,ext_factor*M))

title="Absolute value of P_2,2"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(P_target[2,2,:,:]))); plt.colorbar(im)

title="Absolute value of P_6,3"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(P_target[6,3,:,:]))); plt.colorbar(im) # this exhibit symmetry diferences
title="Absolute value of P_3,6"; fig=plt.figure(title); plt.title(title);  im=plt.imshow(np.asnumpy(np.absolute(P_target[3,6,:,:]))); plt.colorbar(im)

title="Absolute value of extrapolated P_6,3"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(P_target[6,3,:,:]))); plt.colorbar(im) # this exhibit symmetry diferences
title="Absolute value of extrapolated P_3,6"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(P_target[3,6,:,:]))); plt.colorbar(im)

title="Real part of P_25,18"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.real(P_target[25,18,:,:]))); plt.colorbar(im)
title="Real part of P_18,25"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.real(P_target[18,25,:,:]))); plt.colorbar(im)
title="Absolute value of extrapolated P_18,25"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(P_target[18,25,:,:]))); plt.colorbar(im)

plt.show()

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from math import e, pi
from array import *

def dot(weights,x,y):
  return(np.sum(x*weights*np.conjugate(y)))
  
def dot2x2(weights,x,y):
  mul = x*weights*np.conjugate(y)
  f,c,d = mul.shape
  npsum = np.array(list(map(np.sum,list(np.reshape(mul,(f*c,d))))))
  npsum = np.reshape(npsum,(f*c,1))
  aux = np.ones(shape=(f*c,d),dtype=float)
  npsum = aux*npsum
  npsum = np.reshape(npsum,(f,c,d))
  return npsum

def norm(weights,x):
  return(np.sqrt(np.sum(weights*np.absolute(x)**2)))

def form1(P,n):
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = P[k,j,:]/norm(1,P[k,j,:])
            h = 0
            for x in range(j,n):
                if (x==j):
                    h=k+1
                else:
                    h=0
                for y in range(h,n):
                    P[y,x,:] = P[y,x,:] - dot(1,P[k,j,:],P[y,x,:])*P[k,j,:]
            print(P)
            print("")
    return P
    
def form2(P,n):
    M = np.zeros(shape=(n,n,n),dtype=np.complex128)
    for j in range(0,n):
        for k in range(0,n):
                P[k,j,:] = P[k,j,:]/norm(1,P[k,j,:])
                m = np.array(P[k,j,:])
                M[k,j,:] = np.array(m)
                P = P - dot2x2(1,m,P)*m
                print(M)
                #print(M)
                print("")
    return M
P =np.ones(shape=(3,3,3),dtype=np.complex128)
P = P + 1j

P1= form1(P,3)
print("hola")
print("hola")
P =np.ones(shape=(3,3,3),dtype=np.complex128)
P = P + 1j
P2= form2(P,3)

#print(form1(P,3))
#print(form2(P,3))
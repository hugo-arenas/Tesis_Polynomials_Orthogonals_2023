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
  
def recurrence2d(z,w,n,dim,i):
    P = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    V = np.ones(shape=(n,n,1),dtype=int)
    Ig = np.zeros(shape=(dim,dim),dtype=np.complex128)
    Ig_f = np.zeros(shape=(dim,dim),dtype=np.complex128)
    iaux = np.array(i)
    std_a = np.zeros(1,dtype=np.complex128)
    
    value = 100000
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
            M = dot(w,P[k-j+l,j,:],iaux.flatten())
            Psub = np.reshape(P,(n,n,dim,dim))
            Ig = Ig + M*Psub[k-j+l,j,:,:]
            if j==0 and k == 0:
                std = np.std(i)
                std_a[0] = std
            else:
                iaux = iaux - M*Psub[k-j+l,j,:,:]
                std = np.std(iaux)
                std_a = np.concatenate((std_a,np.array([std])),axis=0)
                if std <=value:
                    value = std
                    Ig_f = np.array(Ig)
    
    return P, Ig_f, std_a, std

def disk(low,high,theta,e,a,dim):
  # Domain of measurements
  x=np.linspace(start=low,stop=high,num=dim)
  y=np.linspace(start=low,stop=high,num=dim)

  sigma=(high-low)/2/(3*3) # 3 sigma on 1/3 of the canvas

  x,y=np.meshgrid(x,y)
  b=a*np.sqrt(1-e**2)
  # rotating and distorting the circle
  x = x*np.cos(theta)/a + y*np.sin(theta)/b
  y = x*np.sin(theta)/a - y*np.cos(theta)/b
  # the radious of this distortion
  r = np.sqrt(x**2 + y**2)
  img = (1.0+np.sin(r/(0.01*sigma*2*np.pi)))*np.exp(-r*r/(2*sigma**2))/2.0
  return(img)

def gauss(ini,dim):
    array_x = np.linspace(-ini,ini,dim)
    array_x = np.reshape(array_x,(dim,1))
    array_y = np.reshape(array_x,(1,dim))
    img = np.exp(-pi*(array_x**2 + array_y**2))
    return(img)
    
N = 101#size image
S = 50
#S = 8#polynomial order
#print("Size image N: ",N, " and polynomial order S: ",S)


print("Size image N: ",N)

ini = 1

p = 1

#factor = 3

img = disk(low=-ini,high=ini,theta=0.4, e=0.8,a=1, dim=N)

noise = np.random.rand(N,N)

#hollows = np.zeros(shape=(N,N),dtpye=float)
#create hollows

img1 = img*noise

#fftimg1 = np.fft.fft2(img1)#*pi/N
#fftimg1 = np.fft.fftshift(fftimg1)

fig = plt.figure("image (without noise) vs image (with noise)")
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

im1=ax1.matshow(np.asnumpy(img))

im2=ax2.matshow(np.asnumpy(np.absolute(img1)))

stda_original = np.std(img.flatten())

du = np.linspace(ini,-ini,N)
u,v = np.meshgrid(du,du)
z = u + 1j*v

#u = np.reshape(np.linspace(ini,-ini,N),(N,1)) 
#v = np.reshape(np.linspace(ini,-ini,N),(1,N)) 
#z= u+1j*v

w = np.ones((N,N))

start_time = time.time()

P, Ig, std_a, std_m = recurrence2d(z.flatten(), w.flatten(), S, N, img1)

print(time.time() - start_time)

print("STD original: ",np.asnumpy(stda_original))
print("STD mínimo: ",np.asnumpy(std_m))
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

residual = Ig - img

#title="Absolute value of P_2,2"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(P[2,2,:,:]))); plt.colorbar(im)

#title="Absolute value of P_6,3"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(P[6,3,:,:]))); plt.colorbar(im)
#title="Absolute value of P_3,6"; fig=plt.figure(title); plt.title(title);  im=plt.imshow(np.asnumpy(np.absolute(P[3,6,:,:]))); plt.colorbar(im)

#title="Real part of P_25,18"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.real(P[14,10,:,:]))); plt.colorbar(im)
#title="Real part of P_18,25"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.real(P[10,14,:,:]))); plt.colorbar(im)
title="Model"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(Ig)))

plt.savefig("modelo.png")
#title="Result"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(I)))
title="Residual"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(residual)))
title="Desviation Standar"; fig=plt.figure(title); plt.title(title); plt.plot(np.asnumpy(std_a))

plt.show()


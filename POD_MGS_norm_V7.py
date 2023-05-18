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
  
def recurrence2d(z,z_target,w, w_target, data, size):
    min_n = 5
    max_n = 21
    idx_max = 0
    value_min = 10.0
    s_order = 0
    stda_final = np.zeros(1,dtype=float)
    P_final = np.zeros(1,dtype=np.complex128)
    Ig_final = np.zeros(1,dtype=np.complex128)
    for s in range(min_n,max_n):
        idx = 0
        value = 10.0
        P = np.zeros(shape=(s,s,z.size),dtype=np.complex128)
        P_target = np.zeros(shape=(s,s,z_target.size),dtype=np.complex128)
        
        A = np.zeros(shape=(s,s,z.size),dtype=np.complex128)
        A_target = np.zeros(shape=(s,s,z_target.size),dtype=np.complex128)
        
        B = np.zeros(shape=(s,s,z.size),dtype=np.complex128)
        B_target = np.zeros(shape=(s,s,z_target.size),dtype=np.complex128)
        
        M = 0.0 + 0.0j
        n = np.size(data)
        
        Ig = np.zeros(shape=(size,size),dtype=np.complex128)
        
        Ig_aux = np.zeros(shape=(size,size),dtype=np.complex128)
        
        std_a = np.zeros(1,dtype=np.complex128)
        
        dataux = np.array(data)
        
        D = np.zeros(z.size,dtype=np.complex128)
        D_target = np.zeros(z_target.size,dtype=np.complex128)
        
        kaux=0
        l=0
        for j in range(0,s):
            for k in range(0,s):
                P[k,j,:] = (z**k)*np.conjugate(z**j)
                P_target[k,j,:] = (z_target**k)*np.conjugate(z_target**j)
                
                P[k,j,:] = P[k,j,:]/norm(w,P[k,j,:])
                P_target[k,j,:] = P_target[k,j,:]/norm(w,P[k,j,:])
        
        large1 = np.array(range(0,s))
        large2 = np.ones(s-1)*(s-1)
        large = np.concatenate((large1,large2),axis=0)
        for k in large:
            if kaux!=s-1:
                l=0
            else:
                l=l+1
            k = int(k)
            kaux = k
            for j in range(0,k+1-l):
                P[k-j,l+j,:] = P[k-j,l+j,:]/norm(w,P[k-j,l+j,:])
                P_target[k-j,l+j,:] = P_target[k-j,l+j,:]/norm(w,P[k-j,l+j,:])

                A[k-j,l+j,:] = np.array(P[k-j,l+j,:])
                A_target[k-j,l+j,:] = np.array(P_target[k-j,l+j,:])
                
                P = P - dot2x2(w,P[k-j,l+j,:],P,1)*P[k-j,l+j,:]
                P_target = P_target - dot2x2(w_target,P_target[k-j,l+j,:],P_target,1)*P_target[k-j,l+j,:]
                
                P = P/norm2x2(w,P)
                P_target = P_target/norm2x2(w_target,P_target)

        kaux=0
        l=0
        for k in large:
            if kaux!=s-1:
                l=0
            else:
                l=l+1
            k = int(k)
            kaux = k
            for j in range(0,k+1-l):
                if k==0 and j==0:
                    A[k-j,l+j,:] = A[k-j,l+j,:]/norm(w,A[k-j,l+j,:])
                    A_target[k-j,l+j,:] = A_target[k-j,l+j,:]/norm(w,A[k-j,l+j,:])
                    
                    D=np.array(A[k-j,l+j,:])
                    D_target=np.array(A_target[k-j,l+j,:])
                    
                    B[k-j,l+j,:] = np.array(A[k-j,l+j,:])
                    B_target[k-j,l+j,:] = np.array(A_target[k-j,l+j,:])
                else:
                    if j==1+l and k>0:
                        A=A/norm2x2(w,A)
                        A_target=A_target/norm2x2(w_target,A_target)
                
                    A = A - dot2x2(w,D,A,0)*D
                    A_target = A_target - dot2x2(w_target,D_target,A_target,0)*D_target
                    
                    A[k-j,l+j,:] =  A[k-j,l+j,:]/norm(w,A[k-j,l+j,:])
                    A_target[k-j,l+j,:] =  A_target[k-j,l+j,:]/norm(w,A[k-j,l+j,:])
                    
                    if (j==l):
                        A[k-j,l+j,:] = A[k-j,l+j,:]/norm(w,A[k-j,l+j,:])
                        A_target[k-j,l+j,:] =  A_target[k-j,l+j,:]/norm(w,A[k-j,l+j,:])
                    D=np.array(A[k-j,l+j,:])
                    D_target=np.array(A_target[k-j,l+j,:])
                    
                    B[k-j,l+j,:] = np.array(A[k-j,l+j,:])
                    B_target[k-j,l+j,:] = np.array(A_target[k-j,l+j,:])
                    
                M = dot(w,B[k-j,l+j,:],dataux.flatten())
                print(M)
                Bsub = np.reshape(B_target,(s, s, size, size))
                Ig = Ig + M*Bsub[k-j,l+j,:,:]
                if j==0 and k == 0:
                    std = np.std(data)
                    std_a[0] = std
                else:
                    dataux = dataux - M*B[k-j,l+j,:]
                    std = np.std(dataux)
                    std_a = np.concatenate((std_a,np.array([std])),axis=0)
                    if std <= value:
                        value = std
                        pos = np.where(std_a == value)
                        idx = max(pos[0])
                        Ig_aux = np.array(Ig)
        if value <= value_min:
            value_min = value
            idx_max = idx
            s_order = s
            P_final = np.array(B_target)
            Ig_final = np.array(Ig_aux)
            stda_final = np.array(std_a)
    return P_final, Ig_final, stda_final, s_order, idx_max

N = 101#size image
#S = 8#polynomial order
#print("Size image N: ",N, " and polynomial order S: ",S)
print("Size image N: ",N)

ini = 1

p = 0.5

#factor = 3

array_x = np.linspace(-ini,ini,N)

array_x = np.reshape(array_x,(N,1))

array_y = np.reshape(array_x,(1,N))

img1 = np.exp(-pi*(array_x**2 + array_y**2))

noise = np.random.rand(N,N)

#hollows = np.zeros(shape=(N,N),dtpye=float)
#create hollows

img = np.array(img1)

img1 = img1*noise

noise = np.random.rand(N,N)

img1 = img1*noise

mask = np.random.binomial(n=1,p=0.5,size=(N,N))

img1_corrupt = np.array(img1)
img1_corrupt[np.logical_not(mask)]=0

img2 = img1[mask==1]

#fftimg1 = np.fft.fft2(img1)#*pi/N
#fftimg1 = np.fft.fftshift(fftimg1)

fig = plt.figure("image (original) vs image (noise) vs image (corrupt)")
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

im1=ax1.matshow(np.asnumpy(img))

im2=ax2.matshow(np.asnumpy(np.absolute(img1)))

im3=ax3.matshow(np.asnumpy(np.absolute(img1_corrupt)))

#du = np.linspace(-ini,ini,N)
u0 = np.linspace(-ini,ini,N)
u = np.reshape(u0,(1,N))*np.ones(shape=(N,1))

v0 = np.linspace(-ini,ini,N)
v = np.reshape(v0,(N,1))*np.ones(shape=(1,N))

u_selected = u[mask==1]
v_selected = v[mask==1]

z = u_selected + v_selected*1j

#u,v = np.meshgrid(du,du)
#z = u + 1j*v

u = np.reshape(np.linspace(-ini,ini,N),(N,1)) 
v = np.reshape(np.linspace(-ini,ini,N),(1,N)) 
z_target = u+1j*v

#w = np.ones((N,N))
w = np.ones(np.size(z))
w_target = np.ones((N,N))

start_time = time.time()

#P, Ig, std_a = recurrence2d(z.flatten(), w.flatten(), S, img1)
P, Ig, std_a, S, idx_max = recurrence2d(z,z_target.flatten(), w.flatten(), w_target.flatten(), img2, N)

print("Orden de polinomio al cuadrado es: ", S, "y el polinomio que da menor desviación estándar es: ", idx_max)

print(time.time() - start_time)
# Polynomial correlation

K=np.arange(S)
J=np.arange(S)
K,J=np.meshgrid(K,J)
#idx=K>=J
idx=K>=J # case with diagonal
idx = np.reshape(idx,(S,S,1))
idx = np.ones((S,S,np.size(z_target)))*idx
idx = idx==1
pp=P[idx]
pp =np.reshape(pp,(int(S*(S+1)/2),np.size(z_target)))
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

#residual = Ig - img

#title="Absolute value of P_2,2"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(P[2,2,:,:]))); plt.colorbar(im)

#title="Absolute value of P_6,3"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(P[6,3,:,:]))); plt.colorbar(im)
#title="Absolute value of P_3,6"; fig=plt.figure(title); plt.title(title);  im=plt.imshow(np.asnumpy(np.absolute(P[3,6,:,:]))); plt.colorbar(im)

#title="Real part of P_25,18"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.real(P[14,10,:,:]))); plt.colorbar(im)
#title="Real part of P_18,25"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.real(P[10,14,:,:]))); plt.colorbar(im)
title="Model"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(Ig)))

plt.savefig("modelo.png")
#title="Result"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(I)))
#title="Residual"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(residual)))
title="Desviation Standar"; fig=plt.figure(title); plt.title(title); plt.plot(np.asnumpy(std_a))

plt.show()


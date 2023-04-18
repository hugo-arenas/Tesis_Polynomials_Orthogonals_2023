import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from math import e, pi
from array import *

def dot(weights,x,y):
  return(np.sum(x*weights*np.conjugate(y)))

def norm(weights,x):
  return(np.sqrt(np.sum(weights*np.absolute(x)**2)))

def recurrence2d(z,w,n):
    P =np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    P00 = np.ones_like(z)
    normP00 = norm(w,P00)
    P00 = P00/normP00
    P0_1 = np.zeros_like(z)
    P[0,0,:]=P00[:]
    
    m = np.zeros((n,n,(n+1)*(n+1)),dtype=np.complex128)
    r = np.zeros((n,n),dtype=np.complex128)
    q = np.zeros((n,n,(n+1)*(n+1)),dtype=np.complex128)
    
    for j in range(0,n):
        if (j!=0):
          l=j-1
        else:
          l=j
        for k in range(l,n-1):
        #for k in range(j,n):
          
            P00[:]  = P[k,j  ,:]
            P0_1[:] = P[k,j-1,:]
            
            m00 = (z**k)*np.conj(z)**j
            m[k,j,:] = m00
            r[k,j] = np.sum(m[k,j,:]*m[k,j,:])**(1/2)
            q[k,j,:] = m[k,j,:]/r[k,j]
            
            for y in range(0,k):
                for x in range(0,y):
                    q00 = q[y,x,:].transpose()
                    
                    #print(m[k,j,:].shape)
                    #print(q00.shape)
                    r[y,x] = np.matmul(q00,m[k,j,:])
                    print(r[y,x])
                    m[k,j,:] = m[k,j,:] - r[y,x]*q[y,x,:]
            
            #for y in range(0,k):
            #    for x in range(0,j):
            #        m1 = (z**y)*np.conj(z)**(x)
            #        q = m1/(np.sum(m1*m1)**(1/2))
            #        r = q*m
            #        print(np.sum(m1*m1))
            #        m = m - r*q
            
            #for y in range(0,k):
            #    for x in range(y,k):
            #        m1 = (z**y)*np.conj(z)**(x)
            #        q = m1/(np.sum(m1*m1)**(1/2))
            #        r = q*m
            #        print(np.sum(m1*m1))
            #        m = m - r*q
                    
            #a = dot(w,z*P00,z*P00)-np.absolute(dot(w,z*P00,P00 ))**2-np.absolute(dot(w,z*P00,P0_1))**2
            #a = np.sqrt(np.real(a))
            #c = a*dot(w,z*P00-dot(w,z*P00, P00)*P00  ,P0_1)
            #P10 = (a*z)*P00-c*P0_1
            
            #a = dot(w,m*P00,m*P00)-np.absolute(dot(w,m*P00,P00 ))**2-np.absolute(dot(w,m*P00,P0_1))**2
            #a = np.sqrt(np.real(a))
            #c = a*dot(w,m*P00-dot(w,m*P00, P00)*P00  ,P0_1)
            #P10 = (a*m)*P00-c*P0_1
            
            
            a = dot(w,m[k,j,:]*P00,m[k,j,:]*P00)-np.absolute(dot(w,m[k,j,:]*P00,P00 ))**2-np.absolute(dot(w,m[k,j,:]*P00,P0_1))**2
            a = np.sqrt(np.real(a))
            c = a*dot(w,m[k,j,:]*P00-dot(w,m[k,j,:]*P00, P00)*P00  ,P0_1)
            P10 = (a*m[k,j,:])*P00-c*P0_1

            norm_10 = norm(w,P10)
            if (norm_10!=0):
                P10 /= norm_10

            P[k+1,j,:]=P10[:]
            P[j,k+1,:]=np.conjugate(P10)[:]
    return P

np.set_printoptions(threshold=np.inf)

N = 11

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

#A = img1.flatten()
u0 = -Lu/2
#du = np.linspace(0,N-1,N)*1/(dx*N)
#du = du + u0
du = np.linspace(u0,-u0,N)
u,v = np.meshgrid(du,du)
#duzeros = np.zeros(N)
#mu = duzeros[:, np.newaxis] + du
#dv = mu.flatten()
#du = np.sort(dv)

z = u + 1j*v
z = z.flatten()
du2 =  np.linspace(u0,-u0,N)/(dx*N)
duzeros = np.zeros(N)
mu = duzeros[:, np.newaxis] + du2
dv2 = mu.flatten()
du2 = np.sort(dv2)
z2 = du2 + 1j*dv2
#print(z)
#print(z2)
    
w = np.ones((N,N))

P = recurrence2d(z.flatten(), w.flatten(), N - 1)

K=np.arange(N-1)
J=np.arange(N-1)
K,J=np.meshgrid(K,J)
#idx=K>=J
idx=K>=J # case without diagonal
idx = np.reshape(idx,(N-1,N-1,1))
idx = np.ones((N-1,N-1,N*N))*idx
idx = idx==1
#print(idx)
pp=P[idx]
pp =np.reshape(pp,(int(N*(N-1)/2),N*N))
#pp =np.reshape(pp,(int(N*(N-1)/2),M*M)) # case without diagonal
corr=np.dot(pp,np.conjugate(pp.T))

fig=plt.figure()
im=plt.imshow(np.absolute(corr))
plt.colorbar(im)
plt.show()

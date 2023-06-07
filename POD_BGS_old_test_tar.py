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

def dot2x2(weights,x1,x2,t_gm, d_size):
  f,c,d = x2.shape
  aux = np.ones(shape=(f,c,d),dtype=float)
  w = aux*weights
  if t_gm==0:
    mul = x2*w*np.conjugate(x1)
  else:
    mul = x1*w*np.conjugate(x2)
  npsum = np.sum(mul,axis=2)
  npsum = np.reshape(npsum,(f,c,1))
  aux = np.ones(shape=(f,c,d_size),dtype=float)
  npsum = aux*npsum

  return npsum

def norm2x2(weights,x, d_size):
  f,c,d = x.shape
  aux = np.ones(shape=(f,c,d),dtype=float)
  w = aux*weights
  mul = w*np.absolute(x)**2
  npsum = np.sum(mul,axis=2)
  npsum = np.sqrt(npsum)
  npsum = np.reshape(npsum,(f,c,1))
  aux = np.ones(shape=(f,c,d_size),dtype=float)
  npsum = aux*npsum
  return npsum

def old_r2d(z,z_target,w,n):
    P =np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    P_target = np.zeros(shape=(n,n,z_target.size),dtype=np.complex128)
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = (z**k)*np.conjugate(z**j)
            P_target[k,j,:] = (z_target**k)*np.conjugate(z_target**j)
            #print(np.asnumpy(np.sum(P_target[k,j,:])/z_target.size))
            no = norm(w,P[k,j,:])
            P[k,j,:] = P[k,j,:]/no
            P_target[k,j,:] = P_target[k,j,:]/no
            print(np.asnumpy(no))
    #print(np.asnumpy(np.sum(P_target[0,0,:])/z.size))
    for j in range(0,n):
        for k in range(0,n):
            if (k==n-1):
                j2 = j + 1
            else:
                j2 = j          
            no = norm(w,P[k,j,:])
            print("pos (k,j)=",k,j," and valor norm = ",np.asnumpy(no))
            P[k,j,:] = P[k,j,:]/no
            P_target[k,j,:] = P_target[k,j,:]/no
            
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
                    P_target[y,x,:] -= prod*P_target[k,j,:]
                    no=norm(w,P[y,x,:])
                    P[y,x,:]/=no
                    P_target[y,x,:] = P_target[y,x,:]/no
            no = norm(w,P[k,j,:])
            P[k,j,:] = P[k,j,:]/no    
            P_target[k,j,:] = P_target[k,j,:]/no
            for x in range(0,j+1):
                for y in range(0,n):
                    if (j!=x or k!=y):
                        prod=dot(w,P[k,j,:],P[y,x,:])
                        P[k,j,:] -= prod*P[y,x,:]
                        P_target[k,j,:] -= prod*P_target[y,x,:]
                    else:
                        break
                no=norm(w,P[k,j,:])
                P[k,j,:]/=no
                P_target[k,j,:] = P_target[k,j,:]/no
            #print("pos (k,j)=",k,j," and valor mod = ",np.asnumpy(np.sum(P_target[k,j,:])/z.size))
    print(np.asnumpy(np.sum(np.sum(P,axis=2)/z.size - np.sum(P_target,axis=2)/z_target.size)))  
    return P,P_target

def new_r2d(z,z_target,w,n):
    P = np.zeros(shape=(n,n,z.size),dtype=np.complex128)
    P_target = np.zeros(shape=(n,n,z_target.size),dtype=np.complex128)
    V = np.ones(shape=(n,n,1),dtype=int)
    for j in range(0,n):
        for k in range(0,n):
            P[k,j,:] = (z**k)*np.conjugate(z**j)
            P_target[k,j,:] = (z_target**k)*np.conjugate(z_target**j)
            no = norm(w,P[k,j,:])
            P[k,j,:] = P[k,j,:]/no
            P_target[k,j,:] = P_target[k,j,:]/no
           
    for j in range(0,n):
        for k in range(0,n):
            no=norm(w,P[k,j,:])
            P[k,j,:] = P[k,j,:]/no
            P_target[k,j,:] = P_target[k,j,:]/no            
            V[k,j,:] = 0
          
            dot_data = dot2x2(w,P[k,j,:],P*V,1,z.size)   
            dot_target = dot2x2(w,P[k,j,:],P*V,1,z_target.size)   
            P = P - dot_data*P[k,j,:]         
            P_target = P_target - dot_target*P_target[k,j,:] 
            no_data = norm2x2(w,P,z.size)   
            no_target = norm2x2(w,P,z_target.size)            
            no_data[(V*np.ones(z.size,dtype=int)) == 0] = 1 
            no_target[(V*np.ones(z_target.size,dtype=int)) == 0] = 1           
            P = P/no_data
            P_target = P_target/no_target
 
    V = np.ones(shape=(n,n,1),dtype=int)
    D = np.ones(1,dtype=np.complex128)
    D_target = np.ones(1,dtype=np.complex128)
    no_data = norm2x2(w,P,z.size)   
    no_target = norm2x2(w,P,z_target.size)
    P = P/no_data
    P_target = P_target/no_target
    for j in range(0,n):
        for k in range(0,n):
            if k==0 and j==0:
                no=norm(w,P[k,j,:])
                P[k,j,:] = P[k,j,:]/no
                P_target[k,j,:] = P_target[k,j,:]/no
                V[k,j,:] = 0
                D = np.array(P[k,j,:])
                D_target = np.array(P_target[k,j,:])
            else:
                if (k==1 and j>0):
                    no_data = norm2x2(w,P,z.size)   
                    no_target = norm2x2(w,P,z_target.size)            
                    no_data[(V*np.ones(z.size,dtype=int)) == 0] = 1 
                    no_target[(V*np.ones(z_target.size,dtype=int)) == 0] = 1         
                    P = P/no_data
                    P_target = P_target/no_target
                    
                dot_data = dot2x2(w,D,P*V,0,z.size)   
                dot_target = dot2x2(w,D,P*V,0,z_target.size)   
                P = P - dot_data*D         
                P_target = P_target - dot_target*D_target 
                no=norm(w,P[k,j,:])
                P[k,j,:] = P[k,j,:]/no
                P_target[k,j,:] = P_target[k,j,:]/no
                if (k==0):
                    no=norm(w,P[k,j,:])
                    P[k,j,:] = P[k,j,:]/no
                    P_target[k,j,:] = P_target[k,j,:]/no
                V[k,j,:] = 0
                D = np.array(P[k,j,:])
                D_target = np.array(P_target[k,j,:])
    print(np.asnumpy(np.sum(np.sum(P,axis=2)/z.size - np.sum(P_target,axis=2)/z_target.size)))  
    return P,P_target
#np.set_printoptions(threshold=np.inf)

N = 31

S = 30

print("Size image N: ",N)

ini = 1

p = 0.02

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

mask = np.random.binomial(n=1,p=p,size=(N,N))

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
u = np.reshape(u0,(N,1))*np.ones(shape=(1,N))

v0 = np.linspace(-ini,ini,N)
v = np.reshape(v0,(1,N))*np.ones(shape=(N,1))

u_selected = u[mask==1]
v_selected = v[mask==1]

z = u_selected + v_selected*1j

u,v = np.meshgrid(u0,v0)
#u = np.reshape(np.linspace(-ini,ini,N),(N,1)) 
#v = np.reshape(np.linspace(-ini,ini,N),(1,N)) 
z_target = u+1j*v

w = np.ones(np.size(z))

start_time = time.time()

P1,P1_tar = old_r2d(z.flatten(),z_target.flatten(), w.flatten(), S)

print("Tiempo de ejecución orden N^4:", time.time() - start_time)
start_time = time.time()

P2,P2_tar = new_r2d(z.flatten(),z_target.flatten(), w.flatten(), S)

print("Tiempo de ejecución orden N^2 (nuevo):", time.time() - start_time)

idx = P1==P2
#print("Diferencia parcial:", np.asnumpy(np.sum((P1[:,3,:]-P2[:,3,:]),axis=1)))
print("Diferencia de resultados (target):", np.asnumpy(np.sum(P1_tar-P2_tar)))
print("Diferencia de resultados:", np.asnumpy(np.sum(P1-P2)))
print("Diferencia de suma target (N4 y N2):", np.asnumpy(np.sum(P1_tar/(S*S*N*N))), np.asnumpy(np.sum(P2_tar/(S*S*N*N))))
#print("Diferencia de resultados:", idx)

plt.show()
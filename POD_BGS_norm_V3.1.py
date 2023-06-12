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

def  mse(img1, img2):
    err = np.sum((img1 - img2)**2)
    err = err / (img1.shape[0]*img1.shape[1])
    return err
  
def recurrence2d(z,z_target,w, data, size, limit):
    z = np.array(z)
    z_target = np.array(z_target)
    w = np.array(w)
    min_n = 30
    max_n = 31
    idx_max = 0
    value_min = 1000000.0
    s_order = 0
    stda_final = np.zeros(1,dtype=float)
    P_final = np.zeros(1,dtype=np.complex128)
    Ig_final = np.zeros(1,dtype=np.complex128)
    for s in range(min_n,max_n):
        count = 0
        print(s)
        idx = 0
        value = 10000.0
        P = np.zeros(shape=(s,s,z.size),dtype=np.complex128)
        P_target = np.zeros(shape=(s,s,z_target.size),dtype=np.complex128)      
        V = np.ones(shape=(s,s,1),dtype=int)       
        M = 0.0 + 0.0j       
        Ig = np.zeros(shape=(size,size),dtype=np.complex128)       
        Ig_aux = np.zeros(shape=(size,size),dtype=np.complex128)      
        std_a = np.zeros(1,dtype=np.complex128)     
        dataux = np.array(data)       
        k2=0
        l=0
        for j in range(0,s):
            for k in range(0,s):
                P[k,j,:] = (z**k)*(np.conjugate(z**j))[:]
                P_target[k,j,:] = (z_target**k)*(np.conjugate(z_target**j))[:]
                
                no=norm(w,P[k,j,:])
                P[k,j,:] = P[k,j,:]/no
                P_target[k,j,:] = P_target[k,j,:]/no
        
        large1 = np.array(range(0,s))
        large2 = np.ones(s-1)*(s-1)
        large = np.concatenate((large1,large2),axis=0)
        for k in large:
            if k2!=s-1:
                l=0
            else:
                l=l+1
            k = int(k)
            k2 = k
            for j in range(l,k+1):
                no=norm(w,P[k-j+l,j,:])
                P[k-j+l,j,:] = P[k-j+l,j,:]/no
                P_target[k-j+l,j,:] = P_target[k-j+l,j,:]/no
                
                V[k-j+l,j,:] = 0
                
                dot_data = dot2x2(w,P[k-j+l,j,:],P*V,1,z.size)
                dot_target = dot2x2(w,P[k-j+l,j,:],P*V,1,z_target.size)
                
                P = P - dot_data*P[k-j+l,j,:]
                P_target = P_target - dot_target*P_target[k-j+l,j,:]
                
                no_data = norm2x2(w,P,z.size)
                no_target = norm2x2(w,P,z_target.size)
                
                no_data[(V*np.ones(z.size,dtype=int)) == 0] = 1 
                no_target[(V*np.ones(z_target.size,dtype=int)) == 0] = 1
                
                P = P/no_data
                P_target = P_target/no_target
        
        V = np.ones(shape=(s,s,1),dtype=int)
        k2=0
        l=0
        D = np.zeros(1,dtype=np.complex128)
        D_target = np.zeros(1,dtype=np.complex128)
        
        no_data = norm2x2(w,P,z.size)
        no_target = norm2x2(w,P,z_target.size)
        P = P/no_data
        P_target = P_target/no_target
        
        for k in large:
            if k2!=s-1:
                l=0
            else:
                l=l+1
            k = int(k)
            k2 = k
            for j in range(l,k+1):
                if k==0 and j==0:
                    no=norm(w,P[k-j+l,j,:])
                    P[k-j+l,j,:] = P[k-j+l,j,:]/no
                    P_target[k-j+l,j,:] = P_target[k-j+l,j,:]/no
                    D = np.array(P[k-j+l,j,:])
                    D_target = np.array(P_target[k-j+l,j,:])
                    V[k-j+l,j,:] = 0

                else:
                    if j==1+l and k>=0:
                        no_data = norm2x2(w,P,z.size)
                        no_target = norm2x2(w,P,z_target.size)

                        no_data[(V*np.ones(z.size,dtype=int)) == 0] = 1 
                        no_target[(V*np.ones(z_target.size,dtype=int)) == 0] = 1

                        P=P/no_data
                        P_target=P_target/no_target
                    
                    dot_data = dot2x2(w,D,P*V,0,z.size)
                    dot_target = dot2x2(w,D,P*V,0,z_target.size)
                    
                    P = P - dot_data*D
                    P_target = P_target - dot_target*D_target
                    
                    no=norm(w,P[k-j+l,j,:])
                    P[k-j+l,j,:] =  P[k-j+l,j,:]/no
                    P_target[k-j+l,j,:] =  P_target[k-j+l,j,:]/no
                    
                    if (j==l):
                        no=norm(w,P[k-j+l,j,:])
                        P[k-j+l,j,:] = P[k-j+l,j,:]/no
                        P_target[k-j+l,j,:] =  P_target[k-j+l,j,:]/no

                    V[k-j+l,j,:] = 0
                    D = np.array(P[k-j+l,j,:])
                    D_target = np.array(P_target[k-j+l,j,:])
                       
                M = dot(w,P[k-j+l,j,:],dataux.flatten())
                Psub = np.reshape(P_target,(s, s, size, size))
                Ig = Ig + M*Psub[k-j+l,j,:,:]
                
                count = count + 1
                if j==0 and k == 0:
                    std = np.std(data)
                    std_a = np.array([std])
                else:
                    dataux = dataux - M*P[k-j+l,j,:]
                    std = np.std(dataux)
                    std_a = np.concatenate((std_a,np.array([std])),axis=0)
                    if (limit == 0 and std <= value):
                        value = std
                        pos = np.where(std_a == std)
                        idx = max(pos[0])
                        Ig_aux = np.array(Ig)
                    elif (count == limit):
                        value = std
                        pos = np.where(std_a == std)
                        idx = max(pos[0])
                        Ig_aux = np.array(Ig)
        if value <= value_min:
            value_min = value
            idx_max = idx
            s_order = s
            P_final = np.array(P_target)
            Ig_final = np.array(Ig_aux)
            stda_final = np.array(std_a)
    return P_final, Ig_final, stda_final, s_order, idx_max

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
    
def star4(ini,dim):
    array_x = np.linspace(-ini,ini,dim)
    array_y = np.linspace(-ini,ini,dim)
    array_x = np.reshape(array_x,(dim,1))
    array_y = np.reshape(array_y,(1,dim))
    img = (1/(np.absolute(array_x)+2)) + (1/(np.absolute(array_y)+2))
    return(img)
    
N = 300#size image
#S = 8#polynomial order
#print("Size image N: ",N, " and polynomial order S: ",S)

print("Size image N: ",N)

ini = 1

p = 1.


#img = disk(low=-ini,high=ini,theta=0.4, e=0.8,a=1, dim=N)

img = star4(ini,N)

noise = np.random.rand(N,N)


img1 = np.array(img)*noise

noise = np.random.rand(N,N)

img1 = img1*noise

mask = np.random.binomial(n=1,p=p,size=(N,N))

img1_corrupt = np.array(img1)
img1_corrupt[np.logical_not(mask)]=0

img2 = img[mask==1]

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

#min_uv = np.min(np.array([np.min(u_selected),np.min(v_selected)]))

#max_uv = np.max(np.array([np.max(u_selected),np.max(v_selected)]))

#total_uv = max_uv - min_uv

min_uv = -1.0

max_uv = 1.0

total_uv = 2

z = u_selected + v_selected*1j

 
#u,v = np.meshgrid(u0,v0)
z_target = u+1j*v


w = np.ones(np.size(z))


start_time = time.time()

dim_z = 101
dif = int(N/dim_z)
img_f = np.zeros(shape=(N,N),dtype=np.complex128)

if dif - (N/dim_z) < 0:
    dif = dif + 1

sep = total_uv/dif

idx_max = 0

limit = 0

for num in range(0,2):

    for r in range(0,dif):
        min_y, max_y, min_x, max_x = 0,0,0,0
        min_v, max_v, min_u, max_u = 0,0,0,0
        print(r)
        for c in range(0,dif):
            min_y, max_y = dim_z*r, dim_z*(r+1)
            min_x, max_x = dim_z*c, dim_z*(c+1)
            min_v, max_v = min_uv + sep*r, min_uv + sep*(r+1)
            min_u, max_u = min_uv + sep*c, min_uv + sep*(c+1)

            if r == dif-1:
                min_y, max_y = N-dim_z, N
            if c == dif-1:
                min_x, max_x = N-dim_z, N
            
            #u0 = np.linspace(-ini,ini,dim_z)
            #u = np.reshape(u0,(1,dim_z))*np.ones(shape=(dim_z,1))
            #v = np.reshape(u0,(dim_z,1))*np.ones(shape=(1,dim_z))
            
            #z_target_point = u+1j*v
            
            z_target_point = z_target[min_y:max_y,min_x:max_x]
            z_point = z[np.where((z.real>=min_u)&(z.real<=max_u)&(z.imag>=min_v)&(z.imag<=max_v))]
            
            index = np.where((z.real>=min_u)&(z.real<=max_u)&(z.imag>=min_v)&(z.imag<=max_v))
            img2_point = img2[index]
            
            w_point = w[index]
            
            #title="Model("+str(r)+","+str(c)+")"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(np.reshape(img2_point,(dim_z,dim_z)))))
            
            print(img2_point.size,w_point.size,z_point.size)
            
            if num == 0:
                P, Ig, std_a, S, idx_max = recurrence2d(z_point.flatten(),z_target_point.flatten(), w_point.flatten(), img2_point, dim_z, 0)
                if int(idx_max) >= limit:
                    limit = int(idx_max)
            else:
                P, Ig, std_a, S, idx_max = recurrence2d(z_point.flatten(),z_target_point.flatten(), w_point.flatten(), img2_point, dim_z, limit)
                print("Orden de polinomio al cuadrado es: ", S, "y el polinomio que da menor desviación estándar es: ", idx_max)
                img_f[min_y:max_y,min_x:max_x] = Ig
                title="Model("+str(r)+","+str(c)+")"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(img_f[min_y:max_y,min_x:max_x])))
            #P, Ig, std_a, S, idx_max = recurrence2d(z.flatten(),z_target_point.flatten(), w.flatten(), img2, dim_z, img)
            #img_f[min_y:max_y,min_x:max_x] = Ig


print(time.time() - start_time)
# Polynomial correlation
'''
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
title="Model"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(img_f)))

plt.savefig("modelo.png")
#title="Result"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(I)))
#title="Residual"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(residual)))
#title="Desviation Standar"; fig=plt.figure(title); plt.title(title); plt.plot(np.asnumpy(std_a))

plt.show()


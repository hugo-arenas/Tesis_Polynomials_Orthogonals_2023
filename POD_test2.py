#import numpy as np
import cupy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from math import e, pi
from array import *
import time

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

N = 200#size image
#S = 8#polynomial order
#print("Size image N: ",N, " and polynomial order S: ",S)

print("Size image N: ",N)

ini = 1

p = 1.


img = disk(low=-ini,high=ini,theta=0.4, e=0.8,a=1, dim=N)


title="Image"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(img)))

du = np.linspace(ini,-ini,N)
u,v = np.meshgrid(du,du)

z = u + 1j*v

total_uv = 100

w = np.ones(np.size(z))



dim_z = 100
dif = int(N/dim_z)
img_f = np.zeros(shape=(N,N),dtype=np.complex128)

if dif - (N/dim_z) < 0:
    dif = dif + 1

sep = total_uv/dif


for r in range(0,dif):
    min_y, max_y, min_x, max_x = 0,0,0,0
    print(r)
    for c in range(0,dif):
        min_y, max_y = dim_z*r, dim_z*(r+1)
        min_x, max_x = dim_z*c, dim_z*(c+1)

        if r == dif-1:
            min_y, max_y = N-dim_z, N
        if c == dif-1:
            min_x, max_x = N-dim_z, N
        
        z_point = z[min_y:max_y,min_x:max_x]
        title="Z("+str(r)+","+str(c)+")"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(z_point)))
        img_point = img[min_y:max_y,min_x:max_x]
        title="Model("+str(r)+","+str(c)+")"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.asnumpy(np.absolute(img_point)))
        #z_target_point = z_target[min_y:max_y,min_x:max_x]
        #z_point = z[np.where((z.real>=min_u)&(z.real<=max_u)&(z.imag>=min_v)&(z.imag<=max_v))]
        
        #index = np.where((z.real>=min_u)&(z.real<=max_u)&(z.imag>=min_v)&(z.imag<=max_v))
        
        #img2_point = img2[index]
        #w_point = w[index]
       

plt.show()
import cupy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
from matplotlib import cm
from math import e, pi
from array import *
import time

A = np.random.rand(100)

B = np.random.rand(100)

minim = np.min(np.array([np.min(B),np.min(A)]))

p_A = np.around(A/minim)
p_B = np.around(B/minim)

minim = np.min(np.array([np.min(p_B),np.min(p_A)]))

if int(np.min(p_A))<0 or int(np.min(p_B))<0:
    p_A = p_A + minim
    p_B = p_B + minim

mask = np.random.binomial(n=1,p=0.1,size=(100))

An = A[mask == 1]
Bn = B[mask == 1]

p_An = p_A[mask == 1]

print(p_A)
p_Bn = p_B[mask == 1]

print(p_B)

min_An = np.min(p_An)
min_Bn = np.min(p_Bn)
min_C = np.min(np.array([min_An,min_Bn]))

max_An = np.max(p_An)
max_Bn = np.max(p_Bn)
max_C = np.max(np.array([max_An,max_Bn]))

C = np.zeros((int(max_C)+1,int(max_C)+1),dtype = float)

C[p_Bn.astype(int),p_An.astype(int)] = 1


#vor = Voronoi(np.asnumpy(C), furthest_site=True, incremental=True)
#print(vor.regions)

plt.figure()
plt.plot(np.asnumpy(An),np.asnumpy(Bn),'bo')

plt.figure()
plt.imshow(np.asnumpy(C))
#plt.show()
#fig = voronoi_plot_2d(vor)
plt.show()
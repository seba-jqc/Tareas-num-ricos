#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr
import scipy as sp
import matplotlib.pyplot as plt

def rand_points(seed,n):
    npr.seed(seed)
    rand1=npr.random_sample(n)
    rand2=npr.random_sample(n)
    rand3=npr.random_sample(n)
    return rand1, rand2, rand3

'''
El area a utilizar para el metodo de montecarlo corresponde al cilindro centrado
en x=2, de radio 1 y que va desde y=0 hasta y=4. Con esto se tiene que:
x: [1,3]
y: [-4,4]
z: [-1,1]
Estos intervalos se utilizaran para hacer de los valores de x, y, z uniformemente
distribuidos, de la forma x_i -> U(a,b)
'''

def montecarlo(vol,seed,n):
    x, y, z = rand_points(seed,n)
    X=2*x + 1 #distribucion U(1,3)
    Y=8*y - 4#distribucion U(-4,4)
    Z=2*z -1 #distribucion(-1.1)
    sum_x=sum_y=sum_z=sum_m=0
    for i in range(n):
        if (Z[i]**2+(np.sqrt(X[i]**2+Y[i]**2)-3)**2) <= 1:
            den = 0.5*(X[i]**2+Y[i]**2+Z[i]**2)
            sum_m += den
            sum_x += den*X[i]
            sum_y += den*Y[i]
            sum_z += den*Z[i]
    masa=vol*sum_m/n
    pos_x=vol*sum_x/(n*masa)
    pos_y=vol*sum_y/(n*masa)
    pos_z=vol*sum_z/(n*masa)
    return masa, pos_x, pos_y, pos_z

vec_m=np.array([])
vec_x=np.array([])
vec_y=np.array([])
vec_z=np.array([])

for i in range(100):
    volumen=np.pi*8
    m,x,y,z = montecarlo(volumen, i, 10**5)
    vec_m=np.append(vec_m,m)
    vec_x=np.append(vec_x,x)
    vec_y=np.append(vec_y,y)
    vec_z=np.append(vec_z,z)

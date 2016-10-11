#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Pregunta 1
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq


datos = np.loadtxt('correlacion.dat')

MPhD = datos[:, 1]
Uranio = datos[:, 2]

def polinomio(params, x):
    a,b = params
    return a*x+b

#min_U=np.amin(Uranio)
#max_U=np.amax(Uranio)
#real_params = np.polyfit(Uranio,MPhD,1)
#x_prueba=np.linspace(min_U-20,max_U+20,1000)

def chi_cuadrado(params, x, y):
    return y-polinomio(params,x)

p0= 1000.0, 1000.0

resultado_Uranio=leastsq(chi_cuadrado, p0, args=(Uranio,MPhD))
resultado_MPhD=leastsq(chi_cuadrado, p0, args=(MPhD,Uranio))

Uranio_min=np.roots(resultado_Uranio[0])
MPhD_min=np.roots(resultado_MPhD[0])
print Uranio_min
print MPhD_min

plt.plot(Uranio,polinomio(resultado_Uranio[0],Uranio))
plt.plot(Uranio,polinomio(resultado_MPhD[0],Uranio))
plt.show()

plt.plot(MPhD,polinomio(resultado_MPhD[0],MPhD))
plt.plot(MPhD,polinomio(resultado_MPhD[0],Uranio))
plt.show()

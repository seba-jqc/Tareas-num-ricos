#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Pregunta 1
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq


datos = np.loadtxt('correlacion.dat')

MPhD = datos[:, 1]
Uranio = datos[:, 2]

def poli_grad1(params, x):
    '''
    evalua el polinomio de grado 1 a1*x+a, donde a1, a2 = params
    '''
    a1, a2 = params
    return a1*x+a2

#min_U=np.amin(Uranio)
#max_U=np.amax(Uranio)
#real_params = np.polyfit(Uranio,MPhD,1)
#x_prueba=np.linspace(min_U-20,max_U+20,1000)

def arg_chi_2(params, datos_x, datos_y, func):
    '''
    crea el argumento de la funcion X². Este corresponde al array datos_y menos
    la funcion func.
    params corresponde a los parametros iniciales sobre los cuales se minimizara
    la funcion
    '''
    return datos_y-func(params,datos_x)

p0= 1000.0, 1000.0


resultado_Uranio=leastsq(chi_cuadrado, p0, args=(Uranio,MPhD,poli_grad1))
resultado_MPhD=leastsq(chi_cuadrado, p0, args=(MPhD,Uranio, poli_grad1))

Uranio_min=np.roots(resultado_Uranio[0])
MPhD_min=np.roots(resultado_MPhD[0])
print Uranio_min
print MPhD_min

plt.plot(Uranio,poli_grad1(resultado_Uranio[0],Uranio))
#plt.plot(Uranio,polinomio(resultado_MPhD[0],Uranio))
plt.show()

#plt.plot(MPhD,poli_grad1(resultado_MPhD[0],MPhD))
#plt.plot(MPhD,polinomio(resultado_MPhD[0],Uranio))
#plt.show()


def poli_grad5(params, x):
    a1, a2, a3, a4, a5, a6 = params
    return a1*x**5 + a2*x**4 + a3*x**3 + a4*x**2 + a5*x**1 + a6

p0_1=1, 1., 1., 0.5, 0.5, 0.5

resultado_Uranio2=leastsq(chi_cuadrado, p0_1, args=(Uranio,MPhD,poli_grad5))
plt.plot(Uranio,poli_grad5(resultado_Uranio2[0],Uranio))
#plt.plot(Uranio,polinomio(resultado_MPhD[0],Uranio))
plt.show()

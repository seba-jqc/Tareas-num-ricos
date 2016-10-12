#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.stats import norm, cauchy

datos=np.loadtxt('espectro.dat')

long_onda=datos[:,0]
flujo=datos[:,1]

def f_modelo1(params, x):
    amp, mu, sigma, a1, a2 = params
    gauss=amp*norm(loc=mu, scale=sigma).pdf(x)
    return a1*x+a2-gauss

def f_modelo2(params, x):
    amp, mu, sigma, a1, a2 = params
    lorentz=amp*cauchy(loc=mu, scale=sigma).pdf(x)
    return a1*x+a2-lorentz

def arg_chi_2(params, datos_x, datos_y, func):
    return datos_y-func(params, datos_x)

def estimador_amp(x):
    x_min=np.amin(x)
    x_max=np.amax(x)
    Amp=x_max-x_min
    return Amp

def estimador_mu(x):
    mu=np.mean(x)
    return mu

def estimador_sigma1(x):
    mu=estimador_mu(x)
    n=len(x)
    suma=0
    for i in range(n):
        dif=x[i]-mu
        suma += dif**2
    suma = suma/(n-1)
    return suma

def estimador_sigma2(x):
    sigma=np.std(x)
    return sigma

def estimador_pendiente(x,y):
    l=len(x)
    num=y[l-1]-y[0]
    den=x[l-1]-x[0]
    return num/den

def estimador_cruce(y):
    return y[0]

amp=estimador_amp(flujo)
mu1=estimador_mu(long_onda)
mu2=long_onda[np.argmin(flujo)]
sigma1=estimador_sigma1(long_onda)
sigma2=estimador_sigma2(long_onda)
a1=estimador_pendiente(long_onda, flujo)
a2=estimador_cruce(flujo)

p0= amp, mu1, sigma1, a1, a2
p1= amp, mu2, sigma2, a1, a2

resultado_Gauss1=leastsq(arg_chi_2, p0, args=(long_onda, flujo, f_modelo1))
resultado_Gauss2=leastsq(arg_chi_2, p1, args=(long_onda, flujo, f_modelo1))
resultado_lorentz1=leastsq(arg_chi_2, p0, args=(long_onda, flujo, f_modelo2))
resultado_lorentz2=leastsq(arg_chi_2, p1, args=(long_onda, flujo, f_modelo2))
#print resultado_Gauss[0]
#print resultado_lorentz[0]

plt.plot(long_onda, f_modelo1(resultado_Gauss1[0], long_onda), '-r')
plt.plot(long_onda, f_modelo1(resultado_Gauss2[0], long_onda), '-g')
plt.plot(long_onda, flujo, 'o')

plt.show()

plt.plot(long_onda, f_modelo2(resultado_lorentz1[0], long_onda), '-r')
plt.plot(long_onda, f_modelo2(resultado_lorentz2[0], long_onda), '-g')
plt.plot(long_onda, flujo, 'o')
plt.show()

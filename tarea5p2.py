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
    Amp=x_min-x_max
    return Amp

def estimador_mu(x):
    mu=np.mean(x)
    return mu

def estimador_sigma(x):
    mu=estimador_mu(x)
    n=len(x)
    suma=0
    for i in range(n):
        dif=x[i]-mu
        suma += dif**2
    suma = suma/(n-1)
    return suma

amp=estimador_amp(flujo)
mu=estimador_mu(long_onda)
sigma=estimador_sigma(long_onda)

print amp, mu, sigma

p0= amp, mu, sigma, 0, 0

resultado_Gauss=leastsq(arg_chi_2, p0, args=(long_onda, flujo, f_modelo1))
resultado_lorentz=leastsq(arg_chi_2, p0, args=(long_onda, flujo, f_modelo2))
print resultado_Gauss[0]
print resultado_lorentz[0]

plt.plot(long_onda, f_modelo1(resultado_Gauss[0], long_onda), '-r')
plt.plot(long_onda, flujo, 'o')
plt.show()

plt.plot(long_onda, f_modelo2(resultado_lorentz[0], long_onda), '-r')
plt.plot(long_onda, flujo, 'o')
plt.show()

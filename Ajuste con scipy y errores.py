#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:19:57 2024

@author: isaac
"""

import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, root_mean_squared_error

# Gaussiana
def gaussian(x,amp,mu,sigma):
    return amp*np.exp( -(x-mu)**2 / (2*sigma**2))

# Lorenziana
def lorentz(x,amp,x0,gamma):
    return amp*1/(np.pi*gamma*(1+((x-x0)/gamma)**2))

# Datos
data1 = loadtxt('/home/isaac/OneDrive/tec/IFI/6to semestre/Lab materiales/datos_imagen1_corregidos.csv',delimiter=',')
data2 = loadtxt('/home/isaac/OneDrive/tec/IFI/6to semestre/Lab materiales/datos_imagen2.csv',delimiter=',')

x1 = data1[:, 0]
y1 = data1[:, 1]

x2 = data2[:, 0]
y2 = data2[:, 1]

# Imagen 1
gauss_param1, _ = curve_fit(gaussian, x1, y1)
lorentz_param1, _ = curve_fit(lorentz, x1, y1)
y_gaussian_fit1 = gaussian(x1,*gauss_param1)
y_lorentz_fit1 = lorentz(x1,*lorentz_param1)

# Calcular R^2 y Error cuadrático medio para el ajuste Gaussiano
r2_gaussian1 = r2_score(y1, y_gaussian_fit1)
rmse_gaussian1 = root_mean_squared_error(y1, y_gaussian_fit1)

# Calcular R^2 y Error cuadrático medio para el ajuste Lorenziano
r2_lorentz1 = r2_score(y1, y_lorentz_fit1)
rmse_lorentz1 = root_mean_squared_error(y1, y_lorentz_fit1)

# Gráfica
plt.figure(1)
plt.plot(x1,y1,'b-',label='Datos')
plt.plot(x1,y_gaussian_fit1,'r--',label='Ajuste gaussiano')
plt.plot(x1,y_lorentz_fit1,'g--',label='Ajuste lorenziano')
plt.xlabel('Altura')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

print("Imagen 1:")
print(f'R^2 (Gaussian): {r2_gaussian1:.4f}, RMSE (Gaussian): {rmse_gaussian1:.4f}')
print(f'R^2 (Lorentzian): {r2_lorentz1:.4f}, RMSE (Lorentzian): {rmse_lorentz1:.4f}')
print()

# Imagen 2
gauss_param2, _ = curve_fit(gaussian, x2, y2)
lorentz_param2, _ = curve_fit(lorentz, x2, y2)
y_gaussian_fit2 = gaussian(x2,*gauss_param2)
y_lorentz_fit2 = lorentz(x2,*lorentz_param2)

# Calcular R^2 y Error cuadrático medio para el ajuste Gaussiano 
r2_gaussian2 = r2_score(y2, y_gaussian_fit2)
rmse_gaussian2 = root_mean_squared_error(y2, y_gaussian_fit2)

# Calcular R^2 y Error cuadrático medio para el ajuste Lorenziano
r2_lorentz2 = r2_score(y2, y_lorentz_fit2)
rmse_lorentz2 = root_mean_squared_error(y2, y_lorentz_fit2)

# Gráfica
plt.figure(2)
plt.plot(x2,y2,'b-',label='Datos')
plt.plot(x2,y_gaussian_fit2,'r--',label='Ajuste gaussiano')
plt.plot(x2,y_lorentz_fit2,'g--',label='Ajuste lorenziano')
plt.xlabel('Altura')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

print("Imagen 2:")
print(f'R^2 (Gaussian): {r2_gaussian2:.4f}, RMSE (Gaussian): {rmse_gaussian2:.4f}')
print(f'R^2 (Lorentzian): {r2_lorentz2:.4f}, RMSE (Lorentzian): {rmse_lorentz2:.4f}')

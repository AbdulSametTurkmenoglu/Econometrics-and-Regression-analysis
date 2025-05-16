# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:46:59 2024

@author: samet
"""

import numpy as np


Y_str = input("Lütfen Y dizisini girin: ")
X_str = input("Lütfen X dizisini girin: ")

# Girilen dizileri işleyerek numpy array'e dönüştürmek
Y = np.array(list(map(int, Y_str.split(','))))  # Virgülle ayrılan sayıları int tipine dönüştürüp numpy dizisine çeviriyoruz
X = np.array(list(map(int, X_str.split(','))))  # Aynı işlemi X dizisi için de yapıyoruz

print("Girdiğiniz verilere ilişkin doğrusal modeli EKK yöntemi ile belirleyelim.")

beta1ust = (len(X) * np.sum(X * Y)) - (np.sum(X) * np.sum(Y))
beta1alt = (len(X) * np.sum(X ** 2) - (np.sum(X)**2))
beta1 = beta1ust / beta1alt
print(f"beta1 = {beta1}")

beta0 = np.mean(Y) - (np.mean(X) * beta1)
print(f"beta0 = {beta0}")


print("EKK yöntemi ile oluşturalan model aşağıdaki gibidir")
print(f"y = {beta0} + {beta1}xi")


y_tahmin = beta0 + beta1 * X
print(f"y tahmin = {y_tahmin}")

ei = Y - y_tahmin
print(f"hatalar (ei) = {ei}")


belirtme_katsayisi = np.sum((y_tahmin - np.mean(Y))**2)/np.sum((Y - np.mean(Y))**2)

print(f"Belirtme Katsayısı(R**2) = {belirtme_katsayisi}")






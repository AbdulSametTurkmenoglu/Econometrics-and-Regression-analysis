# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:28:17 2024

@author: samet
"""
import numpy as np

print("UYARI: Lütfen dikkat edin!")
print("Bağımlı (Y) ve bağımsız (X) değişkenlerin her ikisi de aynı uzunlukta olmalıdır. "
      "Çünkü bu hesaplamalar, matris çarpımı gerektiren matematiksel işlemler içerir. "
      "Eğer dizilerin uzunlukları birbirine uymazsa, program hata verecektir.")
print("Veri formatı şu şekilde olmalıdır: "
      "Örnek: 15,10,17,16,15,22,31,8,45,10 (Y dizisi) ve 1,2,3,4,5,6,7,8,9,10 (X dizisi)")

# Y dizisi ile X dizisinin birden fazla değişken içerdiği bir formata uygun şekilde kullanıcıdan veri alıyoruz
Y_str = input("Lütfen Y dizisini girin: ")
X_str = input("Lütfen X dizisini girin: ")

# Y dizisini numpy array'e dönüştürme
Y = np.array(list(map(int, Y_str.split(','))))  # Virgülle ayrılan sayıları int tipine dönüştürüp numpy dizisine çeviriyoruz

# X dizisini (birden fazla bağımsız değişken içerecek şekilde) numpy array'e dönüştürme
X_list_of_lists = [list(map(int, x.split(','))) for x in X_str.split(';')]  # X'i ; ile ayrılmış farklı listelere ayırıyoruz
X = np.array(X_list_of_lists).T  # X dizisini uygun biçimde matrise dönüştürmek için transpoz alıyoruz

# Birler dizisi
birlerdizisi = np.ones(len(X))  # Uzunluğu X'in uzunluğuyla aynı olan birler dizisini oluşturuyoruz

# Tasarım matrisi oluşturma (Birler dizisi ve X'in birleşimi)
x_karamatrixyapma = np.column_stack((birlerdizisi, X))  # Birler dizisi ve X'i yana yana getiririz
print(f"Tasarım matrisi (X): \n{x_karamatrixyapma} \n")

# (X'X) hesaplama
x_tranpozu = x_karamatrixyapma.T  # X'in transpozunu alıyoruz
x_tranpozu_ve_x_carpimi = np.dot(x_tranpozu, x_karamatrixyapma)  # X'in transpozu ile X matrisini çarpıyoruz
print(f"(X'X): \n{x_tranpozu_ve_x_carpimi} \n")

# (X'X) tersini hesaplama
x_tranpozu_ve_x_carpiminin_tersi = np.linalg.inv(x_tranpozu_ve_x_carpimi)  # Matrisin tersini alıyoruz
print(f"(X'X)^-1 = \n{x_tranpozu_ve_x_carpiminin_tersi} \n")

# (X'Y) hesaplama
x_tranpozu_ve_y_carpimi = np.dot(x_tranpozu, Y)  # X'in transpozunu Y ile çarpıyoruz
print(f"(X'Y): \n{x_tranpozu_ve_y_carpimi} \n")

# Beta katsayılarını hesaplama (Normal equation çözümü)
betalar = np.dot(x_tranpozu_ve_x_carpiminin_tersi, x_tranpozu_ve_y_carpimi)  # Beta katsayılarını hesaplıyoruz
print(f"B = (X'X)^-1 (X'Y) --> {betalar} \n")
print(f"beta0 = {betalar[0]} \n")
for i, beta in enumerate(betalar[1:], 1):  # beta1, beta2, ... şeklinde çıktıyı döngü ile alıyoruz
    print(f"beta{i} = {beta} \n")

# Modeli yazdırma
print("Model aşağıda verilmiştir: \n")
print(f"y = {betalar[0]} ", end="")
for i, beta in enumerate(betalar[1:], 1):
    print(f"+ {beta}*X{i} ", end="")
print()

y_tahmin = betalar[0] + betalar[1:] * X

print(f"y tahmin = {y_tahmin}")



belirtme_katsayisi = np.sum((y_tahmin - np.mean(Y))**2)/np.sum((Y - np.mean(Y))**2)

print(f"Belirtme Katsayısı(R**2) = {belirtme_katsayisi}")

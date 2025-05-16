# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:26:34 2024

@author: samet
"""
import numpy as np
import coklu_dogrusal_regresyon_cozumlemesi_tasarim_matrisi_hesaplama as tmh
import cift_logaritmik_regresyon_modeli as clrm

print("MWD TESTİ (MacKinnon, White ve Davidson)")

print("hipotez kurulur")
print("H0 hipotezi doğrusallığın orijinal 𝑋 ve 𝑌 arasında olduğunu")
print("Hs ise 𝑙𝑛𝑋 ve 𝑙𝑛𝑌 arasında olduğunu ifade etmektedir.")



print("UYARI: Lütfen dikkat edin!")
print("Bağımlı (Y) ve bağımsız (X) değişkenlerin her ikisi de aynı uzunlukta olmalıdır. "
      "Çünkü bu hesaplamalar, matris çarpımı gerektiren matematiksel işlemler içerir. "
      "Eğer dizilerin uzunlukları birbirine uymazsa, program hata verecektir.")
print("Veri formatı şu şekilde olmalıdır: "
      "Örnek: 15,10,17,16,15,22,31,8,45,10 (Y dizisi) ve 1,2,3,4,5,6,7,8,9,10 (X dizisi)")

Y_str = input("Lütfen Y dizisini girin: ")
X_str = input("Lütfen X dizisini girin: ")

# Girilen dizileri işleyerek numpy array'e dönüştürmek
Y = np.array(list(map(int, Y_str.split(','))))  # Virgülle ayrılan sayıları int tipine dönüştürüp numpy dizisine çeviriyoruz
X = np.array(list(map(int, X_str.split(','))))  # Aynı işlemi X dizisi için de yapıyoruz

ho = tmh(X,Y)

yd=ho.y_tahmin

hs = clrm(X,Y)
y_sapka = hs.y_tahmin(X,Y)

W = np.log(yd) - np.log(y_sapka)
print(f" W = {W}")



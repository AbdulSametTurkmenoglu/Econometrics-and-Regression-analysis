import numpy as np


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
print(f"beta1 = {betalar[1]} \n")

# Modeli yazdırma
print("Model aşağıda verilmiştir: \n")
print(f"y = {betalar[0]} + {betalar[1]}*X \n")

# Y tahmini ve hata hesaplamaları
print("Şimdi hataları bulalım: \n")
print("y tahmini bulalım: \n")
y_tahmin = betalar[0] + betalar[1] * X  # Tahmin edilen değerler
print(f"y tahmin = {y_tahmin} \n")

# Hatalar (ei) hesaplama
ei = Y - y_tahmin
print(f"Hatalarımız (ei'ler) = {ei} \n")

# Otokorelasyon hesaplama
print("Otokorelasyonu hesaplayalım: \n")
r = np.sum(ei[:-1] * ei[1:]) / np.sum(ei**2)
print(f"Otokorelasyon katsayısı = {r} \n")

# Otokorelasyon durumu
if r == 0 :
    print("Otokorelasyon yoktur. Hata terimlerinin bağımsızlığı varsayımı bozulmamıştır. ")
elif r < 0 :
    print("Negatif otokorelasyon vardır. Hata terimlerinin bağımsızlık varsayımı bozulmuştur. ")
else :
    print("Pozitif otokorelasyon vardır. Hata terimlerinin bağımsızlık varsayımı bozulmuştur.")

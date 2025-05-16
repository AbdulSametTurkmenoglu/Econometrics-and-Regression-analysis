import numpy as np

# Bağımlı ve bağımsız değişkenler
Y = np.array([15, 10, 17, 16, 15, 22, 31, 8, 45, 10])
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Birler dizisi
birlerdizisi = np.ones(len(X)) 
# Tasarım matrisi oluşturma
x_karamatrixyapma = np.column_stack((birlerdizisi, X))  
x_tranpozu = x_karamatrixyapma.T
x_tranpozu_ve_x_carpimi = np.dot(x_tranpozu, x_karamatrixyapma)




x_tranpozu_ve_x_carpiminin_tersi=np.linalg.inv(x_tranpozu_ve_x_carpimi)
print(f"(X'X)={x_tranpozu_ve_x_carpiminin_tersi} \n")

x_tranpozu_ve_y_carpimi= np.dot(x_tranpozu,Y)
print(f"(X'Y)= {x_tranpozu_ve_y_carpimi} \n")


betalar = np.dot(x_tranpozu_ve_x_carpiminin_tersi,x_tranpozu_ve_y_carpimi) 
print(f"B = (X'X)^-1 (X'Y) -->{betalar} \n")
print(f"beta0={betalar[0]} \n")
print(f"beta1={betalar[1]} \n")



print("model aşağıda verlimiştir \n")
print(f"y = {betalar[0]} + {betalar[1]} \n")



print("şimdi hataları bulalım \n")

print("y tahmini bulalım \n")
y_tahmin = []
y_tahmin = betalar[0] + betalar[1] * X  # X ile hesaplama yap

print(f"y tahmin = {y_tahmin} \n")

ei = []
ei = Y - y_tahmin
print(f"Hatalarımız (ei'ler) ={ei} \n")




print("Otokorelasyonu hesaplayalım \n")

r = np.sum(ei[:-1] * ei[1:])  / np.sum(ei**2)

print(f"Otokorelasyon katsayısı = {r} \n")


if r == 0 :
    print("Otokorelasyon yoktur.Hata terimlerinin bağımsızlığı varsayımı bozulmamıştır ")
elif r < 0 :
    print("0'dan farklıdır negatif otokorelasyon vardır.Hata terimlerinin bağımsızlık varsayımı bozulmuştur.")
else :
    print("0'dan farklıdır poziti otokorelasyon vardır.Hata terimlerinin bağımsızlık varsayımı bozulmuştur.")
    































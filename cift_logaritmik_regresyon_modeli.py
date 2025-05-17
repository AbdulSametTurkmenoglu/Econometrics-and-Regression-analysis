import numpy as np

Y_str = input("Lütfen Y dizisini girin: ")
X_str = input("Lütfen X dizisini girin: ")

# Girilen dizileri işleyerek numpy array'e dönüştürmek
Y = np.log(np.array(list(map(int, Y_str.split(',')))))  # Virgülle ayrılan sayıları int tipine dönüştürüp numpy dizisine çeviriyoruz
X = np.log(np.array(list(map(int, X_str.split(',')))))  # Aynı işlemi X dizisi için de yapıyoruz

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
print(f" lny = {betalar[0]} + {betalar[1]}*lnX + et \n")

beta0yıldız = np.exp(betalar[0])
print(f"y = {beta0yıldız}X^{betalar[1]} +e^et \n")

print(f"elastikiyet katsayısı = {betalar[1]}'dır. X %1 artarsa Y %{betalar[1]} artacaktır")


lny_tahmin = betalar[0] + betalar[1] * X 

print(f"ln tahmin {lny_tahmin}")




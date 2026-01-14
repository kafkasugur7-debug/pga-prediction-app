import numpy as np

def model_4(Mw, Repi, Vs30):
    """
    Model-4 için PGAgeo tahmin denklemi
    
    Parametreler:
    - Mw (d0): Moment büyüklüğü
    - Repi (d1): Episantral mesafe (km)
    - Vs30 (d2): Üst 30 metredeki ortalama kayma dalgası hızı (m/s)
    
    Döndürür:
    - y: PGAgeo değeri (gal cinsinden)
    """
    # Sabitler (Tablo 5.8'den)
    C0 = 8.420
    G1C0 = C0  # ET-1 için
    
    G2C5 = 9.385
    G2C6 = -0.766
    
    G3C8 = 4.226
    
    G4C4 = -10.212
    G4C8 = 7.779
    G4C9 = 2.676
    
    # Değişkenler
    d0 = Mw
    d1 = Repi
    d2 = Vs30
    
    # Eşitlik 5.4
    term1 = np.cos(np.cos(np.cos(d1/d0 * G1C0 / (d0/d1))))
    term2 = np.tan(np.cos(G2C6 * d0 * G2C6 * G2C6) * G2C5)
    term3 = np.exp(d1) + np.tan(d2/G3C8) - d1
    term4 = 1 / np.sqrt(G4C8)
    term5 = G4C4 / (d1 - G4C9 / d1)
    
    y = term1 * term2 * term3 * term4 * term5
    
    return y

# Örnek kullanım:
Mw = 5.6      # Moment büyüklüğü
Repi = 9     # Episantral mesafe (km)
Vs30 = 771    # Kayma dalgası hızı (m/s)

PGAgeo = model_4(Mw, Repi, Vs30)
print(f"PGAgeo = {PGAgeo:.4f} gal")
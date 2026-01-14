import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Fonksiyon tanımı
def calculate_metrics_no_conversion():
    file_path = "Veri_sena_4.xlsx"
    
    # 1. Veriyi Oku
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
        return

    # Sütun isimlerindeki olası boşlukları temizle
    df.columns = [str(c).strip().replace('\xa0', ' ') for c in df.columns]

    # 2. Model Fonksiyonu (Eşitlik 5.4)
    # Varsayım: Sonuç doğrudan g birimindedir.
    def model_function(row):
        d0 = row['Mw-f']              # Mw
        d1 = row['Repi']              # Repi
        d2 = row['Mean Vs,30 (m/s)']  # Vs30
        
        # --- 1. Terim ---
        try:
            val_inner = ((d1 / d0) * 8.420) / (d0 / d1)
            cos_part = np.cos(np.cos(val_inner))
            if cos_part < 0: return np.nan
            term1 = 1.0 / np.tan(np.sqrt(cos_part))
        except:
            return np.nan

        # --- 2. Terim ---
        try:
            val_inner_2 = np.cos((-0.766 * d0 * -0.766 * -0.766)) * 9.385
            term2 = np.tan(val_inner_2)
        except:
            return np.nan

        # --- 3. Terim ---
        try:
            term3 = np.exp(d0) + np.tan(d2 / 4.226) - d1
        except:
            return np.nan

        # --- 4. Terim ---
        try:
            part_a = 1.0 / np.sqrt(np.sqrt(7.779))
            denom = (-10.212 / d1) - d1 - 2.676
            if denom == 0: return np.nan
            term4 = part_a * (1.0 / denom)
        except:
            return np.nan

        # Sonuç (Doğrudan g olarak kabul ediliyor)
        # Önceki versiyondaki / 981.0 işlemi kaldırıldı.
        y_result = term1 * term2 * term3 * term4
        return y_result

    # 3. Hesaplama
    print("Model hesaplanıyor (Dönüşümsüz)...")
    df['Predicted_g'] = df.apply(model_function, axis=1)

    # Temizlik
    df_clean = df.dropna(subset=['Predicted_g', 'geo-ort-pga'])
    
    if len(df_clean) == 0:
        print("Hata: Geçerli hesaplama yapılamadı.")
        return

    # 4. Metriklerin Hesaplanması
    y_true = df_clean['geo-ort-pga']
    y_pred = df_clean['Predicted_g']

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print("-" * 30)
    print(f"Analiz edilen satır sayısı: {len(df_clean)}")
    print(f"R^2 (Belirtme Katsayısı): {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("-" * 30)
    
    # Tahminleri görmek isterseniz ilk 5 satırı yazdırın:
    print("\nİlk 5 Tahmin Örneği:")
    print(df_clean[['Mw-f', 'Repi', 'geo-ort-pga', 'Predicted_g']].head())

if __name__ == "__main__":
    calculate_metrics_no_conversion()
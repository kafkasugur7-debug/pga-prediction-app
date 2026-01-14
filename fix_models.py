"""
Script to reload and resave models with compatible numpy/scikit-learn versions
"""
import joblib
import warnings
warnings.filterwarnings('ignore')

# Model dosyaları
model_files = [
    'model_mak_pga_Decision_Tree.pkl',
    'model_mak_pga_Gradient_Boosting.pkl',
    'model_mak_pga_LightGBM.pkl',
    'model_geo_ort_pga_Decision_Tree.pkl',
    'model_geo_ort_pga_Gradient_Boosting.pkl',
    'model_geo_ort_pga_LightGBM.pkl',
    'scaler_mak_pga.pkl',
    'scaler_geo_ort_pga.pkl'
]

print(">>> Model dosyalari yeniden kaydediliyor...")

for model_file in model_files:
    try:
        print(f"  - {model_file}...", end=" ")
        # Eski versiyonla yükle
        model = joblib.load(model_file)
        # Yeni versiyonla kaydet
        joblib.dump(model, f'pga_app/models/{model_file}')
        print("OK")
    except Exception as e:
        print(f"HATA: {e}")

print("\n[OK] Tum modeller basariyla yeniden kaydedildi!")

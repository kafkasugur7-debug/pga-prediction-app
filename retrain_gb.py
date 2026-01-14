"""
Sadece Gradient Boosting modellerini yeniden eğit
"""
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings('ignore')

print(">>> Gradient Boosting modelleri yeniden egitiliyor...")

# Veri yükleme
df = pd.read_excel("Veri_sena_4.xlsx", engine='openpyxl')
df.columns = df.columns.str.strip()

# Sütun isimlerini düzenleme
rename_map = {}
for col in df.columns:
    if 'Mw' in col or 'mw' in col.lower():
        rename_map[col] = 'Mw'
    elif 'Vs' in col or 'vs' in col.lower():
        rename_map[col] = 'Vs30'
    elif 'Repi' in col or 'repi' in col.lower():
        rename_map[col] = 'Repi'
    elif 'geo' in col.lower() and 'pga' in col.lower():
        rename_map[col] = 'geo_ort_pga'
    elif 'mak' in col.lower() and 'pga' in col.lower():
        rename_map[col] = 'mak_pga'

df.rename(columns=rename_map, inplace=True)

X = df[['Vs30', 'Repi', 'Mw']]
targets = ['geo_ort_pga', 'mak_pga']

# Gradient Boosting parametreleri
gb_params = {
    'n_estimators': [100, 200, 300], 
    'learning_rate': [0.01, 0.05, 0.1, 0.2], 
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5]
}

for target in targets:
    print(f"\n  - {target} icin Gradient Boosting egitiliyor...")
    
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Scaler'ı yükle (zaten var)
    scaler = joblib.load(f'scaler_{target}.pkl')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # GridSearchCV ile eğit
    clf = GridSearchCV(
        GradientBoostingRegressor(random_state=42), 
        gb_params, 
        cv=5, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1,
        verbose=0
    )
    clf.fit(X_train_scaled, y_train)
    
    best_model = clf.best_estimator_
    
    # Test
    y_pred = best_model.predict(X_test_scaled)
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Kaydet
    model_filename = f'model_{target}_Gradient_Boosting.pkl'
    joblib.dump(best_model, model_filename)
    
    # pga_app/models klasörüne de kopyala
    joblib.dump(best_model, f'pga_app/models/{model_filename}')
    
    print(f"    R2 = {r2:.4f} | RMSE = {rmse:.2f}")
    print(f"    Kaydedildi: {model_filename}")

print("\n[OK] Gradient Boosting modelleri basariyla yeniden egitildi!")

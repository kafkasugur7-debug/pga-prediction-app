import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
sns.set_style("white")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

print("=" * 70)
print("🎯 SEÇİLMİŞ 3 MODEL: Gradient Boosting, Decision Tree, LightGBM")
print("=" * 70)

# ===============================================================
# 1. VERİ YÜKLEME VE HAZIRLIK
# ===============================================================
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

# Girdi ve hedef değişkenler
X = df[['Vs30', 'Repi', 'Mw']]
targets = ['geo_ort_pga', 'mak_pga']
model_names = ['Gradient_Boosting', 'Decision_Tree', 'LightGBM']
model_labels = ['Gradient Boosting', 'Decision Tree', 'LightGBM']
target_labels = {
    'geo_ort_pga': 'Geometric Mean PGA (cm/s²)',
    'mak_pga': 'Maximum PGA (cm/s²)'
}

print(f"📊 Toplam Veri Sayısı: {len(df)}")
print(f"📈 Hedef Değişkenler: {targets}")
print(f"🔧 Özellikler: {list(X.columns)}\n")

# ===============================================================
# 2. MODEL TANIMLARI VE HİPERPARAMETRELER
# ===============================================================
selected_models = {
    'Gradient_Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300], 
            'learning_rate': [0.01, 0.05, 0.1, 0.2], 
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5]
        }
    },
    'Decision_Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'max_depth': [5, 10, 15, 20, None], 
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None]
        }
    },
    'LightGBM': {
        'model': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 70],
            'max_depth': [-1, 10, 20, 30],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
}

# ===============================================================
# 3. MODEL EĞİTİMİ VE DEĞERLENDİRME
# ===============================================================
all_results = []
predictions_data = {}
hyperparams_list = []

for target in targets:
    print(f"\n{'='*70}")
    print(f"🎯 HEDEF: {target}")
    print(f"{'='*70}")
    
    y = df[target]
    
    # Veri Bölme (%80 Eğitim, %20 Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Standardizasyon
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scaler'ı kaydet
    joblib.dump(scaler, f'scaler_{target}.pkl')
    print(f"✅ Scaler kaydedildi: scaler_{target}.pkl")
    
    for model_name, mp in selected_models.items():
        print(f"\n⚙️  {model_name} eğitiliyor...")
        
        try:
            # GridSearchCV ile en iyi parametreleri bulma
            clf = GridSearchCV(
                mp['model'], 
                mp['params'], 
                cv=5, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1,
                verbose=0
            )
            clf.fit(X_train_scaled, y_train)
            
            # En iyi model
            best_model = clf.best_estimator_
            best_params = clf.best_params_
            
            # Test tahmini
            y_pred = best_model.predict(X_test_scaled)
            
            # Performans metrikleri
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)
            residuals = y_test.values - y_pred
            
            # Sonuçları kaydet
            all_results.append({
                'Target': target,
                'Model': model_name,
                'R2_Score': r2,
                'RMSE': rmse,
                'MAPE': mape
            })
            
            # Hiperparametreleri kaydet
            if model_name == 'Gradient_Boosting':
                selected_params = {
                    'n_estimators': best_params.get('n_estimators'),
                    'learning_rate': best_params.get('learning_rate'),
                    'max_depth': best_params.get('max_depth'),
                    'subsample': best_params.get('subsample'),
                    'min_samples_split': best_params.get('min_samples_split')
                }
            elif model_name == 'Decision_Tree':
                selected_params = {
                    'max_depth': best_params.get('max_depth'),
                    'min_samples_split': best_params.get('min_samples_split'),
                    'min_samples_leaf': best_params.get('min_samples_leaf'),
                    'max_features': best_params.get('max_features')
                }
            elif model_name == 'LightGBM':
                selected_params = {
                    'n_estimators': best_params.get('n_estimators'),
                    'learning_rate': best_params.get('learning_rate'),
                    'num_leaves': best_params.get('num_leaves'),
                    'max_depth': best_params.get('max_depth'),
                    'subsample': best_params.get('subsample')
                }
            
            hyperparams_list.append({
                'Target': target,
                'Model': model_name,
                'Hyperparameters': str(selected_params)
            })
            
            # Modeli kaydet
            model_filename = f'model_{target}_{model_name}.pkl'
            joblib.dump(best_model, model_filename)
            
            # Tahminleri sakla (grafik için)
            predictions_data[f"{target}_{model_name}"] = {
                'model': best_model,
                'y_test': y_test.values,
                'y_pred': y_pred,
                'residuals': residuals,
                'r2': r2,
                'rmse': rmse
            }
            
            print(f"   ✅ R² = {r2:.4f} | RMSE = {rmse:.2f}")
            print(f"   📦 Model kaydedildi: {model_filename}")
            
        except Exception as e:
            print(f"   ❌ HATA: {e}")

# ===============================================================
# 4. PERFORMANS VE HİPERPARAMETRE TABLOLARINI KAYDET
# ===============================================================
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(['Target', 'R2_Score'], ascending=[True, False])
results_df.to_csv('Secilmis_3_Model_Performans.csv', index=False)

hyperparams_df = pd.DataFrame(hyperparams_list)
hyperparams_df.to_csv('Model_Hyperparameters_Q1.csv', index=False)

summary_df = pd.merge(results_df, hyperparams_df, on=['Target', 'Model'])
summary_df.to_csv('Summary_Models_Performance_Params_Q1.csv', index=False)

print(f"\n{'='*70}")
print("✅ PERFORMANS RAPORU")
print(f"{'='*70}\n")
print(results_df.to_string(index=False))

# ===============================================================
# 5. GRAFİKLER - FIGURE 1: MODEL PERFORMANS KARŞILAŞTIRMASI
# ===============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = ['#2E86AB', '#A23B72', '#F18F01']

for idx, target in enumerate(targets):
    ax = axes[idx]
    subset = results_df[results_df['Target'] == target].sort_values('R2_Score', ascending=False)
    
    x_pos = np.arange(len(subset))
    bars = ax.bar(x_pos, subset['R2_Score'], color=colors, edgecolor='black', 
                  linewidth=1.2, alpha=0.85)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels, rotation=0, ha='center')
    ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_title(f'({chr(97+idx)}) {target_labels[target]}', fontsize=11, 
                fontweight='bold', loc='left')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, (bar, r2_val) in enumerate(zip(bars, subset['R2_Score'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{r2_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('Figure1_Model_Performance_Q1.png', dpi=600, bbox_inches='tight')
plt.savefig('Figure1_Model_Performance_Q1.pdf', bbox_inches='tight')
print("\n✅ Figure 1 kaydedildi: Figure1_Model_Performance_Q1.png / .pdf")
plt.close()

# ===============================================================
# 6. GRAFİKLER - FIGURE 2: ACTUAL VS PREDICTED
# ===============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
colors_scatter = ['#2E86AB', '#A23B72', '#F18F01']

for row_idx, target in enumerate(targets):
    for col_idx, model_name in enumerate(model_names):
        ax = axes[row_idx, col_idx]
        
        key = f"{target}_{model_name}"
        y_test = predictions_data[key]['y_test']
        y_pred = predictions_data[key]['y_pred']
        r2 = predictions_data[key]['r2']
        rmse = predictions_data[key]['rmse']
        
        ax.scatter(y_test, y_pred, alpha=0.6, s=40, color=colors_scatter[col_idx], 
                  edgecolors='white', linewidth=0.5)
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.7)
        
        ax.set_xlabel('Observed PGA (cm/s²)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted PGA (cm/s²)', fontsize=10, fontweight='bold')
        
        panel_label = chr(97 + row_idx*3 + col_idx)
        if row_idx == 0:
            title_text = f'({panel_label}) {model_labels[col_idx]}\n{target_labels[targets[0]].split("(")[0].strip()}\nR²={r2:.3f}, RMSE={rmse:.2f}'
        else:
            title_text = f'({panel_label}) {model_labels[col_idx]}\n{target_labels[targets[1]].split("(")[0].strip()}\nR²={r2:.3f}, RMSE={rmse:.2f}'
        
        ax.set_title(title_text, fontsize=9, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Figure2_Actual_vs_Predicted_Q1.png', dpi=600, bbox_inches='tight')
plt.savefig('Figure2_Actual_vs_Predicted_Q1.pdf', bbox_inches='tight')
print("✅ Figure 2 kaydedildi: Figure2_Actual_vs_Predicted_Q1.png / .pdf")
plt.close()

# ===============================================================
# 7. GRAFİKLER - FIGURE 3: RESIDUAL ANALYSIS
# ===============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for row_idx, target in enumerate(targets):
    for col_idx, model_name in enumerate(model_names):
        ax = axes[row_idx, col_idx]
        
        key = f"{target}_{model_name}"
        y_pred = predictions_data[key]['y_pred']
        residuals = predictions_data[key]['residuals']
        
        ax.scatter(y_pred, residuals, alpha=0.6, s=40, color=colors_scatter[col_idx], 
                  edgecolors='white', linewidth=0.5)
        
        ax.axhline(y=0, color='k', linestyle='--', lw=2, alpha=0.7)
        
        ax.set_xlabel('Predicted PGA (cm/s²)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Residuals (cm/s²)', fontsize=10, fontweight='bold')
        
        panel_label = chr(97 + row_idx*3 + col_idx)
        if row_idx == 0:
            title_text = f'({panel_label}) {model_labels[col_idx]}\n{target_labels[targets[0]].split("(")[0].strip()}'
        else:
            title_text = f'({panel_label}) {model_labels[col_idx]}\n{target_labels[targets[1]].split("(")[0].strip()}'
        
        ax.set_title(title_text, fontsize=9, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Figure3_Residual_Analysis_Q1.png', dpi=600, bbox_inches='tight')
plt.savefig('Figure3_Residual_Analysis_Q1.pdf', bbox_inches='tight')
print("✅ Figure 3 kaydedildi: Figure3_Residual_Analysis_Q1.png / .pdf")
plt.close()

# ===============================================================
# 8. GRAFİKLER - FIGURE 4: FEATURE IMPORTANCE
# ===============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
feature_names = ['Vs30', 'Repi', 'Mw']
colors_bar = ['#2E86AB', '#A23B72', '#F18F01']

for row_idx, target in enumerate(targets):
    for col_idx, model_name in enumerate(model_names):
        ax = axes[row_idx, col_idx]
        
        key = f"{target}_{model_name}"
        model = predictions_data[key]['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            bars = ax.barh(range(len(importances)), importances[indices], 
                          color=colors_bar[col_idx], edgecolor='black', 
                          linewidth=1.0, alpha=0.85)
            ax.set_yticks(range(len(importances)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance', fontsize=10, fontweight='bold')
            
            panel_label = chr(97 + row_idx*3 + col_idx)
            if row_idx == 0:
                title_text = f'({panel_label}) {model_labels[col_idx]}\n{target_labels[targets[0]].split("(")[0].strip()}'
            else:
                title_text = f'({panel_label}) {model_labels[col_idx]}\n{target_labels[targets[1]].split("(")[0].strip()}'
            
            ax.set_title(title_text, fontsize=9, fontweight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('Figure4_Feature_Importance_Q1.png', dpi=600, bbox_inches='tight')
plt.savefig('Figure4_Feature_Importance_Q1.pdf', bbox_inches='tight')
print("✅ Figure 4 kaydedildi: Figure4_Feature_Importance_Q1.png / .pdf")
plt.close()

# ===============================================================
# 9. ÖZET RAPOR
# ===============================================================
print("\n" + "=" * 70)
print("✅ TÜM DOSYALAR BAŞARIYLA OLUŞTURULDU!")
print("=" * 70)

print("\n📦 .pkl Dosyaları (6 model + 2 scaler):")
for target in targets:
    print(f"   - scaler_{target}.pkl")
    for model_name in model_names:
        print(f"   - model_{target}_{model_name}.pkl")

print("\n📊 CSV Dosyaları:")
print("   - Secilmis_3_Model_Performans.csv")
print("   - Model_Hyperparameters_Q1.csv")
print("   - Summary_Models_Performance_Params_Q1.csv")

print("\n📈 Q1 Makale Grafikleri (PNG + PDF - 600 DPI):")
print("   - Figure1_Model_Performance_Q1")
print("   - Figure2_Actual_vs_Predicted_Q1")
print("   - Figure3_Residual_Analysis_Q1")
print("   - Figure4_Feature_Importance_Q1")

print("\n" + "=" * 70)
print("EN İYİ PERFORMANS GÖSTEREN MODELLER:")
print("=" * 70)
best_models = results_df.sort_values(['Target', 'R2_Score'], ascending=[True, False]).groupby('Target').first()
print(best_models[['Model', 'R2_Score', 'RMSE']].to_string())
print("=" * 70)
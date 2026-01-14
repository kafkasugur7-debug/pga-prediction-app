from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)

# Model yolları
MODEL_DIR = Path(__file__).parent / 'models'

# Modelleri yükle
print(">>> Modeller yukleniyor...")

models = {
    'pga_max': {
        'Decision_Tree': joblib.load(MODEL_DIR / 'model_mak_pga_Decision_Tree.pkl'),
        'Gradient_Boosting': joblib.load(MODEL_DIR / 'model_mak_pga_Gradient_Boosting.pkl'),
        'LightGBM': joblib.load(MODEL_DIR / 'model_mak_pga_LightGBM.pkl')
    },
    'pga_avg': {
        'Decision_Tree': joblib.load(MODEL_DIR / 'model_geo_ort_pga_Decision_Tree.pkl'),
        'Gradient_Boosting': joblib.load(MODEL_DIR / 'model_geo_ort_pga_Gradient_Boosting.pkl'),
        'LightGBM': joblib.load(MODEL_DIR / 'model_geo_ort_pga_LightGBM.pkl')
    }
}

scalers = {
    'pga_max': joblib.load(MODEL_DIR / 'scaler_mak_pga.pkl'),
    'pga_avg': joblib.load(MODEL_DIR / 'scaler_geo_ort_pga.pkl')
}

print("[OK] Tum modeller basariyla yuklendi!")

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """PGA tahmin endpoint'i"""
    try:
        # JSON verisini al
        data = request.get_json()
        
        # Input validasyonu
        vs30 = float(data.get('Vs30', 0))
        repi = float(data.get('Repi', 0))
        mw = float(data.get('Mw', 0))
        
        # Değer kontrolü
        if vs30 <= 0:
            return jsonify({'error': 'Vs30 değeri 0\'dan büyük olmalıdır'}), 400
        if repi <= 0:
            return jsonify({'error': 'Repi değeri 0\'dan büyük olmalıdır'}), 400
        if mw < 3.0 or mw > 9.5:
            return jsonify({'error': 'Mw değeri 3.0 ile 9.5 arasında olmalıdır'}), 400
        
        # Özellik vektörü oluştur
        features = np.array([[vs30, repi, mw]])
        
        # Tahminler
        results = {
            'pga_max': {},
            'pga_avg': {},
            'inputs': {
                'Vs30': vs30,
                'Repi': repi,
                'Mw': mw
            }
        }
        
        # PGA Max tahminleri
        features_scaled_max = scalers['pga_max'].transform(features)
        for model_name, model in models['pga_max'].items():
            prediction = model.predict(features_scaled_max)[0]
            results['pga_max'][model_name] = float(prediction)
        
        # PGA Average tahminleri
        features_scaled_avg = scalers['pga_avg'].transform(features)
        for model_name, model in models['pga_avg'].items():
            prediction = model.predict(features_scaled_avg)[0]
            results['pga_avg'][model_name] = float(prediction)
        
        # İstatistikler
        pga_max_values = list(results['pga_max'].values())
        pga_avg_values = list(results['pga_avg'].values())
        
        results['statistics'] = {
            'pga_max': {
                'mean': float(np.mean(pga_max_values)),
                'min': float(np.min(pga_max_values)),
                'max': float(np.max(pga_max_values)),
                'std': float(np.std(pga_max_values))
            },
            'pga_avg': {
                'mean': float(np.mean(pga_avg_values)),
                'min': float(np.min(pga_avg_values)),
                'max': float(np.max(pga_avg_values)),
                'std': float(np.std(pga_avg_values))
            }
        }
        
        return jsonify(results)
    
    except ValueError as e:
        return jsonify({'error': f'Geçersiz değer: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Sunucu hatası: {str(e)}'}), 500

@app.route('/health')
def health():
    """Sağlık kontrolü endpoint'i"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'pga_max': list(models['pga_max'].keys()),
            'pga_avg': list(models['pga_avg'].keys())
        }
    })

if __name__ == '__main__':
    print("=" * 70)
    print(">>> PGA Tahmin Uygulamasi Baslatiliyor...")
    print(">>> http://localhost:5000 adresinde calisiyor")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)

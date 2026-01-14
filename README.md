# 🌍 Earthquake PGA Prediction System - Aegean Region

A machine learning-based web application for predicting Peak Ground Acceleration (PGA) values for earthquakes in the Aegean Region. The system provides predictions from three different ML models for both maximum and average PGA values.

![Application Banner](static/images/earthquake-hero.png)

## 📊 Overview

This application uses ensemble machine learning models trained on Aegean Region earthquake data to predict:
- **PGA Maximum (mak_pga)**: Maximum peak ground acceleration
- **PGA Average (geo_ort_pga)**: Geometric mean peak ground acceleration

### Input Parameters
- **Vs30** (m/s): Shear wave velocity at 30 meters depth
- **Repi** (km): Epicentral distance
- **Mw**: Moment magnitude (3.0 - 9.5)

### ML Models
Three optimized models for each PGA type:
1. **Decision Tree Regressor** - R² > 0.71
2. **Gradient Boosting Regressor** - R² > 0.71
3. **LightGBM Regressor** - R² > 0.64

## 🚀 Model Performance

### PGA Maximum (mak_pga)
| Model | R² Score | RMSE |
|-------|----------|------|
| Decision Tree | 0.7606 | 25.21 |
| Gradient Boosting | 0.7137 | 27.56 |
| LightGBM | 0.6447 | 30.71 |

### PGA Average (geo_ort_pga)
| Model | R² Score | RMSE |
|-------|----------|------|
| Gradient Boosting | 0.7801 | 15.91 |
| LightGBM | 0.7458 | 17.11 |
| Decision Tree | 0.7156 | 18.09 |

## 🛠️ Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd pga_app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

## 🌐 Deployment

### Deploy to Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python Version**: 3.11.9

### Deploy to Railway

1. Create a new project on [Railway](https://railway.app)
2. Connect your GitHub repository
3. Railway will automatically detect the Flask app
4. The app will deploy using the Procfile configuration

### Deploy to Fly.io

1. Install Fly CLI
2. Run:
   ```bash
   fly launch
   fly deploy
   ```

## 📁 Project Structure

```
pga_app/
│
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── Procfile              # Deployment configuration
├── runtime.txt           # Python version
├── .gitignore           # Git ignore rules
│
├── models/              # Pre-trained ML models
│   ├── model_mak_pga_Decision_Tree.pkl
│   ├── model_mak_pga_Gradient_Boosting.pkl
│   ├── model_mak_pga_LightGBM.pkl
│   ├── model_geo_ort_pga_Decision_Tree.pkl
│   ├── model_geo_ort_pga_Gradient_Boosting.pkl
│   ├── model_geo_ort_pga_LightGBM.pkl
│   ├── scaler_mak_pga.pkl
│   └── scaler_geo_ort_pga.pkl
│
├── static/
│   ├── css/
│   │   └── style.css    # Premium styling
│   ├── js/
│   │   └── app.js       # Interactive functionality
│   └── images/          # Visual assets
│       ├── earthquake-hero.png
│       ├── aegean-map.png
│       └── acceleration-wave.png
│
└── templates/
    └── index.html       # Main page
```

## 💡 Usage Example

1. **Enter earthquake parameters:**
   - Vs30: `300` m/s
   - Repi: `50` km
   - Mw: `6.5`

2. **Click "PGA Tahmini Yap"**

3. **View results:**
   - Individual predictions from 3 models
   - Statistical summary (mean, max, min)
   - Interactive charts with range visualization
   - Results shown for both PGA Max and PGA Average

## 🎨 Features

- ✅ Modern, responsive UI with earthquake theme
- ✅ Glassmorphism and gradient effects
- ✅ Real-time predictions from 3 ML models
- ✅ Interactive Chart.js visualizations
- ✅ Statistical aggregations (mean, max, min)
- ✅ Mobile-friendly design
- ✅ Input validation and error handling
- ✅ Smooth animations and transitions

## 🔬 Methodology

The models were trained using:
- **Data Source**: Aegean Region earthquake records
- **Features**: Vs30, Repi, Mw
- **Train/Test Split**: 80/20
- **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation
- **Preprocessing**: StandardScaler normalization
- **Evaluation Metrics**: R², RMSE, MAPE

## 📝 API Endpoints

### `POST /predict`
Predict PGA values based on input parameters.

**Request Body:**
```json
{
  "Vs30": 300,
  "Repi": 50,
  "Mw": 6.5
}
```

**Response:**
```json
{
  "inputs": { "Vs30": 300, "Repi": 50, "Mw": 6.5 },
  "pga_max": {
    "Decision_Tree": 112.5,
    "Gradient_Boosting": 108.3,
    "LightGBM": 115.7
  },
  "pga_avg": {
    "Decision_Tree": 78.2,
    "Gradient_Boosting": 82.1,
    "LightGBM": 80.5
  },
  "statistics": {
    "pga_max": { "mean": 112.17, "min": 108.3, "max": 115.7 },
    "pga_avg": { "mean": 80.27, "min": 78.2, "max": 82.1 }
  }
}
```

### `GET /health`
Check application health status.

## 🤝 Contributing

This is a scientific research application. For questions or collaboration opportunities, please contact the research team.

## 📄 License

This project is developed for scientific and educational purposes.

## 🙏 Acknowledgments

- Earthquake data from Aegean Region seismic networks
- Developed using Flask, scikit-learn, LightGBM, and Chart.js
- UI design inspired by modern seismic monitoring systems

---

**Note**: PGA values are predicted in gal (cm/s²). This application is for research purposes and should not be used as the sole source for critical earthquake engineering decisions.

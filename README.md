# 🏒 NHL Attendance Predictor

> **ML-powered game attendance forecasting for NHL arenas** — built to support operational planning, staffing decisions, and revenue optimization for sports & entertainment organizations.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green.svg)](https://xgboost.readthedocs.io)

---

## 📌 Project Overview

Predicting how many fans will attend an NHL game is a real operational challenge — it affects staffing levels, concession inventory, security deployment, and revenue forecasting. This project builds a full end-to-end machine learning pipeline that:

1. **Collects** multi-season NHL game data via the public NHL Stats API
2. **Engineers** 22 features across 5 categories (temporal, team quality, matchup, logistics, arena)
3. **Trains and compares** 4 ML models: Ridge Regression, Random Forest, Gradient Boosting, XGBoost
4. **Deploys** a Streamlit web app with live predictions and what-if scenario analysis

**Achieved ~83% prediction accuracy (R²) with RMSE of ~900 fans on held-out test data.**

---

## 🎯 Business Context

This project directly mirrors challenges faced by organizations like MLSE, which manages attendance across Scotiabank Arena (Toronto Maple Leafs, Toronto Raptors), BMO Field (Toronto FC), and Coca-Cola Coliseum (Toronto Marlies).

Key use cases:
- **Operations:** Align staffing, security, and F&B inventory to expected crowd size
- **Marketing:** Identify low-attendance risk games for targeted promotional campaigns
- **Finance:** Improve revenue forecasting accuracy for gate receipts and in-venue spend
- **Fan Experience:** Anticipate crowding and optimize entry/exit flow

---

## 🗂️ Project Structure

```
nhl-attendance-predictor/
│
├── src/
│   ├── data_collection.py      # NHL API fetching + synthetic data generator
│   ├── feature_engineering.py  # Full feature pipeline (Elo, temporal, matchup)
│   └── train.py                # Model training, evaluation, SHAP analysis
│
├── app/
│   └── app.py                  # Streamlit web application
│
├── notebooks/
│   └── 01_EDA.ipynb            # Exploratory data analysis (full)
│
├── models/                     # Saved model artifacts (auto-created on train)
├── data/                       # Raw + processed data (auto-created)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Feature Engineering

| Category | Features |
|----------|----------|
| **Temporal** | Day of week, month, weekend flag, Saturday flag, near-holiday, season phase |
| **Team Quality** | Rolling win %, Elo rating (home + away), Elo differential, Elo sum |
| **Matchup** | Original Six rivalry flag, historic rivalry flag, combined win %, win % differential |
| **Logistics** | Home back-to-back, away back-to-back |
| **Arena** | Capacity, encoded team identifiers |

### Elo Rating System
Team quality is tracked using a dynamic Elo rating system (K=20, initial=1500), updated after every game. This captures team momentum far better than static win percentage — a hot team entering a stretch is rated higher than a team with the same overall record but recent losses.

**Critically, all features are computed using only data available BEFORE the game being predicted — zero data leakage.**

---

## 🤖 Models Compared

| Model | RMSE | R² | MAPE |
|-------|------|----|------|
| Ridge Regression | ~2,100 | 0.61 | 11.4% |
| Random Forest | ~1,200 | 0.79 | 6.8% |
| Gradient Boosting | ~980 | 0.82 | 5.9% |
| **XGBoost** ✓ | **~870** | **0.84** | **5.1%** |

*Results on held-out 15% time-based test split.*

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/nhl-attendance-predictor.git
cd nhl-attendance-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
# Fast demo mode (synthetic data)
python src/train.py

# Live NHL API data (slower, requires internet)
python src/train.py --live
```

### 4. Launch the web app
```bash
streamlit run app/app.py
```

### 5. Run the EDA notebook
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

## 📊 App Features

The Streamlit app provides:
- **Live prediction** — select teams, date, and form to get instant attendance forecast
- **Arena fill gauge** — visual indicator of expected fill rate vs. capacity
- **Key factor insights** — auto-generated plain-English explanations of what's driving the prediction
- **What-if scenario analysis** — compare predictions across game conditions (weekend vs. weekday, rivalry vs. standard, back-to-back, etc.)

---

## 📈 Key Findings from EDA

- **Weekend games** drive +5–8% fill rate vs. weekday games
- **Original Six rivalry matchups** carry a +3–5% attendance premium
- **April (playoff push)** is consistently the highest-attendance month
- **Arena capacity** is the single strongest predictor — larger markets fill more in absolute terms
- **Back-to-back games** suppress attendance by ~1–2%, likely correlated with team fatigue and performance concerns

---

## 🔧 Tech Stack

| Layer | Tools |
|-------|-------|
| Data | NHL Stats API, Pandas, NumPy |
| ML | scikit-learn, XGBoost, SHAP |
| Visualization | Matplotlib, Seaborn |
| App | Streamlit |
| Serialization | joblib |
| Notebooks | Jupyter |

---

## 🔮 Future Improvements

- [ ] Integrate real-time weather data (temperature, precipitation) as a feature
- [ ] Add opponent visiting city travel distance as fatigue proxy
- [ ] Build time-series forecasting layer (LSTM) to capture streaks
- [ ] Deploy to Streamlit Cloud with live NHL API integration
- [ ] Add Databricks-compatible version of the training pipeline

---

## 👤 Author

**Khalid Wasim Mushir**  
Computer Science & IT — George Brown College  
📧 khalidwasimofficial@gmail.com

---

*This project was built independently as part of a data science portfolio targeting sports & entertainment analytics roles.*

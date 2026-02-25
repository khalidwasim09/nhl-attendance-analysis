"""
train.py
--------
Trains, evaluates, and serializes the NHL attendance prediction model.

Pipeline:
    1. Load data  →  feature engineering
    2. Train/test split (time-based, no shuffling — avoids leakage)
    3. Train 4 candidate models
    4. Evaluate with RMSE, MAE, R², MAPE
    5. Select best model
    6. SHAP feature importance analysis
    7. Save model + encoders + scaler to models/

Usage:
    python src/train.py
    python src/train.py --live     # use live NHL API (slower)
"""

import argparse
import os
import sys
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping XGB model")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not installed — skipping SHAP analysis")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_collection import collect_all_data
from src.feature_engineering import build_features, FEATURE_COLS, TARGET_COL


# ── Metrics ────────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    print(f"\n  {name}")
    print(f"    RMSE : {rmse:,.0f} fans")
    print(f"    MAE  : {mae:,.0f} fans")
    print(f"    R²   : {r2:.4f}")
    print(f"    MAPE : {mape:.2f}%")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ── Plot Helpers ────────────────────────────────────────────────────────────────

def plot_predictions(y_true, y_pred, name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{name} — Predictions vs Actual", fontsize=14, fontweight="bold")

    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.35, s=10, color="#1a73e8")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Attendance")
    axes[0].set_ylabel("Predicted Attendance")
    axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=40, color="#34a853", edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Residual (Predicted − Actual)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{name.replace(' ', '_')}_eval.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_feature_importance(model, feature_names, out_dir):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color="#1a73e8", edgecolor="white"
    )
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Top 15 Features — NHL Attendance Predictor", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, importances[indices[::-1]]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance chart saved to {path}")


def plot_model_comparison(results: list[dict], out_dir: str):
    df = pd.DataFrame(results).sort_values("rmse")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")

    colors = ["#1a73e8", "#34a853", "#fbbc04", "#ea4335"]
    metrics = [("rmse", "RMSE (fans)", False), ("r2", "R²", True), ("mape", "MAPE (%)", False)]

    for ax, (metric, label, higher_better) in zip(axes, metrics):
        vals = df[metric].values
        names = df["model"].values
        bar_colors = colors[:len(names)]
        if higher_better:
            bar_colors = list(reversed(bar_colors))
        bars = ax.bar(names, vals, color=bar_colors, edgecolor="white")
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.3f}" if metric != "rmse" else f"{v:,.0f}",
                    ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main Training Pipeline ─────────────────────────────────────────────────────

def train(use_live: bool = False):
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Data
    print("=" * 60)
    print("NHL ATTENDANCE PREDICTOR — TRAINING PIPELINE")
    print("=" * 60)
    print("\n[1/5] Loading data...")
    raw = collect_all_data(use_synthetic=not use_live)
    raw.to_csv("data/nhl_games_raw.csv", index=False)
    print(f"  {len(raw)} games loaded")

    # 2. Features
    print("\n[2/5] Engineering features...")
    X, enriched, meta = build_features(raw)
    y = enriched[TARGET_COL].values
    print(f"  Feature matrix: {X.shape}")

    # 3. Time-based train/test split (last 15% = test, no shuffle)
    print("\n[3/5] Splitting data (time-based)...")
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"  Train: {len(X_train)} games | Test: {len(X_test)} games")

    # 4. Define models
    print("\n[4/5] Training models...")
    models = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=4,
            max_features="sqrt", n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.8, min_samples_leaf=5, random_state=42
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=0
        )

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(name, y_test, y_pred)
        results.append(metrics)
        trained_models[name] = (model, y_pred)

        plot_predictions(y_test, y_pred, name, "models")

    # 5. Select best by RMSE
    print("\n[5/5] Selecting best model & saving...")
    best = min(results, key=lambda r: r["rmse"])
    best_name = best["model"]
    best_model, best_preds = trained_models[best_name]
    print(f"\n  ✓ Best model: {best_name} (RMSE: {best['rmse']:,.0f})")

    # Feature importance
    core_model = best_model.steps[-1][1] if hasattr(best_model, "steps") else best_model
    plot_feature_importance(core_model, FEATURE_COLS, "models")
    plot_model_comparison(results, "models")

    # SHAP analysis
    if HAS_SHAP and hasattr(core_model, "feature_importances_"):
        print("\n  Running SHAP analysis...")
        try:
            explainer = shap.TreeExplainer(core_model)
            shap_values = explainer.shap_values(X_test.values[:200])
            fig = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test.values[:200],
                              feature_names=FEATURE_COLS, show=False)
            plt.tight_layout()
            plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  SHAP summary saved to models/shap_summary.png")
        except Exception as e:
            print(f"  SHAP failed: {e}")

    # Save artifacts
    scaler = StandardScaler().fit(X_train)
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(meta["encoders"], "models/encoders.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(FEATURE_COLS, "models/feature_cols.joblib")

    model_meta = {
        "best_model": best_name,
        "metrics": best,
        "all_results": results,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_cols": FEATURE_COLS,
    }
    with open("models/model_meta.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Training complete.")
    print(f"  Model artifacts saved to models/")
    print("=" * 60)

    return best_model, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Use live NHL API instead of synthetic data")
    args = parser.parse_args()
    train(use_live=args.live)

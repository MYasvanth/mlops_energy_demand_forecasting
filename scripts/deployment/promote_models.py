"""
Model Promotion Script - MLOps Best Practice
Promotes trained models from staging to production with versioning and metadata.
"""
import shutil
from pathlib import Path
from datetime import datetime
import json

def promote_to_production():
    """Promote all recently trained models to production."""
    models_dir = Path("models")
    production_dir = models_dir / "production"
    production_dir.mkdir(exist_ok=True)
    
    # Model files to promote
    model_files = [
        "lstm_model_lstm.h5",
        "lstm_model_scaler.pkl",
        "prophet_model_prophet.pkl",
        "arima_model_fitted.pkl",
        "arima_model_params.pkl",
        "lightgbm_gbm_model.joblib",
        "lightgbm_gbm_model_feature_engineer.joblib",
        "xgboost_gbm_model.joblib",
        "xgboost_gbm_model_feature_engineer.joblib"
    ]
    
    promoted = []
    for model_file in model_files:
        src = models_dir / model_file
        if src.exists():
            dst = production_dir / model_file
            shutil.copy2(src, dst)
            promoted.append(model_file)
            print(f"[OK] Promoted: {model_file}")
    
    # Create metadata
    metadata = {
        "promotion_date": datetime.now().isoformat(),
        "models": promoted,
        "version": "v1.0.0"
    }
    
    with open(production_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[SUCCESS] Promoted {len(promoted)} models to production")
    return promoted

if __name__ == "__main__":
    promote_to_production()

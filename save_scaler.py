from src.utils.data_processor import fetch_credit_card_data, engineer_risk_features, prepare_model_data
import joblib
import os

# Load and prepare data
df = fetch_credit_card_data('data/credit_card_default.xls')
df = engineer_risk_features(df)
X_train, X_test, y_train, y_test, scaler = prepare_model_data(df)

# Save the scaler
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.joblib')
print("✅ Scaler saved to models/scaler.joblib")

# Also save the feature columns (to ensure correct order during inference)
feature_cols = X_train.columns.tolist()
joblib.dump(feature_cols, 'models/feature_columns.joblib')
print("✅ Feature columns saved to models/feature_columns.joblib")

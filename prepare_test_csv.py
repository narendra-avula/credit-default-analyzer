# prepare_test_csv.py
# Create a test CSV file for the Streamlit app

from src.utils.data_processor import fetch_credit_card_data, engineer_risk_features, prepare_model_data

df = fetch_credit_card_data('data/credit_card_default.xls')
df = engineer_risk_features(df)
X_train, X_test, y_train, y_test, scaler = prepare_model_data(df)

# Combine X_test and y_test
test_data = X_test.copy()
test_data['default_flag'] = y_test.values

# Save to CSV
test_data.to_csv('test_data.csv', index=False)
print("✅ test_data.csv created successfully!")
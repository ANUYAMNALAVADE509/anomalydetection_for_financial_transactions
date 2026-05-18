# =========================================================
# IMPORT LIBRARIES
# =========================================================
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import traceback
from datetime import datetime

# =========================================================
# INITIALIZE FLASK APP
# =========================================================
app = Flask(__name__)
CORS(app)

# =========================================================
# GLOBALS FOR CACHING
# =========================================================
MODELS_LOADED = False
iso_forest = None
scaler_standard = None
feature_columns = None

# Pre‑computed mappings for one‑hot encoding (speed)
CAT_MAPPINGS = {
    'TransactionType': {},
    'Channel': {},
    'Location': {}
}

# =========================================================
# LOAD MODELS AND PREPARE MAPPINGS
# =========================================================
try:
    iso_forest = joblib.load('models/isolation_forest.pkl')
    scaler_standard = joblib.load('models/scaler_standard.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')

    # Extract unique categories for each prefix from the saved feature columns
    for col in feature_columns:
        if col.startswith('TransactionType_'):
            cat = col.replace('TransactionType_', '')
            CAT_MAPPINGS['TransactionType'][cat] = col
        elif col.startswith('Channel_'):
            cat = col.replace('Channel_', '')
            CAT_MAPPINGS['Channel'][cat] = col
        elif col.startswith('Location_'):
            cat = col.replace('Location_', '')
            CAT_MAPPINGS['Location'][cat] = col

    MODELS_LOADED = True
    print("\n===================================")
    print("MODELS LOADED SUCCESSFULLY")
    print(f"Total features expected: {len(feature_columns)}")
    print("===================================")
except Exception as e:
    print("\n===================================")
    print("MODEL LOADING FAILED")
    print("===================================")
    print(str(e))


# =========================================================
# HOME ROUTE
# =========================================================
@app.route('/')
def home():
    return render_template('index.html')


# =========================================================
# HEALTH CHECK
# =========================================================
@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'models_loaded': MODELS_LOADED,
        'total_features': len(feature_columns) if MODELS_LOADED else 0
    })


# =========================================================
# PREDICTION ROUTE (OPTIMIZED)
# =========================================================
@app.route('/predict', methods=['POST'])
def predict():
    if not MODELS_LOADED:
        return jsonify({'success': False, 'error': 'Models not loaded'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data'}), 400

        # ----- Extract & validate inputs -----
        required = ['amount', 'duration', 'balance', 'age',
                    'login_attempts', 'transaction_type',
                    'channel', 'location', 'transaction_time']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'success': False, 'error': f'Missing fields: {missing}'}), 400

        amount = float(data['amount'])
        duration = float(data['duration'])
        balance = float(data['balance'])
        age = int(data['age'])
        login_attempts = int(data['login_attempts'])
        trans_type = data['transaction_type'].strip()
        channel = data['channel'].strip()
        location = data['location'].strip()

        # Parse datetime
        try:
            trans_time = pd.to_datetime(data['transaction_time'])
        except:
            trans_time = datetime.now()
        hour = trans_time.hour
        day_of_week = trans_time.weekday()
        month = trans_time.month
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # ----- Feature engineering -----
        amount_balance_ratio = amount / (balance + 1)
        insufficient_balance_flag = 1 if amount > balance else 0
        high_login_attempt_flag = 1 if login_attempts >= 3 else 0
        high_amount_flag = 1 if amount > 500000 else 0
        odd_hour_flag = 1 if (hour < 5 or hour > 23) else 0

        # Rolling features – not available for single transaction → set to 0
        rolling_mean_5 = 0.0
        rolling_std_5 = 0.0
        amount_deviation = 0.0

        # ----- Build feature dictionary -----
        features = {
            'TransactionAmount': amount,
            'TransactionDuration': duration,
            'AccountBalance': balance,
            'CustomerAge': age,
            'LoginAttempts': login_attempts,
            'TransactionHour': hour,
            'TransactionDayOfWeek': day_of_week,
            'TransactionMonth': month,
            'IsWeekend': is_weekend,
            'AmountBalanceRatio': amount_balance_ratio,
            'InsufficientBalanceFlag': insufficient_balance_flag,
            'HighLoginAttemptFlag': high_login_attempt_flag,
            'HighAmountFlag': high_amount_flag,
            'OddHourFlag': odd_hour_flag,
            'RollingMean_5': rolling_mean_5,
            'RollingStd_5': rolling_std_5,
            'AmountDeviation': amount_deviation
        }

        # ----- One‑hot encoding (using pre‑computed mappings) -----
        # TransactionType
        for cat, col_name in CAT_MAPPINGS['TransactionType'].items():
            features[col_name] = 1 if trans_type == cat else 0
        # Channel
        for cat, col_name in CAT_MAPPINGS['Channel'].items():
            features[col_name] = 1 if channel == cat else 0
        # Location
        for cat, col_name in CAT_MAPPINGS['Location'].items():
            features[col_name] = 1 if location == cat else 0

        # ----- Ensure all expected columns exist -----
        X_input = pd.DataFrame([features])
        for col in feature_columns:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[feature_columns]  # correct order

        # ----- Scale and predict -----
        X_scaled = scaler_standard.transform(X_input)
        pred = iso_forest.predict(X_scaled)[0]  # -1 = anomaly, 1 = normal
        anomaly_score_raw = iso_forest.score_samples(X_scaled)[0]
        anomaly_score = abs(float(anomaly_score_raw))  # make positive for display

        # ----- Business rules + risk scoring -----
        risk_points = 0
        fraud_reasons = []

        if amount > balance:
            fraud_reasons.append("Amount exceeds account balance")
            risk_points += 5
        if login_attempts >= 3:
            fraud_reasons.append("Multiple login attempts detected")
            risk_points += 2
        if amount > 500000:
            fraud_reasons.append("Very high transaction amount")
            risk_points += 3
        if odd_hour_flag:
            fraud_reasons.append("Odd‑hour transaction detected")
            risk_points += 1
        if amount_balance_ratio > 0.9:
            fraud_reasons.append("Transaction amount close to total balance")
            risk_points += 2
        if pred == -1:
            fraud_reasons.append("ML model detected anomalous behaviour")
            risk_points += 3

        # Final risk classification
        if risk_points >= 7:
            risk_level = "High"
            recommendation = "Block transaction immediately"
            is_anomaly = True
        elif risk_points >= 4:
            risk_level = "Medium"
            recommendation = "Manual verification required"
            is_anomaly = True
        else:
            risk_level = "Low"
            recommendation = "Transaction approved"
            is_anomaly = False

        return jsonify({
            'success': True,
            'is_anomaly': is_anomaly,
            'risk_level': risk_level,
            'risk_points': risk_points,
            'anomaly_score': round(anomaly_score, 4),
            'fraud_reasons': fraud_reasons,
            'recommendation': recommendation
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    print("\n===================================")
    print("FRAUD DETECTION SYSTEM STARTED")
    print("===================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
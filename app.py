import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
from model_definition import RegressionNN
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ Load checkpoint
checkpoint = torch.load('model_checkpoint_without_b_star.pth', map_location=device, weights_only=False)

model = RegressionNN(input_dim=checkpoint['input_dim'], output_dim=checkpoint['output_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

X_scaler = checkpoint['x_scaler']
y_scaler = checkpoint['y_scaler2']

FEATURES = ['pred_x_km', 'pred_y_km', 'pred_z_km', 'pred_vx_km_s', 'pred_vy_km_s', 'pred_vz_km_s',
            'KP_SUM_mean', 'AP_AVG_mean', 'F10.7_ADJ_mean', 'F10.7_ADJ_LAST81_mean', 'F10.7_OBS_mean',
           'grav_potential', 'grav_x', 'grav_y', 'grav_z', 'grav_magnitude']
TARGETS = ['real_x_km', 'real_y_km', 'real_z_km', 'real_vx_km_s', 'real_vy_km_s', 'real_vz_km_s']

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Prepare input
        if isinstance(data, dict):
            input_data = [data[feature] for feature in FEATURES]
            X = np.array([input_data])
        elif isinstance(data, list):
            X = np.array([[row[feature] for feature in FEATURES] for row in data])
        else:
            return jsonify({"error": "Invalid input format"}), 400

        # Scale and predict
        X_scaled = X_scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            y_pred_scaled = model(X_tensor).cpu().numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        # ✅ Convert float32 to native Python float
        predictions = [
            {key: float(value) for key, value in zip(TARGETS, row)}
            for row in y_pred
        ]

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "ML Model Prediction API is running!"

if __name__ == "__main__":
    app.run(debug=True)


# import warnings
# warnings.filterwarnings('ignore')

# from flask import Flask, request, jsonify
# import torch
# import numpy as np
# import pandas as pd
# from model_definition import RegressionNN
# from sklearn.preprocessing import StandardScaler
#ballistic_coefficient
# app = Flask(__name__)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # ✅ Safe load without needing _safe_globals
# checkpoint = torch.load('model_checkpoint.pth', map_location=device, weights_only=False)

# model = RegressionNN(input_dim=checkpoint['input_dim'], output_dim=checkpoint['output_dim'])
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(device)
# model.eval()

# X_scaler = checkpoint['x_scaler']
# y_scaler = checkpoint['y_scaler2']

# FEATURES = ['pred_x_km', 'pred_y_km', 'pred_z_km', 'pred_vx_km_s', 'pred_vy_km_s', 'pred_vz_km_s',
#             'KP_SUM_mean', 'AP_AVG_mean', 'F10.7_ADJ_mean', 'F10.7_ADJ_LAST81_mean', 'F10.7_OBS_mean',
#             'ballistic_coefficient', 'grav_potential', 'grav_x', 'grav_y', 'grav_z', 'grav_magnitude']
# TARGETS = ['real_x_km', 'real_y_km', 'real_z_km', 'real_vx_km_s', 'real_vy_km_s', 'real_vz_km_s']

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         if isinstance(data, dict):
#             input_data = [data[feature] for feature in FEATURES]
#             X = np.array([input_data])
#         elif isinstance(data, list):
#             X = np.array([[row[feature] for feature in FEATURES] for row in data])
#         else:
#             return jsonify({"error": "Invalid input format"}), 400

#         X_scaled = X_scaler.transform(X)
#         X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

#         with torch.no_grad():
#             y_pred_scaled = model(X_tensor).cpu().numpy()
#         y_pred = y_scaler.inverse_transform(y_pred_scaled)
        

#         predictions = [dict(zip(TARGETS, row)) for row in y_pred]
#         return jsonify(predictions)
      

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/", methods=["GET"])
# def home():
#     return "ML Model Prediction API is running!"

# if __name__ == "__main__":
#     app.run(debug=True)

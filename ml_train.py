import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, Events, JSONLogger
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import shap

from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

# PID Optimization Function
def pid_objective_function(Kp, Ki, Kd):
    env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=False, demo_mode=False)
    obs, _ = env.reset()
    
    dt = 1.0 / env.CTRL_FREQ
    max_steps = 50
    total_error = 0.0
    total_control = 0.0
    error_integral = 0.0
    prev_error = 0.0
    lambda_coef = 0.1
    steps_run = 0

    for _ in range(max_steps):
        steps_run += 1
        pos = np.array(obs[0:3])
        target = np.array(env.current_target)
        
        error_scalar = np.linalg.norm(target - pos)
        error_integral += error_scalar * dt
        error_derivative = (error_scalar - prev_error) / dt
        prev_error = error_scalar

        control_value = Kp * error_scalar + Ki * error_integral + Kd * error_derivative
        thrust = np.clip(0.5 + 0.1 * control_value, 0, 1)
        action = np.array([thrust, 0.5, 0.5, 0.5])

        obs, reward, done, _, info = env.step(action)

        total_error += error_scalar
        total_control += np.abs(control_value)

        if done:
            break

    env.close()

    avg_error = total_error / steps_run
    avg_control = total_control / steps_run
    loss = avg_error + lambda_coef * avg_control

    return -loss

# Define parameter bounds for Bayesian Optimization
pid_pbounds = {
    'Kp': (0.1, 10.0),
    'Ki': (0.001, 1.0),
    'Kd': (0.01, 5.0)
}

optimizer_pid = BayesianOptimization(
    f=pid_objective_function,
    pbounds=pid_pbounds,
    random_state=42,
    verbose=2
)

logger_pid = JSONLogger(path="./bayes_opt_pid_log.json")
optimizer_pid.subscribe(Events.OPTIMIZATION_STEP, logger_pid)

print("Starting Bayesian Optimization for PID gains...")

try:
    optimizer_pid.maximize(init_points=3, n_iter=10)
except KeyboardInterrupt:
    print("PID Bayesian Optimization interrupted by user!")

print("PID Optimization complete.")
print("Best PID parameters:", optimizer_pid.max)

# Extract best PID values
if optimizer_pid.max is None or "params" not in optimizer_pid.max:
    best_pid = {"Kp": 5.0, "Ki": 0.2, "Kd": 1.5}
else:
    best_pid = optimizer_pid.max["params"]

waypoint_threshold = 0.2
desired_extra_thrust = 0.05

if 'scaling_factor' not in best_pid:
    if best_pid["Kp"] != 0:
        best_pid["scaling_factor"] = desired_extra_thrust / (best_pid["Kp"] * waypoint_threshold)
    else:
        best_pid["scaling_factor"] = 0.1

print("Using PID gains:", best_pid)

# Save best PID values
pid_output_path = r"C:\Users\varsh\OneDrive\Documents\Drone_navi\data\best_pid_values.csv"
os.makedirs(os.path.dirname(pid_output_path), exist_ok=True)
pid_df = pd.DataFrame([best_pid])
pid_df.to_csv(pid_output_path, index=False)
print(f"Best PID values saved to {pid_output_path}")

# Load collected data
csv_path = r"C:\Users\varsh\OneDrive\Documents\Drone_navi\data\pid_data_preprocessed_stratified.csv"
data = pd.read_csv(csv_path)
print("Data loaded. Data shape:", data.shape)

if 'Error_Dist_prev' not in data.columns or 'Error_Dist_diff' not in data.columns:
    data['Error_Dist_prev'] = data['Error_Dist'].shift(1)
    data['Error_Dist_diff'] = data['Error_Dist'] - data['Error_Dist_prev']
    data['Error_Dist_prev'] = data['Error_Dist_prev'].fillna(method='bfill')

# Feature and target selection
feature_columns = [
    "x", "y", "z", "Roll", "Pitch", "Yaw", "vx", "vy", "vz", "wx", "wy", "wz",
    "Target_x", "Target_y", "Target_z", "Error_Dist", "Error_Dist_prev", "Error_Dist_diff"
]
target_columns = ["RPM_1", "RPM_2", "RPM_3", "RPM_4"]

X_orig = data[feature_columns].values
y = data[target_columns].values

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_orig = imputer.fit_transform(X_orig)

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_orig)

print("Original feature shape:", X_orig.shape)
print("Expanded feature shape (degree=2):", X_poly.shape)

# Normalize Data
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X_poly)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
xgb_estimator = xgb.XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=1
)

model_xgb = MultiOutputRegressor(xgb_estimator)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

# Train Random Forest Model
rf_estimator = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model_rf = MultiOutputRegressor(rf_estimator)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Ensemble Model Prediction
y_pred_ensemble = (y_pred_xgb + y_pred_rf) / 2

# Evaluation Function
def evaluate_model(y_true, y_pred, model_name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Evaluation:")
    print(" RMSE:", rmse)
    print(" MAE:", mae)
    print(" R2 Score:", r2)
    
    return rmse, mae, r2

# Evaluate Models
print("XGBoost Metrics:")
metrics_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost")

print("Random Forest Metrics:")
metrics_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

print("Ensemble Metrics:")
ensemble_metrics = evaluate_model(y_test, y_pred_ensemble, "Ensemble")

# Feature Importance and SHAP Analysis
importances = np.mean([est.feature_importances_ for est in model_xgb.estimators_], axis=0)
feature_names_poly = poly.get_feature_names_out(feature_columns)
sorted_idx = np.argsort(importances)[::-1]

print("\nTop 20 important features:")
for idx in sorted_idx[:20]:
    print(f"{feature_names_poly[idx]}: {importances[idx]:.4f}")

for i, target in enumerate(target_columns):
    def predict_i(x, target_idx=i):
        return model_xgb.predict(x)[:, target_idx]

    background = X_train[:50]
    explainer = shap.KernelExplainer(predict_i, background)
    shap_vals = explainer.shap_values(X_test[:50], nsamples=100)

    print(f"SHAP summary for {target} (XGBoost component):")
    shap.summary_plot(shap_vals, X_test[:50], feature_names=feature_names_poly)

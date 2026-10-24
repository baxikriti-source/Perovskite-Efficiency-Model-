import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. DATASET CREATION / PREPROCESSING
# ==========================================
# Simulating experimental synthesis & device operation data for CsPbBr3 photodetectors
np.random.seed(42)
num_samples = 300

data = {
    # Synthesis & Material Features
    'Cs_Pb_Ratio': np.random.uniform(0.8, 1.2, num_samples),
    'Annealing_Temp_C': np.random.uniform(80, 180, num_samples),
    'Annealing_Time_min': np.random.uniform(5, 60, num_samples),
    'Film_Thickness_nm': np.random.uniform(100, 500, num_samples),
    'Grain_Size_nm': np.random.uniform(20, 200, num_samples),
    
    # Device Operating Conditions
    'Bias_Voltage_V': np.random.uniform(0.0, 5.0, num_samples),
    'Light_Intensity_mW_cm2': np.random.uniform(0.1, 10.0, num_samples),
    'Wavelength_nm': np.random.choice([365, 405, 450, 520, 635], num_samples)
}

df = pd.DataFrame(data)

# Target physics relationships (responsivity & detectivity based on real physical trends)
# R increases with optimal stoichiometry (~1.0), optimal grain size, bias voltage, and blue/UV light
ideal_ratio_factor = np.exp(-((df['Cs_Pb_Ratio'] - 1.0) ** 2) / 0.05)
wavelength_factor = np.where(df['Wavelength_nm'] <= 520, (530 - df['Wavelength_nm']) / 100, 0.05)

df['Responsivity_A_W'] = (
    0.5 * ideal_ratio_factor +
    0.02 * (df['Grain_Size_nm'] / 10) +
    0.15 * df['Bias_Voltage_V'] +
    0.8 * wavelength_factor +
    np.random.normal(0, 0.05, num_samples)
)
df['Responsivity_A_W'] = df['Responsivity_A_W'].clip(lower=0.01) # Non-negative constraint

# Detectivity (Jones) correlates with Responsivity and inverse dark current noise
df['Detectivity_Jones_x1011'] = (
    df['Responsivity_A_W'] * 3.5 - 
    0.05 * (df['Annealing_Temp_C'] - 120)**2 / 1000 + 
    np.random.normal(0, 0.1, num_samples)
).clip(lower=0.1)

# Split features (X) and targets (y)
X = df.drop(columns=['Responsivity_A_W', 'Detectivity_Jones_x1011'])
y_responsivity = df['Responsivity_A_W']

# Partition Data: Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_responsivity, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. MODEL TRAINING & SELECTION
# ==========================================
# Using Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ==========================================
# 3. EVALUATION & METRICS
# ==========================================
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("="*45)
print("  CsPbBr3 PHOTODETECTOR MODEL EVALUATION")
print("="*45)
print(f"Root Mean Square Error (RMSE): {rmse:.4f} A/W")
print(f"Coefficient of Determination (R²): {r2 * 100:.2f}%")
print("="*45)

# ==========================================
# 4. FEATURE IMPORTANCE ANALYSIS
# ==========================================
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Visualization Plot
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance for CsPbBr3 Photodetector Responsivity Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Experimental & Operating Feature')
plt.tight_layout()
plt.show()

# ==========================================
# 5. PREDICTING FOR A NEW SAMPLE
# ==========================================
new_experimental_sample = pd.DataFrame([{
    'Cs_Pb_Ratio': 1.0,
    'Annealing_Temp_C': 120,
    'Annealing_Time_min': 30,
    'Film_Thickness_nm': 250,
    'Grain_Size_nm': 150,
    'Bias_Voltage_V': 2.0,
    'Light_Intensity_mW_cm2': 1.0,
    'Wavelength_nm': 405
}])

new_sample_scaled = scaler.transform(new_experimental_sample)
predicted_R = model.predict(new_sample_scaled)[0]

print(f"\n[INFERENCE] Predicted Responsivity for New Setup: {predicted_R:.3f} A/W")

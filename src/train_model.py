# src/train_model.py
import json
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Load dataset ---
data = fetch_california_housing(as_frame=True)
X = data.data[['MedInc', 'HouseAge', 'AveRooms']]
y = data.target

# --- 2. Split and scale ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- 3. Train model ---
model = LinearRegression()
model.fit(X_train_s, y_train)

# --- 4. Save tiny artifact ---
artifact = {
    "feature_names": list(X.columns),
    "coef": list(model.coef_),
    "intercept": float(model.intercept_),
    "scaler_mean": list(scaler.mean_),
    "scaler_scale": list(scaler.scale_),
}

with open("model.json", "w") as f:
    json.dump(artifact, f, indent=2)

# --- 5. Create sample inputs for later testing ---
X_test_sample = X_test.iloc[:3].to_dict(orient="records")
with open("sample_input.json", "w") as f:
    json.dump(X_test_sample, f, indent=2)

print("✅ Done — created model.json and sample_input.json")

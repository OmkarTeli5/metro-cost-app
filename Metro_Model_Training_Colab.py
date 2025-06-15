
# 🚧 Metro Civil Cost Prediction - Model Training (Colab Pro+ Ready)
# ================================================================

# 📌 STEP 1: Install & Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 📌 STEP 2: Load Dataset
df = pd.read_csv("Metro_Data_Cleaned.csv")

# 📌 STEP 3: Define Features and Target
X = df.drop(columns=["Total_Civil_Cost_Crore"])
y = df["Total_Civil_Cost_Crore"]

# 📌 STEP 4: Preprocessing for Categorical Columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Custom: Drop TBM_Diameter if TBM_Used = No
if "TBM_Used" in categorical_cols and "TBM_Diameter_m" in numeric_cols:
    numeric_cols.remove("TBM_Diameter_m")

# 📌 STEP 5: Define Column Transformer
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder='passthrough')

# 📌 STEP 6: Create Pipeline with RandomForest
model = Pipeline([
    ("pre", preprocessor),
    ("reg", RandomForestRegressor(n_estimators=100, random_state=42))
])

# 📌 STEP 7: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 STEP 8: Fit the Model
model.fit(X_train, y_train)

# 📌 STEP 9: Evaluate Model
y_pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("R² Score:", round(r2_score(y_test, y_pred), 3))

# 📌 STEP 10: Save Model to File
joblib.dump(model, "metro_cost_model.pkl")
print("✅ Model saved as 'metro_cost_model.pkl'")

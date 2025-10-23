# Step 1: I Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Step 2: Load Dataset
# Download the dataset from Kaggle:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Place 'train.csv' in the same directory as this script.

data = pd.read_csv(r"C:\Users\Lenovo\Desktop\Python\Internship-1\train.csv")
print("✅ Dataset Loaded Successfully")
print("Shape of dataset:", data.shape)
print(data.head())

# =====================================
# Step 3: Data Exploration
# =====================================
print("\n--- Dataset Info ---")
print(data.info())

print("\n--- Missing Values ---")
print(data.isnull().sum().sort_values(ascending=False).head(10))

# Visualize missing values
plt.figure(figsize=(10,6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# =====================================
# Step 4: Feature Selection
# =====================================
# Select numerical and categorical features
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
            'FullBath', 'YearBuilt', 'Neighborhood']

target = 'SalePrice'

X = data[features]
y = data[target]

# =====================================
# Step 5: Handle Missing Values
# =====================================
X = X.fillna(X.mean())

# =====================================
# Step 6: One-Hot Encode Categorical Data
# =====================================
categorical_features = ['Neighborhood']
numerical_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# Preprocess: encode categorical + pass numeric as is
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_features),
    ('num', 'passthrough', numerical_features)
])

# =====================================
# Step 7: Split Dataset
# =====================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n✅ Data split completed.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# =====================================
# Step 8: Build Linear Regression Pipeline
# =====================================
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# =====================================
# Step 9: Train the Model
# =====================================
model.fit(X_train, y_train)
print("\n✅ Model training completed.")

# =====================================
# Step 10: Predict on Test Data
# =====================================
y_pred = model.predict(X_test)

# =====================================
# Step 11: Evaluate Model Performance
# =====================================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(np.sqrt(mse), 2))
print("R-squared Score:", round(r2, 4))

# =====================================
# Step 12: Visualize Actual vs Predicted Prices
# =====================================
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Sale Prices")
plt.ylabel("Predicted Sale Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# =====================================
# Step 13: Display Sample Predictions
# =====================================
comparison = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print("\n--- Actual vs Predicted Prices ---")
print(comparison.head(10))

# =====================================
# Step 14: Save the Trained Model
# =====================================
joblib.dump(model, "house_price_model.pkl")
print("\n💾 Model saved as 'house_price_model.pkl'")

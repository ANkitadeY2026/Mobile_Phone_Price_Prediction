import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load cleaned dataset
df = pd.read_csv(
    "dataset/mobile_cleaned_data.csv"
)


# Features
X = df[
    [
        "Brand",
        "RAM_GB",
        "Storage_GB",
        "Camera_MP",
        "Battery_mAh",
        "Screen_Size"
    ]
]


# Target
y = df["Price"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Categorical feature
categorical_features = ["Brand"]


# Numerical features
numeric_features = [
    "RAM_GB",
    "Storage_GB",
    "Camera_MP",
    "Battery_mAh",
    "Screen_Size"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "brand",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train model
pipeline.fit(X_train, y_train)


# Prediction
predictions = pipeline.predict(X_test)


# Evaluation
mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print("=" * 50)
print("MACHINE LEARNING MODEL")
print("=" * 50)

print("\nModel Training Completed!")

print("\nMean Absolute Error:")
print(round(mae, 2))

print("\nR2 Score:")
print(round(r2, 2))


# Save model
joblib.dump(
    pipeline,
    "model/phone_price_prediction_model.pkl"
)

print("\nModel saved successfully!")
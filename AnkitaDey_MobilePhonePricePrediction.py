# ============================================================
# Mobile Phone Price Analysis & Prediction Using Machine Learning
# Student Name: Ankita Dey
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

df = pd.read_csv("dataset/mobile_raw_data.csv")

print("=" * 60)
print("RAW DATASET")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 2. DATA CLEANING
# ============================================================

clean_df = df.copy()

# Remove extra spaces and standardize brand names
clean_df["Brand"] = clean_df["Brand"].str.strip()
clean_df["Brand"] = clean_df["Brand"].str.title()

# Fill missing numerical values with median
numeric_columns = [
    "RAM_GB",
    "Storage_GB",
    "Camera_MP",
    "Battery_mAh",
    "Screen_Size",
    "Price"
]

for column in numeric_columns:
    clean_df[column] = clean_df[column].fillna(
        clean_df[column].median()
    )

# Remove duplicate rows
clean_df = clean_df.drop_duplicates()

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)

print("\nMissing Values After Cleaning:")
print(clean_df.isnull().sum())

print("\nFinal Dataset Shape:")
print(clean_df.shape)

# Save cleaned dataset
clean_df.to_csv(
    "dataset/mobile_cleaned_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")


# ============================================================
# 3. DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DATA ANALYSIS")
print("=" * 60)

average_price = clean_df["Price"].mean()

print("\nAverage Mobile Phone Price:")
print(round(average_price, 2))

print("\nMinimum Phone Price:")
print(clean_df["Price"].min())

print("\nMaximum Phone Price:")
print(clean_df["Price"].max())


# Average price by brand
brand_price = (
    clean_df.groupby("Brand")["Price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Price by Brand:")
print(brand_price)


# Average price by RAM
ram_price = (
    clean_df.groupby("RAM_GB")["Price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Price by RAM:")
print(ram_price)


# Most expensive phone
expensive_phone = clean_df.loc[
    clean_df["Price"].idxmax()
]

print("\nMost Expensive Phone:")
print(expensive_phone)


# Cheapest phone
cheap_phone = clean_df.loc[
    clean_df["Price"].idxmin()
]

print("\nCheapest Phone:")
print(cheap_phone)


# ============================================================
# 4. DATA VISUALIZATION
# ============================================================

# Brand vs Price
plt.figure(figsize=(9, 5))

brand_price.plot(kind="bar")

plt.title("Average Mobile Phone Price by Brand")
plt.xlabel("Brand")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("brand_price_analysis.png")
plt.close()


# RAM vs Price
plt.figure(figsize=(8, 5))

plt.scatter(
    clean_df["RAM_GB"],
    clean_df["Price"]
)

plt.title("RAM vs Mobile Phone Price")
plt.xlabel("RAM (GB)")
plt.ylabel("Price")
plt.tight_layout()

plt.savefig("ram_price_analysis.png")
plt.close()


# Storage vs Price
plt.figure(figsize=(8, 5))

plt.scatter(
    clean_df["Storage_GB"],
    clean_df["Price"]
)

plt.title("Storage vs Mobile Phone Price")
plt.xlabel("Storage (GB)")
plt.ylabel("Price")
plt.tight_layout()

plt.savefig("storage_price_analysis.png")
plt.close()


print("\nVisualization completed successfully!")


# ============================================================
# 5. MACHINE LEARNING MODEL
# ============================================================

X = clean_df[
    [
        "Brand",
        "RAM_GB",
        "Storage_GB",
        "Camera_MP",
        "Battery_mAh",
        "Screen_Size"
    ]
]

y = clean_df["Price"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Categorical feature
categorical_features = [
    "Brand"
]


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


# Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Create complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train model
pipeline.fit(
    X_train,
    y_train
)


# Make predictions
predictions = pipeline.predict(
    X_test
)


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n" + "=" * 60)
print("MACHINE LEARNING MODEL")
print("=" * 60)

print("\nModel Training Completed!")

print("\nMean Absolute Error:")
print(round(mae, 2))

print("\nR2 Score:")
print(round(r2, 2))


# Save trained model
joblib.dump(
    pipeline,
    "model/phone_price_prediction_model.pkl"
)

print("\nModel saved successfully!")


# ============================================================
# 7. SAMPLE PRICE PREDICTION
# ============================================================

sample_phone = pd.DataFrame([
    {
        "Brand": "Samsung",
        "RAM_GB": 8,
        "Storage_GB": 128,
        "Camera_MP": 50,
        "Battery_mAh": 5000,
        "Screen_Size": 6.5
    }
])


sample_prediction = pipeline.predict(
    sample_phone
)[0]

print("\n" + "=" * 60)
print("SAMPLE PRICE PREDICTION")
print("=" * 60)

print("\nPhone Specifications:")
print(sample_phone)

print("\nPredicted Price:")
print("₹", round(sample_prediction, 2))


print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
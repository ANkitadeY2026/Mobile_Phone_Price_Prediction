import pandas as pd
import matplotlib.pyplot as plt

# Load raw dataset
df = pd.read_csv("dataset/mobile_raw_data.csv")

print("=" * 50)
print("RAW DATASET")
print("=" * 50)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# DATA CLEANING
# -----------------------------

clean_df = df.copy()

# Remove extra spaces from Brand
clean_df["Brand"] = clean_df["Brand"].str.strip()

# Standardize brand names
clean_df["Brand"] = clean_df["Brand"].str.title()

# Numerical columns
numeric_columns = [
    "RAM_GB",
    "Storage_GB",
    "Camera_MP",
    "Battery_mAh",
    "Screen_Size",
    "Price"
]

# Fill missing values using median
for column in numeric_columns:
    clean_df[column] = clean_df[column].fillna(
        clean_df[column].median()
    )

# Remove duplicate rows
clean_df = clean_df.drop_duplicates()

print("\n" + "=" * 50)
print("DATA CLEANING COMPLETED")
print("=" * 50)

print("\nMissing Values After Cleaning:")
print(clean_df.isnull().sum())

print("\nFinal Dataset Shape:")
print(clean_df.shape)

print("\nCleaned Data:")
print(clean_df.head())


# Save cleaned dataset
clean_df.to_csv(
    "dataset/mobile_cleaned_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
# -----------------------------
# DATA ANALYSIS
# -----------------------------

print("\n" + "=" * 50)
print("DATA ANALYSIS")
print("=" * 50)

# Average price
average_price = clean_df["Price"].mean()

print("\nAverage Mobile Phone Price:")
print(round(average_price, 2))


# Minimum price
print("\nMinimum Phone Price:")
print(clean_df["Price"].min())


# Maximum price
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


# -----------------------------
# GRAPH 1
# -----------------------------

plt.figure(figsize=(9, 5))

brand_price.plot(kind="bar")

plt.title("Average Mobile Phone Price by Brand")
plt.xlabel("Brand")
plt.ylabel("Average Price")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("brand_price_analysis.png")

plt.show()


# -----------------------------
# GRAPH 2
# -----------------------------

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

plt.show()


# -----------------------------
# GRAPH 3
# -----------------------------

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

plt.show()


print("\nData Analysis Completed Successfully!")
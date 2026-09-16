# Step 1: Read the data
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Separate the features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Check the training and testing shapes
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# you can save the boxplot...
plt.figure(figsize=(6, 4))
df["MedHouseVal"].plot.box()
plt.title("Boxplot of Median House Value")
plt.ylabel("Median House Value")
 
plt.tight_layout()
plt.savefig("figures/med_house_value_boxplot.png")  # <-- saved file
plt.close()

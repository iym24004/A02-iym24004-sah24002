# Step 1: Read the data

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

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


# Create the model pipeline

model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        random_state=42
    ))
])


# Train the model

model.fit(X_train, y_train)

print("Model training completed.")
print("Training iterations:", model.named_steps["mlp"].n_iter_)


# Generate predictions for the training data

train_predictions = model.predict(X_train)


# Create the actual vs predicted training plot

plt.figure(figsize=(6, 6))

plt.scatter(
    y_train,
    train_predictions,
    alpha=0.3,
    color="blue"
)

plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs. Predicted House Values - Training Data")

plt.tight_layout()
plt.savefig("figures/train_actual_vs_pred.png")
plt.close()

print("Training prediction plot saved.")


# Save the boxplot

plt.figure(figsize=(6, 4))
df["MedHouseVal"].plot.box()

plt.title("Boxplot of Median House Value")
plt.ylabel("Median House Value")

plt.tight_layout()
plt.savefig("figures/med_house_value_boxplot.png")
plt.close()

print("Boxplot saved.")
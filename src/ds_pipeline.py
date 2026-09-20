# Step 1: Read the data

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

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


# Calculate training evaluation metrics

train_rmse = mean_squared_error(y_train, train_predictions) ** 0.5
train_r2 = r2_score(y_train, train_predictions)

print("Training RMSE:", round(train_rmse, 3))
print("Training R-squared:", round(train_r2, 3))


# Create the actual vs predicted training plot

plt.figure(figsize=(6, 6))

plt.scatter(
    y_train,
    train_predictions,
    alpha=0.3,
    color="blue",
    label="Predictions"
)

train_min = min(y_train.min(), train_predictions.min())
train_max = max(y_train.max(), train_predictions.max())

plt.plot(
    [train_min, train_max],
    [train_min, train_max],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title(
    f"Training Data: Actual vs. Predicted\n"
    f"RMSE = {train_rmse:.3f}, R² = {train_r2:.3f}"
)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figures/train_actual_vs_pred.png")
plt.close()

print("Improved training prediction plot saved.")


# Generate predictions for the testing data

test_predictions = model.predict(X_test)


# Calculate testing evaluation metrics

test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
test_r2 = r2_score(y_test, test_predictions)

print("Testing RMSE:", round(test_rmse, 3))
print("Testing R-squared:", round(test_r2, 3))


# Create the actual vs predicted testing plot

plt.figure(figsize=(6, 6))

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.3,
    color="green",
    label="Predictions"
)

test_min = min(y_test.min(), test_predictions.min())
test_max = max(y_test.max(), test_predictions.max())

plt.plot(
    [test_min, test_max],
    [test_min, test_max],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title(
    f"Testing Data: Actual vs. Predicted\n"
    f"RMSE = {test_rmse:.3f}, R² = {test_r2:.3f}"
)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figures/test_actual_vs_pred.png")
plt.close()

print("Improved testing prediction plot saved.")


# Save the boxplot

plt.figure(figsize=(6, 4))
df["MedHouseVal"].plot.box()

plt.title("Boxplot of Median House Value")
plt.ylabel("Median House Value")

plt.tight_layout()
plt.savefig("figures/med_house_value_boxplot.png")
plt.close()

print("Boxplot saved.")
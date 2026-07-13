import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
        # Load Dataset
# ==========================================
def load_data(path):
    return pd.read_csv(path)


# ==========================================
      # Data Preprocessing
# ==========================================
def preprocess_data(df):

    print("\nDataset Information")
    print(df.info())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDataset Description")
    print(df.describe())

    # One-Hot Encoding
    df = pd.get_dummies(df, drop_first=True)

    return df


# ==========================================
             # Split Dataset
# ==========================================
def split_dataset(df):

    X = df.drop("charges", axis=1)
    y = df["charges"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# ==========================================
         # Feature Scaling
# ==========================================
def scale_features(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# ==========================================
       # Linear Regression Model
# ==========================================
def train_linear_regression(X_train, X_test, y_train, y_test):

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n========== Linear Regression ==========")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return predictions


# ==========================================
          # KNN Regression Model
# ==========================================
def train_knn(X_train, X_test, y_train, y_test):

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n========== KNN Regression ==========")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return predictions


# ==========================================
          # Visualization
# ==========================================
def plot_results(actual, linear_predictions, knn_predictions):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        actual,
        linear_predictions,
        alpha=0.6,
        label="Linear Regression"
    )

    plt.scatter(
        actual,
        knn_predictions,
        alpha=0.6,
        label="KNN Regression"
    )

    plt.plot(
        [actual.min(), actual.max()],
        [actual.min(), actual.max()],
        "k--",
        linewidth=2
    )

    plt.title("Actual vs Predicted Healthcare Charges")
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.legend()

    # Save figure
    plt.savefig("../results/actual_vs_predicted.png", dpi=300)

    plt.show()

# ==========================================
             # Main Function
# ==========================================
def main():

    print("Loading Dataset...\n")

    df = load_data("../data/insurance.csv")

    print("Dataset Shape:", df.shape)

    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_dataset(df)

    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test
    )

    linear_predictions = train_linear_regression(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    knn_predictions = train_knn(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    plot_results(
        y_test,
        linear_predictions,
        knn_predictions
    )


if __name__ == "__main__":
    main()
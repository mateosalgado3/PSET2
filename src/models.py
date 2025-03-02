import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

#  Cargar dataset procesado
DATA_PATH = "/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/Airbnb_Featured.csv"

df = pd.read_csv(DATA_PATH)

#  Separar en X (features) e y (target)
X = df.drop(columns=["log_price"])  # Variables independientes
y = df["log_price"]  # Variable dependiente

#  Dividir en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dimensiones de X_train: {X_train.shape}")
print(f"Dimensiones de X_test: {X_test.shape}")

#  Guardar datasets para modelado
X_train.to_csv("/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/X_train.csv", index=False)
X_test.to_csv("/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/X_test.csv", index=False)
y_train.to_csv("/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/y_train.csv", index=False)
y_test.to_csv("/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/y_test.csv", index=False)


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from numpy.linalg import inv, svd

#  Cargar datasets
X_train = pd.read_csv("/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/X_train.csv")
X_test = pd.read_csv("/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/X_test.csv")
y_train = pd.read_csv("/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/y_test.csv").values.ravel()

#  Función para calcular métricas de evaluación
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        "MSE_train": mean_squared_error(y_train, y_train_pred),
        "MSE_test": mean_squared_error(y_test, y_test_pred),
        "RMSE_train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "MAE_train": mean_absolute_error(y_train, y_train_pred),
        "MAE_test": mean_absolute_error(y_test, y_test_pred),
        "R2_train": r2_score(y_train, y_train_pred),
        "R2_test": r2_score(y_test, y_test_pred),
    }
    
    return metrics


###  1. Regresión Lineal con Ecuación Normal (Implementación Propia)
def normal_equation(X, y):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Agregar bias
    theta = inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)  # Fórmula de ecuación normal
    return theta

theta_normal = normal_equation(X_train, y_train)
y_pred_train = np.c_[np.ones((X_train.shape[0], 1)), X_train].dot(theta_normal)
y_pred_test = np.c_[np.ones((X_test.shape[0], 1)), X_test].dot(theta_normal)

metrics_normal = {
    "MSE_train": mean_squared_error(y_train, y_pred_train),
    "MSE_test": mean_squared_error(y_test, y_pred_test),
    "RMSE_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
    "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    "MAE_train": mean_absolute_error(y_train, y_pred_train),
    "MAE_test": mean_absolute_error(y_test, y_pred_test),
    "R2_train": r2_score(y_train, y_pred_train),
    "R2_test": r2_score(y_test, y_pred_test),
}

print(" Evaluación Regresión Lineal (Ecuación Normal)", metrics_normal)


###  2. Regresión Lineal con Scikit-Learn
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
metrics_linear = evaluate_model(lin_reg, X_train, y_train, X_test, y_test)
print(" Evaluación Regresión Lineal (sklearn)", metrics_linear)


###  3. Regresión Polinomial
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
metrics_poly = evaluate_model(poly_reg, X_train_poly, y_train, X_test_poly, y_test)
print(" Evaluación Regresión Polinomial", metrics_poly)


### 4. Ridge Regression
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X_train, y_train)
metrics_ridge = evaluate_model(ridge_reg, X_train, y_train, X_test, y_test)
print(" Evaluación Ridge Regression", metrics_ridge)


###  5. Lasso Regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
metrics_lasso = evaluate_model(lasso_reg, X_train, y_train, X_test, y_test)
print(" Evaluación Lasso Regression", metrics_lasso)


###  6. Regresión con SVD (Descomposición en Valores Singulares)
U, S, Vt = svd(X_train, full_matrices=False)
theta_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y_train
y_pred_train_svd = X_train @ theta_svd
y_pred_test_svd = X_test @ theta_svd

metrics_svd = evaluate_model(lin_reg, X_train, y_train, X_test, y_test)
print(" Evaluación Regresión SVD", metrics_svd)


# Función para guardar modelos
def save_model(model, filename):
    path = f"models/{filename}.pkl"
    joblib.dump(model, path)
    print(f"Modelo guardado: {path}")

# Guardar cada modelo entrenado
save_model(lin_reg, "linear_regression")
save_model(ridge_reg, "ridge_regression")
save_model(lasso_reg, "lasso_regression")
save_model(poly_reg, "polynomial_regression")
import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV y devuelve un DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

def check_missing_values(df: pd.DataFrame):
    """
    Muestra el porcentaje de valores faltantes en cada columna.
    """
    missing_percent = df.isnull().sum() / len(df) * 100
    return missing_percent[missing_percent > 0].sort_values(ascending=False)

def drop_irrelevant_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Elimina columnas irrelevantes o innecesarias del DataFrame.
    """
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df

def detect_outliers(df: pd.DataFrame, column: str, threshold: float = 1.5):
    """
    Identifica valores atípicos en la columna dada usando el método IQR.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def clean_dataset(filepath: str, output_filepath: str, drop_cols: list):
    """
    Función principal que carga, limpia y guarda el dataset procesado.
    """
    df = load_data(filepath)
    print("Valores faltantes antes de la limpieza:")
    print(check_missing_values(df))

    df = drop_irrelevant_columns(df, drop_cols)
    
    # Identificar y remover outliers en la columna 'price'
    outliers = detect_outliers(df, 'log_price')
    print(f"Cantidad de outliers detectados en 'price': {len(outliers)}")
    
    # Guardar dataset limpio
    df.to_csv(output_filepath, index=False)
    print(f"Dataset limpio guardado en: {output_filepath}")

if __name__ == "__main__":
    RAW_PATH = "/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/raw/Airbnb_Data.csv"
    PROCESSED_PATH = "/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/Airbnb_Cleaned.csv"
    DROP_COLUMNS = [
    "id", "name", "description", "thumbnail_url", "first_review", "last_review",
    "host_since", "host_response_rate", "host_has_profile_pic", "host_identity_verified",
    "neighbourhood", "zipcode"
    ]

    clean_dataset(RAW_PATH, PROCESSED_PATH, DROP_COLUMNS)

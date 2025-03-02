import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def count_amenities(amenities_str):
    """Corrige la extracción de amenities y cuenta el total."""
    try:
        amenities_list = ast.literal_eval(amenities_str)  # Convierte string a lista
        return len(amenities_list)
    except:
        return 0

def preprocess_features(filepath: str, output_filepath: str):
    """
    Función principal para la creación de variables, codificación y escalado.
    """
    df = pd.read_csv(filepath)

    print(" Creando variables derivadas...")

    #  **Crear Variable: Total de Amenidades**
    df["total_amenities"] = df["amenities"].apply(count_amenities)

    #  **Imputación de valores bajos en `total_amenities`**
    df["total_amenities"] = df["total_amenities"].replace(0, df["total_amenities"].median())

    #  **Codificación de Variables Categóricas**
    print(" Codificando variables categóricas...")

    # Label Encoding para 'property_type' (ya que hay muchas categorías)
    le = LabelEncoder()
    df["property_type_encoded"] = le.fit_transform(df["property_type"])

    # Convertir 'instant_bookable' y 'cleaning_fee' a valores 0/1
    df["instant_bookable"] = df["instant_bookable"].map({"t": 1, "f": 0})
    df["cleaning_fee"] = df["cleaning_fee"].astype(int)

    # **Manejo de valores faltantes SOLO en variables numéricas**
    print(" Imputando valores faltantes...")
    num_cols = ["accommodates", "bathrooms", "bedrooms", "beds", "number_of_reviews", "review_scores_rating", "total_amenities"]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())  # Solo columnas numéricas

    # **Escalado de Variables Numéricas**
    print(" Escalando variables numéricas...")
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    #  **Selección de Variables con Mayor Correlación**
    print(" Seleccionando variables más relevantes...")
    selected_features = ["log_price", "accommodates", "bathrooms", "bedrooms", "beds", "number_of_reviews", "review_scores_rating", "property_type_encoded", "total_amenities"]
    df = df[selected_features]

    #  **Guardar Dataset Procesado**
    df.to_csv(output_filepath, index=False)
    print(f"-/ Dataset FINAL guardado sin NaN en: {output_filepath}")

if __name__ == "__main__":
    CLEANED_PATH = "/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/Airbnb_Cleaned.csv"
    FINAL_PATH = "/mnt/c/Users/pmate/OneDrive - Universidad San Francisco de Quito/8 SEMESTRE/DataMining/Deberes/PSet2/data/processed/Airbnb_Featured.csv"
    
    preprocess_features(CLEANED_PATH, FINAL_PATH)

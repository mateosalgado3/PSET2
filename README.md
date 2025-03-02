Predicción de Precios de Airbnb
Este proyecto tiene como objetivo predecir los precios de propiedades en Airbnb utilizando modelos de Regresión Lineal y sus variantes. A través de Data Wrangling, Feature Engineering, Modelado y Evaluación, se entrenaron diferentes modelos y se compararon sus desempeños.

📂 Estructura del Proyecto
bash
Copiar
Editar
PSet2/
│── data/
│   ├── raw/                # Datos originales sin procesar
│   ├── processed/          # Datos después de limpieza y Feature Engineering
│── models/                 # Modelos entrenados guardados (.pkl)
│── notebooks/              # Notebooks con análisis, visualización y evaluación
│── src/                    # Código fuente del proyecto
│   ├── data_wrangling.py   # Limpieza y preprocesamiento de datos
│   ├── feature_engineering.py  # Creación de variables y preparación de datos
│   ├── models.py           # Entrenamiento y guardado de modelos
│── README.md               # Documentación del proyecto
│── requirements.txt        # Librerías necesarias
│── .gitignore              # Archivos a excluir
🛠 Requisitos
Para ejecutar este proyecto, necesitas:

Python 3.10+
WSL (si ejecutas en Windows)
Librerías de Python (instalables con requirements.txt)
Instala todas las dependencias con:

bash
Copiar
Editar
pip install -r requirements.txt
🚀 Cómo ejecutar el proyecto
1️⃣ Preparar los Datos
Coloca el archivo de datos original en data/raw/ y ejecuta:

bash
Copiar
Editar
python src/data_wrangling.py
Esto genera el dataset limpio en data/processed/Airbnb_Cleaned.csv.

2️⃣ Feature Engineering
Para generar nuevas variables y preparar los datos para el modelado, ejecuta:

bash
Copiar
Editar
python src/feature_engineering.py
Esto generará Airbnb_Featured.csv en data/processed/.

3️⃣ Entrenamiento de Modelos
Ejecuta:

bash
Copiar
Editar
python src/models.py
Esto entrenará los siguientes modelos y guardará los archivos .pkl en models/:

Regresión Lineal (Ecuación Normal)
Regresión Lineal con Scikit-learn
Regresión Polinomial
Ridge Regression
Lasso Regression
SGD Batch Gradient Descent
SGD Stochastic Gradient Descent
Los modelos se guardarán en:

Copiar
Editar
models/
│── linear_regression.pkl
│── ridge_regression.pkl
│── lasso_regression.pkl
│── polynomial_regression.pkl
│── sgd_batch.pkl
│── sgd_stochastic.pkl
4️⃣ Evaluación de Modelos
Los modelos guardados fueron evaluados utilizando los datos de prueba (X_test y y_test), comparando métricas como MSE, RMSE, MAE y R².

Para visualizar los resultados, abre el notebook:

bash
Copiar
Editar
notebooks/4_evaluation_and_results.ipynb
Este notebook carga los modelos desde models/ y genera comparaciones gráficas.

📊 Resultados y Conclusiones
Regresión Polinomial tuvo el mejor desempeño con el RMSE más bajo y el mejor R², lo que indica que captura relaciones no lineales.
SGD Stochastic Gradient Descent tuvo el peor desempeño, posiblemente debido a falta de convergencia o mala selección de hiperparámetros.
Ridge y Lasso Regression ayudaron a controlar la complejidad y reducir el sobreajuste.
Para mejorar, podríamos probar modelos más avanzados como árboles de decisión o redes neuronales, o mejorar la ingeniería de características.
📌 Notas y Recomendaciones
Asegúrate de que los archivos de datos están en las rutas correctas.
Si usas Windows, ejecuta en WSL para evitar problemas con rutas.
Si ves errores de features missing, verifica que X_train y X_test coincidan en número de columnas con los datos con los que fueron entrenados los modelos.
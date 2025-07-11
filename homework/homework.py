#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import os
import json
import gzip
import zipfile
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


def transformar_datos(df):
    df = df.copy()
    df["Age"] = 2021 - df["Year"]
    return df.drop(columns=["Year", "Car_Name"])


def dividir_entrada_salida(df):
    y = df["Present_Price"]
    X = df.drop(columns=["Present_Price"])
    return X, y


def definir_pipeline():
    categorias = ["Fuel_Type", "Selling_type", "Transmission"]
    numericos = ["Selling_Price", "Driven_kms", "Owner", "Age"]

    transformador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorias),
            ("num", MinMaxScaler(), numericos)
        ]
    )

    modelo = Pipeline(steps=[
        ("transformador", transformador),
        ("selector", SelectKBest(score_func=f_regression)),
        ("regresor", LinearRegression())
    ])

    return modelo


def buscar_hiperparametros(modelo, X, y):
    espacio_parametros = {
        "selector__k": [4, 5, 6, 7, 8, 9, 10, 11],
        "regresor__fit_intercept": [True, False]
    }

    optimizador = GridSearchCV(
        estimator=modelo,
        param_grid=espacio_parametros,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    optimizador.fit(X, y)
    return optimizador


def guardar_modelo_gzip(modelo, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with gzip.open(ruta, "wb") as f:
        pd.to_pickle(modelo, f)


def evaluar_modelo(modelo, X_train, y_train, X_test, y_test):
    resultado = []

    for X, y, grupo in [(X_train, y_train, "train"), (X_test, y_test, "test")]:
        y_pred = modelo.predict(X)
        resultado.append({
            "type": "metrics",
            "dataset": grupo,
            "r2": r2_score(y, y_pred),
            "mse": mean_squared_error(y, y_pred),
            "mad": median_absolute_error(y, y_pred)
        })

    return resultado


def guardar_metricas_json(lista_metricas, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as f:
        for entrada in lista_metricas:
            f.write(json.dumps(entrada) + "\n")


def cargar_zip_y_leer_csv(ruta_zip):
    with zipfile.ZipFile(ruta_zip, "r") as zf:
        archivo_csv = zf.namelist()[0]
        with zf.open(archivo_csv) as f:
            return pd.read_csv(f)


def ejecutar_entrenamiento():
    train_df = cargar_zip_y_leer_csv("files/input/train_data.csv.zip")
    test_df = cargar_zip_y_leer_csv("files/input/test_data.csv.zip")

    train_df = transformar_datos(train_df)
    test_df = transformar_datos(test_df)

    X_train, y_train = dividir_entrada_salida(train_df)
    X_test, y_test = dividir_entrada_salida(test_df)

    modelo = definir_pipeline()
    modelo_ajustado = buscar_hiperparametros(modelo, X_train, y_train)

    guardar_modelo_gzip(modelo_ajustado, "files/models/model.pkl.gz")
    metricas = evaluar_modelo(modelo_ajustado, X_train, y_train, X_test, y_test)
    guardar_metricas_json(metricas, "files/output/metrics.json")


if __name__ == "__main__":
    ejecutar_entrenamiento()
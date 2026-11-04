"""Funciones compartidas para el proyecto Adult Income."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "adult_income.csv"
RANDOM_STATE = 42


def preparar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los nombres, faltantes y etiqueta del dataset UCI Adult."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    objetivo = next(
        (col for col in df.columns if col.lower() in {"income", "income_raw", "class"}),
        None,
    )
    if objetivo is None:
        raise ValueError("No se encontró la columna de ingresos (income) en el dataset.")

    if objetivo != "income_raw":
        df = df.rename(columns={objetivo: "income_raw"})
    df["income_raw"] = df["income_raw"].astype("string").str.strip().str.rstrip(".")
    df["objetivo"] = df["income_raw"].str.contains(">50K", regex=False, na=False).astype("int8")

    # UCI representa algunos datos ausentes con el texto "?".
    for col in df.select_dtypes(include=["object", "string"]).columns:
        valores = df[col].astype("object")
        df[col] = valores.map(
            lambda value: np.nan if pd.isna(value) or str(value).strip() == "?" else str(value).strip()
        )
    return df


def cargar_datos(ruta: str | Path = DATA_PATH) -> pd.DataFrame:
    """Carga la copia local del dataset; no realiza conexiones de red implícitas."""
    ruta = Path(ruta)
    if not ruta.is_absolute():
        ruta = ROOT_DIR / ruta
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Descarga primero el dataset con "
            "python data/descargar_datos.py."
        )
    return preparar_dataset(pd.read_csv(ruta, na_values=["?"]))


def guardar_datos(df: pd.DataFrame, ruta: str | Path = DATA_PATH) -> Path:
    """Guarda el dataset en una ruta estable dentro del proyecto."""
    ruta = Path(ruta)
    if not ruta.is_absolute():
        ruta = ROOT_DIR / ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    preparar_dataset(df).to_csv(ruta, index=False)
    return ruta


def obtener_features_y_objetivo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa predictores y etiqueta; excluye identificador de peso fnlwgt."""
    if "objetivo" not in df.columns:
        df = preparar_dataset(df)
    excluir = {"objetivo", "income_raw", "fnlwgt"}
    features = [col for col in df.columns if col not in excluir]
    return df[features].copy(), df["objetivo"].astype("int8").copy()


def identificar_columnas(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retorna nombres de columnas numéricas y categóricas."""
    numericas = X.select_dtypes(include=["number"]).columns.tolist()
    categoricas = [col for col in X.columns if col not in numericas]
    return numericas, categoricas


def construir_estimador(tipo: str = "logistica"):
    """Construye el clasificador solicitado, con semilla fija."""
    if tipo == "logistica":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    if tipo == "arbol":
        return DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE, class_weight="balanced")
    if tipo == "dummy":
        return DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    raise ValueError(f"Tipo de modelo no reconocido: {tipo}")


def construir_pipeline_completo(
    numericas: Iterable[str], categoricas: Iterable[str], tipo: str = "logistica"
) -> Pipeline:
    """Crea un pipeline que ajusta imputación, codificación y escala solo en train."""
    transformador_numerico = Pipeline(
        steps=[("imputador", SimpleImputer(strategy="median")), ("escalador", StandardScaler())]
    )
    transformador_categorico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("codificador", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocesador = ColumnTransformer(
        transformers=[
            ("numericas", transformador_numerico, list(numericas)),
            ("categoricas", transformador_categorico, list(categoricas)),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[("preprocesamiento", preprocesador), ("modelo", construir_estimador(tipo))]
    )


def calcular_metricas(y_true, y_pred, nombre_modelo: str = "Modelo") -> dict[str, float | str]:
    """Calcula métricas de clasificación para la clase positiva >50K."""
    return {
        "Modelo": nombre_modelo,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def tabla_comparativa(lista_metricas: list[dict]) -> pd.DataFrame:
    """Construye una tabla indexada por el nombre del modelo."""
    return pd.DataFrame(lista_metricas).set_index("Modelo")


def graficar_confusion(y_true, y_pred, nombre_modelo="Modelo", ruta_guardado=None):
    """Dibuja una matriz con orden de etiquetas fijo: <=50K y >50K."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    etiquetas = ["<=50K (negativo)", ">50K (positivo)"]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=etiquetas,
        yticklabels=etiquetas,
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"Matriz de confusión - {nombre_modelo}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    fig.tight_layout()
    if ruta_guardado:
        destino = Path(ruta_guardado)
        if not destino.is_absolute():
            destino = ROOT_DIR / destino
        destino.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino, dpi=160, bbox_inches="tight")
    return fig, cm


def crear_grupo_edad(edad: float) -> str:
    """Agrupa edades para la auditoría descriptiva de equidad."""
    if edad < 30:
        return "18-29"
    if edad < 40:
        return "30-39"
    if edad < 50:
        return "40-49"
    if edad < 60:
        return "50-59"
    return "60+"


def metricas_por_grupo(df_test, y_true, y_pred, columna_grupo: str) -> pd.DataFrame:
    """Calcula tamaño, recall y tasas positivas por cada grupo demográfico."""
    eval_df = pd.DataFrame(
        {
            "grupo": df_test[columna_grupo].reset_index(drop=True),
            "real": pd.Series(y_true).reset_index(drop=True),
            "predicho": np.asarray(y_pred),
        }
    )
    resultados = []
    for grupo, datos in eval_df.groupby("grupo", dropna=False, observed=True):
        positivos_reales = int(datos["real"].sum())
        resultados.append(
            {
                "Grupo": "Sin dato" if pd.isna(grupo) else str(grupo),
                "N": len(datos),
                "Tasa_Positivos_Real": round(float(datos["real"].mean()), 4),
                "Tasa_Positivos_Predicha": round(float(datos["predicho"].mean()), 4),
                "Recall": round(
                    recall_score(datos["real"], datos["predicho"], zero_division=0), 4
                ),
                "Precision": round(
                    precision_score(datos["real"], datos["predicho"], zero_division=0), 4
                ),
                "F1-Score": round(f1_score(datos["real"], datos["predicho"], zero_division=0), 4),
                "Positivos_Reales": positivos_reales,
            }
        )
    return pd.DataFrame(resultados)


def diferencia_max_min(tabla: pd.DataFrame, metrica: str) -> float:
    """Calcula max-min de una métrica grupal disponible."""
    valores = pd.to_numeric(tabla[metrica], errors="coerce").dropna()
    return round(float(valores.max() - valores.min()), 4) if len(valores) else float("nan")

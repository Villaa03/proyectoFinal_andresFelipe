"""
Módulo de utilidades para el proyecto Adult Income ML.
Contiene funciones de carga, preprocesamiento, entrenamiento y evaluación.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_datos():
    """
    Descarga el dataset Adult Income desde UCI ML Repository.
    Retorna un DataFrame limpio con columnas nombradas.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        adult = fetch_ucirepo(id=2)
        X = adult.data.features
        y = adult.data.targets
        df = pd.concat([X, y], axis=1)
        # Normalizar nombre de la columna objetivo
        if 'income' in df.columns:
            df = df.rename(columns={'income': 'income_raw'})
        elif 'Income' in df.columns:
            df = df.rename(columns={'Income': 'income_raw'})
        else:
            # Última columna es el objetivo
            cols = df.columns.tolist()
            df = df.rename(columns={cols[-1]: 'income_raw'})
        # Crear columna binaria: 1 si >50K, 0 si <=50K
        df['objetivo'] = df['income_raw'].apply(
            lambda x: 1 if '>50K' in str(x) else 0
        )
        return df
    except Exception as e:
        print(f"Error cargando datos desde UCI: {e}")
        print("Intentando carga alternativa desde URL directa...")
        return _cargar_datos_url()


def _cargar_datos_url():
    """Carga alternativa del dataset desde URL directa."""
    columnas = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income'
    ]
    url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    df_train = pd.read_csv(url_train, names=columnas, sep=', ', engine='python',
                           skipinitialspace=True, na_values='?')
    df_test = pd.read_csv(url_test, names=columnas, sep=', ', engine='python',
                          skipinitialspace=True, na_values='?', skiprows=1)
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.rename(columns={'income': 'income_raw'})
    df['objetivo'] = df['income_raw'].apply(
        lambda x: 1 if '>50K' in str(x) else 0
    )
    return df


def guardar_datos(df, ruta='data/adult_income.csv'):
    """Guarda el DataFrame procesado en disco."""
    df.to_csv(ruta, index=False)
    print(f"Dataset guardado en: {ruta}")


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

def obtener_features_y_objetivo(df):
    """
    Separa las features del objetivo.
    Retorna X (features), y (objetivo).
    """
    columnas_excluir = ['objetivo', 'income_raw', 'fnlwgt']
    columnas_features = [c for c in df.columns if c not in columnas_excluir]
    X = df[columnas_features].copy()
    y = df['objetivo'].copy()
    return X, y


def identificar_columnas(X):
    """Identifica columnas numéricas y categóricas."""
    numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()
    return numericas, categoricas


def construir_pipeline(tipo='logistica'):
    """
    Construye un Pipeline de scikit-learn con preprocesamiento y modelo.

    Parámetros
    ----------
    tipo : str
        'logistica' para Regresión Logística
        'arbol' para Árbol de Decisión
        'dummy' para DummyClassifier (baseline)

    Retorna
    -------
    modelo : estimador sklearn
    """
    if tipo == 'logistica':
        modelo = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    elif tipo == 'arbol':
        modelo = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
    elif tipo == 'dummy':
        modelo = DummyClassifier(strategy='most_frequent', random_state=42)
    else:
        raise ValueError(f"Tipo de modelo no reconocido: {tipo}")

    return modelo


def construir_pipeline_completo(numericas, categoricas, tipo='logistica'):
    """
    Construye el Pipeline completo con preprocesamiento y modelo especificado.

    Parámetros
    ----------
    numericas : list de str
        Nombres de columnas numéricas.
    categoricas : list de str
        Nombres de columnas categóricas.
    tipo : str
        Tipo de modelo: 'logistica', 'arbol', 'dummy'.

    Retorna
    -------
    Pipeline : sklearn.pipeline.Pipeline
    """
    transformador_numerico = Pipeline(steps=[
        ('escalador', StandardScaler())
    ])
    transformador_categorico = Pipeline(steps=[
        ('codificador', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocesador = ColumnTransformer(
        transformers=[
            ('num', transformador_numerico, numericas),
            ('cat', transformador_categorico, categoricas)
        ],
        remainder='drop'
    )

    if tipo == 'logistica':
        modelo = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    elif tipo == 'arbol':
        modelo = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
    elif tipo == 'dummy':
        modelo = DummyClassifier(strategy='most_frequent', random_state=42)
    else:
        raise ValueError(f"Tipo de modelo no reconocido: {tipo}")

    pipeline = Pipeline(steps=[
        ('preprocesador', preprocesador),
        ('modelo', modelo)
    ])

    return pipeline


# =============================================================================
# EVALUACIÓN
# =============================================================================

def calcular_metricas(y_true, y_pred, nombre_modelo='Modelo'):
    """
    Calcula Accuracy, Precision, Recall y F1-Score.

    Retorna
    -------
    dict con las métricas.
    """
    metricas = {
        'Modelo': nombre_modelo,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1-Score': round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    return metricas


def tabla_comparativa(lista_metricas):
    """
    Genera un DataFrame comparativo de métricas para varios modelos.

    Parámetros
    ----------
    lista_metricas : list de dict
        Lista de diccionarios retornados por calcular_metricas().

    Retorna
    -------
    pd.DataFrame con las métricas comparadas.
    """
    df = pd.DataFrame(lista_metricas)
    df = df.set_index('Modelo')
    return df


def graficar_confusion(y_true, y_pred, nombre_modelo='Modelo', ruta_guardado=None):
    """
    Genera y opcionalmente guarda la matriz de confusión.

    Parámetros
    ----------
    y_true : array-like
    y_pred : array-like
    nombre_modelo : str
    ruta_guardado : str o None
    """
    cm = confusion_matrix(y_true, y_pred)
    etiquetas = ['≤50K (Neg)', '>50K (Pos)']

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=etiquetas, yticklabels=etiquetas,
        ax=ax
    )
    ax.set_title(f'Matriz de Confusión — {nombre_modelo}', fontsize=14, pad=15)
    ax.set_ylabel('Etiqueta Real', fontsize=12)
    ax.set_xlabel('Etiqueta Predicha', fontsize=12)

    # Anotar cuadrantes
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                tipo = 'VN'
            elif i == 0 and j == 1:
                tipo = 'FP'
            elif i == 1 and j == 0:
                tipo = 'FN'
            else:
                tipo = 'VP'
            ax.text(j + 0.5, i + 0.75, tipo, ha='center', va='center',
                    fontsize=9, color='gray')

    plt.tight_layout()
    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=150, bbox_inches='tight')
        print(f"Matriz guardada en: {ruta_guardado}")
    return fig, cm


# =============================================================================
# ANÁLISIS ÉTICO
# =============================================================================

def metricas_por_grupo(df_test, y_true, y_pred, columna_grupo):
    """
    Calcula recall y tasa de positivos predichos por grupo demográfico.

    Parámetros
    ----------
    df_test : pd.DataFrame con los datos de prueba (con columna de grupo)
    y_true : pd.Series con etiquetas reales
    y_pred : array con predicciones
    columna_grupo : str nombre de la columna de agrupación

    Retorna
    -------
    pd.DataFrame con métricas por grupo.
    """
    df_eval = df_test[[columna_grupo]].copy().reset_index(drop=True)
    df_eval['y_true'] = y_true.reset_index(drop=True)
    df_eval['y_pred'] = y_pred

    resultados = []
    for grupo, grp_df in df_eval.groupby(columna_grupo):
        yt = grp_df['y_true']
        yp = grp_df['y_pred']
        n = len(yt)
        tasa_pos_real = yt.mean()
        tasa_pos_pred = yp.mean()
        rec = recall_score(yt, yp, zero_division=0)
        prec = precision_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        resultados.append({
            'Grupo': grupo,
            'N': n,
            'Tasa_Positivos_Real': round(tasa_pos_real, 4),
            'Tasa_Positivos_Predicha': round(tasa_pos_pred, 4),
            'Recall': round(rec, 4),
            'Precision': round(prec, 4),
            'F1-Score': round(f1, 4),
        })

    return pd.DataFrame(resultados)


def crear_grupo_edad(edad):
    """Clasifica la edad en rangos para análisis ético."""
    if edad < 30:
        return '18-29'
    elif edad < 40:
        return '30-39'
    elif edad < 50:
        return '40-49'
    elif edad < 60:
        return '50-59'
    else:
        return '60+'

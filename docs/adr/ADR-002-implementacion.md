# ADR-002: Enfoque de implementación y preprocesamiento

- **Estado:** Aceptado
- **Fecha:** 2026-11-03
- **Responsable:** Andrés Felipe Villanueva Parra

## Contexto

El proyecto necesita mostrar exploración, preparación, entrenamiento y evaluación de un dataset tabular. Hay faltantes y categorías, y las transformaciones no deben aprenderse con el conjunto de prueba.

## Decisión

Implementar en Python 3 con pandas, scikit-learn y Jupyter. Mantener la lógica reutilizable en `src/utils.py` y el recorrido académico en tres notebooks ordenados. Un script descarga Adult desde UCI y crea `data/adult_income.csv`; el modelado lee la copia local para evitar conexiones implícitas.

El pipeline imputa valores numéricos con mediana, imputa categorías con moda, escala variables numéricas y aplica one-hot encoding con categorías desconocidas toleradas. Cada paso se ajusta junto con el estimador solo durante `fit` sobre train.

## Alternativas consideradas

- Transformaciones manuales fuera del pipeline: simples de mostrar, pero propensas a fuga de información y a diferencias entre train y test.
- Guardar el dataset en Git: facilitaría una ejecución sin red, pero versionaría una copia de casi 49.000 filas que puede descargarse de la fuente.
- Descargar los datos desde cada notebook: menos archivos iniciales, pero acopla análisis y conectividad y dificulta reproducir ejecuciones previas.

## Consecuencias

- La descarga puede ejecutarse por separado; los notebooks dependen de que exista el CSV local.
- Los resultados son reproducibles con la semilla fija y las dependencias declaradas, sujetos a diferencias de versión o plataforma.
- La imputación se mantiene dentro del pipeline y se aplica tanto a entrenamiento como a test usando estadísticos de train.
- Las salidas y tablas se regeneran con los notebooks; se puede rastrear el resultado mostrado en el informe.

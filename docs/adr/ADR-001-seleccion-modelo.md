# ADR-001: Selección del modelo principal

- **Estado:** Aceptado
- **Fecha:** 2026-11-03
- **Responsable:** Andrés Felipe Villanueva Parra

## Contexto

El proyecto requiere clasificar la etiqueta binaria de ingresos del dataset Adult. Los predictores mezclan variables numéricas y categóricas; la clase `>50K` representa cerca del 24% de los registros. La actividad universitaria exige una implementación básica, reproducible e interpretable y una comparación con un baseline.

## Decisión

Usar **Regresión Logística** como modelo principal, con `max_iter=1000`, `random_state=42` y `class_weight="balanced"`. Comparar con un `DummyClassifier(strategy="most_frequent")` y un Árbol de Decisión de profundidad máxima 5.

El preprocesamiento vive en un pipeline de scikit-learn y se ajusta únicamente con el conjunto de entrenamiento. La auditoría reporta accuracy, precision, recall y F1; la elección no implica que el modelo sea adecuado para decisiones de alto impacto.

## Alternativas consideradas

| Alternativa | Ventaja | Desventaja para esta actividad |
|---|---|---|
| Árbol de Decisión como principal | Representa reglas no lineales y es explicable. | Puede variar con la muestra y sobreajustarse; se conserva como comparación acotada. |
| Random Forest | Captura relaciones no lineales con robustez. | Mayor complejidad de explicación, innecesaria para el alcance docente. |
| SVM | Puede producir fronteras de clasificación útiles. | Más costosa con codificación categórica de alta dimensión. |
| Regresión Logística | Rápida, auditable y compatible con variables codificadas. | Frontera lineal y posibles interacciones no representadas. |

## Consecuencias

- Se obtiene un punto de partida reproducible con una salida probabilística y coeficientes inspeccionables.
- `class_weight="balanced"` cambia el compromiso entre precision y recall, por lo que deben publicarse ambas métricas.
- El test observado supera al Dummy en las cuatro métricas, pero el resultado es descriptivo de un único split.
- La interpretación de coeficientes no establece causalidad ni justifica decisiones sobre personas.

## Criterios de aceptación académica

En el test estratificado del 20% y semilla 42: accuracy > 0.75, F1 > 0.60, recall de `>50K` > 0.55 y mejora frente al Dummy en las cuatro métricas. Los resultados reproducidos están en el informe final y en el Notebook 02.

# C4 - Contexto del sistema

El diagrama identifica a quienes interactúan con el mini-proyecto, su límite y la fuente externa del dataset.

```mermaid
C4Context
    title Contexto - Proyecto Adult Income ML

    Person(estudiante, "Estudiante", "Ejecuta los notebooks y revisa los resultados")
    Person(docente, "Docente", "Evalúa el método, resultados y documentación")
    System(proyecto, "Proyecto Adult Income ML", "Explora datos, entrena modelos básicos y calcula métricas agregadas")
    System_Ext(uci, "UCI Machine Learning Repository", "Publica el dataset Adult, ID 2")

    Rel(estudiante, proyecto, "Ejecuta y consulta")
    Rel(docente, proyecto, "Revisa")
    Rel(proyecto, uci, "Descarga Adult mediante script")
```

La solución se ejecuta localmente y no expone un servicio de predicción. La descarga desde UCI ocurre solo cuando se invoca el script correspondiente.

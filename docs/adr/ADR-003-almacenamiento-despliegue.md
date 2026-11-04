# ADR-003: Almacenamiento y despliegue

- **Estado:** Aceptado
- **Fecha:** 2026-11-03
- **Responsable:** Andrés Felipe Villanueva Parra

## Contexto

El entregable es un mini-proyecto de curso que debe ser fácil de inspeccionar y reproducir. No se necesita un servicio web ni una interfaz que procese datos de personas.

## Decisión

Ejecutar localmente mediante notebooks y scripts de Python. Mantener en Git el código, la documentación, los outputs de los notebooks, las tablas métricas y las gráficas de evaluación. Excluir `data/adult_income.csv` y regenerarlo desde UCI mediante `data/descargar_datos.py`. No desplegar un endpoint de inferencia ni almacenar perfiles individuales.

## Alternativas consideradas

- Aplicación web o API: facilitaría consultas, pero añade infraestructura y crea una vía de uso fuera del propósito académico.
- Almacenamiento en nube o base de datos: innecesario para un análisis estático y aumenta la exposición de datos.
- Versionar el CSV fuente: simplifica la primera ejecución sin red, pero duplica datos fuente y aumenta el repositorio.

## Consecuencias

- La primera reproducción requiere conexión a UCI para descargar el CSV; los pasos siguientes usan datos locales.
- El historial Git conserva las decisiones y resultados académicos, pero no un modelo serializado ni registros personales.
- La ausencia de despliegue reduce complejidad, aunque cualquier uso posterior requeriría nueva revisión técnica, ética y legal.
- Los datos de Adult son públicos, históricos y sensibles en su contexto; las métricas publicadas son agregadas y no autorizan aplicaciones reales.

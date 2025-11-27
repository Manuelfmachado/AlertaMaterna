# AlertaMaterna

## Sistema de Anticipación del Riesgo Obstétrico en la Región Orinoquía

AlertaMaterna es un sistema de Machine Learning diseñado para predecir el riesgo obstétrico en zonas territoriales de la región Orinoquía de Colombia (Meta, Arauca, Casanare, Vichada y Guaviare). El sistema analiza 38 variables demográficas, clínicas, institucionales y socioeconómicas para clasificar municipios como zonas de alto o bajo riesgo obstétrico.

## Características Principales

- **Modelo Predictivo**: XGBoost y Random Forest con 96% de precisión
- **38 Features Engineered**: Variables demográficas, clínicas, institucionales y socioeconómicas
- **Dashboard Interactivo**: Visualización con Streamlit, mapas interactivos con Folium
- **Análisis Temporal**: Datos de 2015-2018
- **Simulador de Riesgo**: Herramienta para evaluar escenarios hipotéticos

## Métricas del Modelo

- **Accuracy**: 96.1% (XGBoost), 100% (Random Forest)
- **ROC-AUC**: 1.000
- **Precision**: 0.97 (macro avg)
- **Recall**: 0.95 (macro avg)

### Top Features Predictivos

1. Tasa mortalidad fetal (20.6%)
2. Camas per cápita (20.5%)
3. Presión obstétrica (13.3%)
4. Total nacimientos (8.9%)
5. Porcentaje de prematuros (8.6%)

## Instalación

### Requisitos Previos

- Python 3.8+
- pip

### Instalar Dependencias

```bash
cd AlertaMaterna
pip install -r requirements.txt
```

## Estructura del Proyecto

```
AlertaMaterna/
├── data/
│   ├── processed/        # Features generadas
│   └── predictions/      # Predicciones del modelo
├── models/               # Modelos entrenados (.pkl)
├── src/
│   ├── config.py        # Configuración central
│   ├── utils.py         # Funciones auxiliares
│   ├── feature_engineering.py  # Generación de features
│   └── model_training.py       # Entrenamiento de modelos
├── app.py               # Dashboard Streamlit
├── requirements.txt     # Dependencias
└── README.md           # Este archivo
```

## Uso

### 1. Generar Features

```bash
python src/feature_engineering.py
```

Este script:
- Carga datos normalizados de nacimientos, defunciones, indicadores y prestadores
- Filtra los 5 departamentos objetivo
- Genera 38 features por municipio-año
- Guarda en `data/processed/features_alerta_materna.csv`

### 2. Entrenar Modelo

```bash
python src/model_training.py
```

Este script:
- Crea variable target (alto/bajo riesgo)
- Balancea clases con SMOTE
- Entrena XGBoost y Random Forest
- Evalúa métricas y genera reportes
- Guarda modelos en `models/`

### 3. Lanzar Dashboard

```bash
streamlit run app.py
```

El dashboard se abrirá automáticamente en `http://localhost:8501`

## Características del Dashboard

### 1. Resumen Ejecutivo

- Indicadores clave (registros, zonas de alto riesgo, probabilidad promedio)
- Distribución de riesgo (gráfico circular)
- Evolución temporal del riesgo
- Top 10 factores predictivos

### 2. Mapa de Riesgo

- Visualización geográfica interactiva
- Marcadores por municipio (verde=bajo riesgo, rojo=alto riesgo)
- Popups con detalles de cada zona
- Filtros por año

### 3. Análisis Detallado

- Información general por municipio-año
- Indicadores clínicos detallados
- Métricas institucionales
- Comparación de variables

### 4. Simulador de Riesgo

- Ajuste de parámetros demográficos, clínicos e institucionales
- Predicción en tiempo real
- Visualización de probabilidad (medidor tipo gauge)
- Recomendaciones según nivel de riesgo

## Features Generadas

### Demográficas (6)

- total_nacimientos
- edad_materna_promedio
- pct_madres_adolescentes
- pct_madres_edad_avanzada
- pct_bajo_nivel_educativo
- total_defunciones

### Clínicas (11)

- tasa_mortalidad_fetal
- pct_bajo_peso
- pct_embarazo_multiple
- pct_cesarea
- pct_prematuro
- apgar_bajo_promedio

### Institucionales (6)

- num_instituciones
- presion_obstetrica (nacimientos/institución)
- camas_per_capita
- pct_instituciones_publicas

### Socioeconómicas (3)

- pct_sin_seguridad_social
- pct_area_rural
- pct_regimen_subsidiado

### Atención Prenatal (2)

- pct_sin_control_prenatal
- consultas_promedio

## Datos de Entrada

El sistema utiliza datos normalizados de:

- **Nacimientos** (2015-2018): 2.6M registros
- **Defunciones Fetales** (2015-2018): 183K registros
- **Indicadores de Mortalidad/Morbilidad**: 266K registros
- **Prestadores de Salud (REPS)**: 76K registros

Filtrado a 5 departamentos:

- Meta (50)
- Arauca (81)
- Casanare (85)
- Vichada (99)
- Guaviare (95)

**Total**: 114,383 nacimientos, 11,478 defunciones fetales

## Metodología

1. **Carga de datos**: Desde archivos CSV normalizados
2. **Filtrado territorial**: 5 departamentos de la Orinoquía
3. **Feature Engineering**: 38 variables agregadas por municipio-año
4. **Creación de target**: Clasificación alto/bajo riesgo basada en umbrales
5. **Balanceo**: SMOTE para equilibrar clases
6. **Entrenamiento**: XGBoost y Random Forest
7. **Evaluación**: Métricas de clasificación, ROC-AUC, feature importance
8. **Predicción**: Clasificación de todas las zonas
9. **Visualización**: Dashboard interactivo

## Criterios de Alto Riesgo

Una zona se clasifica como alto riesgo si cumple AL MENOS UNO de:

- Tasa mortalidad fetal > P75 (26.09‰)
- Camas per cápita < P25 (155.77)
- Presión obstétrica > 100 nacimientos/institución
- Sin instituciones de salud

## Casos de Uso

1. **Planificación de Salud Pública**: Identificar zonas que requieren inversión
2. **Asignación de Recursos**: Priorizar departamentos con mayor riesgo
3. **Monitoreo Temporal**: Evaluar evolución del riesgo obstétrico
4. **Análisis de Impacto**: Simular efectos de mejoras en infraestructura
5. **Alertas Tempranas**: Detectar deterioro de indicadores

## Resultados

El modelo genera:

- **252 registros** (municipios-año únicos)
- **101 zonas de alto riesgo** (40.1%)
- **151 zonas de bajo riesgo** (59.9%)
- Probabilidad promedio de riesgo: 40.6%

## Configuración

Editar `src/config.py` para:

- Modificar departamentos objetivo
- Ajustar umbrales de riesgo
- Cambiar hiperparámetros del modelo
- Actualizar lista de features

## Tecnologías

### Machine Learning

- **XGBoost**: Gradient Boosting de alto rendimiento
- **Random Forest**: Ensemble learning robusto
- **Scikit-learn**: Preprocessing, métricas, validación
- **Imbalanced-learn**: SMOTE para balanceo de clases

### Análisis de Datos

- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas

### Visualización

- **Streamlit**: Dashboard web interactivo
- **Plotly**: Gráficos interactivos
- **Folium**: Mapas geográficos

## Licencia

MIT License - Ver archivo LICENSE para detalles.

## Contacto

Proyecto AlertaMaterna - Anticipación del riesgo obstétrico en la región Orinoquía

## Agradecimientos

Datos proporcionados por:

- Ministerio de Salud de Colombia
- DANE (Departamento Administrativo Nacional de Estadística)
- Registros vitales de nacimientos y defunciones

---

**AlertaMaterna v1.0** | 2025 | Sistema de Anticipación del Riesgo Obstétrico

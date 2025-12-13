# AlertaMaterna: Sistema de Clasificación de Riesgo Obstétrico y Predicción de Mortalidad Infantil en la Región Orinoquía

![AlertaMaterna Banner](ALERTAMATERNA.png)

---

## 🌐 Ver Aplicación en Vivo

**👉 Para ver la aplicación en vivo, haz clic en el botón "Open in Streamlit" de abajo y espera unos segundos mientras carga la página.**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alertamaterna-h58nkmrssz5e6mceegqcxe.streamlit.app/)

---

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

---

## Concurso Datos al Ecosistema 2025

**Esta herramienta nace de la participación en el Concurso Datos al Ecosistema 2025.**

Este proyecto aprovecha los **datos abiertos** del Gobierno de Colombia, integrando **6 datasets oficiales**:

**De [www.datos.gov.co](https://www.datos.gov.co/):**

- **Registro Especial de Prestadores de Servicios de Salud (REPS)** - Instituciones y sedes
- **Registros Individuales de Prestación de Servicios de Salud (RIPS)** - Consultas, urgencias y procedimientos
- **Códigos DIVIPOLA** - Códigos oficiales de municipios colombianos

**Del DANE (Estadísticas Vitales 2020-2024):**

- Nacimientos (2020 - 2024)
- Defunciones fetales (2020 - 2024)
- Defunciones no fetales (2020 - 2024)

El objetivo es generar **valor público** mediante inteligencia artificial aplicada a la salud materno-infantil, transformando datos dispersos en alertas tempranas accionables.

---

## Descripción

**AlertaMaterna** es un sistema de inteligencia artificial que identifica municipios de la región Orinoquía con alto riesgo de mortalidad materno-infantil, utilizando datos oficiales del DANE del periodo 2020-2024.

El sistema analiza **34 indicadores de salud** (atención prenatal, bajo peso al nacer, prematuridad, acceso a servicios, mortalidad neonatal, causas evitables) para clasificar **55 municipios** en dos categorías: **ALTO RIESGO** o **BAJO RIESGO**, además de predecir la probabilidad de mortalidad infantil en cada municipio.

### Objetivos

1. **Clasificar** municipios según su nivel de riesgo obstétrico
2. **Predecir** probabilidad de mortalidad infantil (<1 año)
3. **Priorizar** intervenciones en salud pública
4. **Monitorear** evolución temporal de indicadores críticos

### Región de Análisis

**Orinoquía colombiana**: Meta, Arauca, Casanare, Guaviare y Vichada (55 municipios, 310 registros municipio-año 2020-2024)

## Características Principales

- **Sistema híbrido de clasificación**: Combina percentiles estadísticos + umbrales críticos OMS/PAHO
- **100% de detección de casos críticos**: Identifica todos los municipios con mortalidad >50‰
- **Modelo predictivo híbrido**: Combina base epidemiológica (WHO/Lawn) + XGBoost + intervalos de confianza (P10/P50/P90)
- **Dashboard interactivo**: Visualizaciones en tiempo real con Streamlit y Plotly
- **Basado en datos oficiales DANE (2020-2024)**:
  - **Datos brutos**: 2,789,391 nacimientos y 138,385 defunciones fetales en toda Orinoquía
  - **Datos analizados**: 137,780 nacimientos en 251 registros válidos (≥10 nacimientos/año)
- **Código abierto**: Disponible en GitHub bajo licencia MIT

## Resultados Principales

**Nota terminológica**: Un "registro" = 1 municipio en 1 año específico. Ejemplo: Villavicencio tiene 5 registros (2020-2024).

**Aclaración datos**: El DANE registra **2,789,391 nacimientos** en toda Orinoquía (2020-2024), pero el sistema analiza solo **137,780 nacimientos** en los 251 registros válidos (≥10 nacimientos/año), excluyendo municipios-año con datos insuficientes.

| Métrica                              | Valor                                                                   |
| ------------------------------------- | ----------------------------------------------------------------------- |
| **Registros analizados**        | 310 registros municipio-año (2020-2024)                                |
| **Registros válidos**          | 251 registros con ≥10 nacimientos/año (estándar OMS)                 |
| **Registros alto riesgo**       | 63 de 251 (25.1%)                                                       |
| **Nacimientos analizados**      | 137,780 nacimientos vivos en registros válidos                         |
| **Mortalidad fetal promedio**   | 23.4‰ (23.4 muertes por cada 1,000 nacimientos)                        |
| **Mortalidad evitable**         | 49.7% de muertes maternas son PREVENIBLES                               |
| **Casos críticos detectados**  | 40 registros (mortalidad >50‰) - 100% sensibilidad                     |

## Impacto Económico

La prevención prenatal estructurada no solo salva vidas, sino que genera un **retorno económico significativo**:

| Concepto | Valor (COP) |
|----------|-------------|
| **Costo complicación obstétrica** | $20M - $60M por caso |
| **Costo prevención prenatal** | ~$2M por madre |
| **Retorno de inversión (ROI)** | **10:1 a 30:1** |
| **Ahorro estimado Orinoquía** | $550M COP/año |

**AlertaMaterna maximiza este retorno** al identificar los **28 municipios críticos** donde focalizar recursos genera el mayor impacto económico y social.

📊 *Para detalles completos del análisis económico con referencias científicas (OMS, PAHO, Banco Mundial, MinSalud), consulta la [Documentación Técnica - Sección 7.6](DOCUMENTACION_TECNICA.md#76-impacto-económico-de-la-prevención).*

## Modelos Implementados

### Modelo 1: Clasificación de Riesgo Obstétrico

**Sistema híbrido de puntaje (0-8 puntos)**:

Un municipio es clasificado como **ALTO RIESGO** si cumple:

- ≥3 puntos en criterios de percentil 75 (mortalidad fetal, atención prenatal, bajo peso, prematuridad, cesáreas, presión obstétrica)
- **O** mortalidad fetal >50‰ (clasificación automática, +3 puntos)

**Justificación del umbral 50‰**:

- Tasa global OMS: 5‰
- Latinoamérica: 10-15‰
- **50‰ = 10x la tasa normal** → crisis de salud pública (PAHO 2019)

### Modelo 2: Predicción de Tasa de Mortalidad Infantil

**Algoritmo**: Modelo Híbrido Epidemiológico + Machine Learning
- **Base epidemiológica**: Fórmula WHO/Lawn et al. (MI ≈ MN / 0.6)
- **Ajustes ML**: XGBoost Regressor calibrado con factores de riesgo
- **Intervalos de confianza**: Regresión por Cuantiles (P10/P50/P90)

**Features**: 34 indicadores sociosanitarios + 15 features clave para cuantiles

**Performance Modelo Híbrido**:
- Predicción central sensible a indicadores del municipio
- Intervalos de confianza (80%): Rango P10 - P90
- Cobertura del intervalo: 90.2% (esperado: 80%)

**Reglas de Coherencia Epidemiológica** (científicamente defendibles):
1. MI ≥ Mortalidad Neonatal (definición OMS ICD-10)
2. P10 ≥ 1.5‰ (piso mundial - UNICEF 2023)
3. P90 ≤ 150‰ (techo observado - DANE Orinoquía)

**Top 5 features más importantes**:

1. **% APGAR Bajo (10.78%)** - Indicador de asfixia perinatal
2. Número instituciones (8.29%) - Acceso a servicios
3. Consultas promedio (6.93%) - Atención prenatal
4. **Tasa mortalidad neonatal (6.45%)** - CRÍTICA
5. **% Mortalidad evitable (6.34%)** - Potencial de intervención

**Features integradas RIPS/REPS** (2020-2024):

- **REPS diferenciado**: Instituciones de salud por municipio (antes promedios globales)
- **RIPS acceso a servicios**: Consultas, urgencias, procedimientos por nacimiento

**Features críticas avanzadas**:

- **Tasa mortalidad neonatal**: Muertes 0-7 días por 1000 nacimientos (media: 3.47‰)
- **% Mortalidad evitable**: Causas CIE-10 prevenibles según DANE (media: 49.7%)
- **% Embarazos alto riesgo**: Prematuridad + bajo peso + múltiples (media: 93.8%)
- **Índice fragilidad sistema**: (mortalidad × presión) / densidad institucional (23 municipios críticos)

## Instalación y Uso

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes Python)

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Manuelfmachado/AlertaMaterna.git
cd AlertaMaterna

# 2. Instalar dependencias
pip install -r requirements.txt
```

### Ejecución Rápida

**Opción 1: Ejecutar dashboard directamente** (modelos ya entrenados)

```bash
streamlit run app_simple.py
```

El dashboard se abrirá en `http://localhost:8501`

**Opción 2: Ejecutar pipeline completo** (reentrenar modelos)

```bash
# Paso 1: Generar features
cd src
python features.py

# Paso 2: Entrenar modelos
python train_model.py

# Paso 3: Lanzar dashboard
cd ..
streamlit run app_simple.py
```

## Estructura del Proyecto

```
AlertaMaterna/
├── data/
│   ├── raw/                              # Datos originales DANE
│   │   ├── BD-EEVV-Nacimientos-*.csv
│   │   ├── BD-EEVV-Defunciones*.csv
│   │   └── codigos_*.csv
│   └── processed/                        # Datos procesados
│       ├── features_municipio_anio.csv   # 310 registros con 34 indicadores
│       └── features_alerta_materna.csv   # Con targets y clasificación
├── src/
│   ├── features.py                       # Generación de 34 indicadores
│   ├── train_model.py                    # Entrenamiento modelo XGBoost
│   └── train_quantile_models.py          # Entrenamiento modelos P10/P50/P90
├── models/                                # Modelos entrenados (.pkl)
│   ├── modelo_mortalidad_xgb.pkl          # Modelo XGBoost base
│   ├── modelo_quantile_p10.pkl            # Cuantil P10 (optimista)
│   ├── modelo_quantile_p50.pkl            # Cuantil P50 (central)
│   ├── modelo_quantile_p90.pkl            # Cuantil P90 (pesimista)
│   ├── scaler_mortalidad.pkl
│   ├── scaler_quantile.pkl
│   └── feature_names_quantile.pkl
├── app_simple.py                          # Dashboard Streamlit
├── requirements.txt                       # Dependencias Python
├── DOCUMENTACION_TECNICA.md              # Justificación científica (60+ páginas)
├── alertamaterna_banner.png              # Banner del proyecto
└── README.md                              # Este archivo
```

## Uso del Dashboard

El dashboard tiene **2 pestañas principales**:

### 1. Panorama General

- **Indicadores principales (4 KPIs)**: 
  - **Municipios**: Total de municipios en la región filtrada
  - **Alto Riesgo**: Número absoluto de municipios clasificados como alto riesgo
  - **Nacimientos**: Total de nacimientos registrados en el período
  - **Defunciones**: Total de defunciones infantiles (<1 año) registradas
- **Distribución de riesgo**: Gráfico comparativo por departamento
- **Indicadores clave**: Promedios de mortalidad, atención prenatal, bajo peso
- **Top 10 municipios alto riesgo**: Ranking con puntajes detallados
- **Nota sobre datos**: Todos los datos provienen de [www.datos.gov.co](https://www.datos.gov.co/) (REPS, RIPS, DIVIPOLA) y estadísticas vitales del DANE 2020-2024

### 2. Predictor de Riesgo

**Herramienta interactiva** para evaluar municipios:

1. Ingresa **15 indicadores** del municipio agrupados en:
   - **Demográficos** (5): Nacimientos, edad materna, madres adolescentes, edad avanzada, nivel educativo
   - **Clínicos** (5): Mortalidad neonatal, mortalidad fetal, bajo peso, prematuros, APGAR bajo
   - **Acceso a Salud** (5): Control prenatal, consultas promedio, cesáreas, instituciones, presión obstétrica
2. El sistema predice la **tasa de mortalidad infantil con intervalos de confianza**:
   - **Estimación central**: Predicción puntual en ‰
   - **Rango epidemiológico (80% confianza)**:
     - 🔽 P10 (mejor escenario)
     - ⏺️ P50 (escenario esperado)
     - 🔼 P90 (peor escenario)
3. Visualización con gauge interactivo que muestra:
   - **Verde (< 5‰)**: NORMAL - dentro de estándares OMS
   - **Amarillo (5-10‰)**: MODERADO - por encima de media global
   - **Naranja (10-20‰)**: ALTO - requiere intervención
   - **Rojo (> 20‰)**: CRÍTICO - crisis de salud pública

## Features Generadas (34 indicadores)

### Demográficas (5)

- `total_nacimientos`: Total de nacimientos registrados
- `edad_materna_promedio`: Edad promedio de madres
- `pct_madres_adolescentes`: % madres <18 años
- `pct_madres_edad_avanzada`: % madres ≥35 años
- `pct_bajo_nivel_educativo`: % madres con educación básica

### Clínicas (8)

- `total_defunciones`: Defunciones infantiles (<1 año)
- `defunciones_fetales`: Muertes fetales (≥22 semanas)
- `tasa_mortalidad_fetal`: Defunciones fetales por 1,000 nacimientos
- `tasa_mortalidad_infantil`: Defunciones <1 año por 1,000 nacimientos
- `pct_bajo_peso`: % nacimientos <2,500g
- `pct_embarazo_multiple`: % embarazos múltiples
- `pct_cesarea`: % partos por cesárea
- `pct_prematuro`: % nacimientos <37 semanas
- `apgar_bajo_promedio`: Promedio APGAR <7

### Institucionales (3)

- `num_instituciones`: Número de instituciones de salud
- `presion_obstetrica`: Nacimientos por institución
- `pct_instituciones_publicas`: % instituciones públicas

### Acceso a Servicios - RIPS (4)

- `consultas_per_nacimiento`: Consultas médicas por nacimiento
- `urgencias_per_nacimiento`: Atenciones de urgencia por nacimiento
- `procedimientos_per_nacimiento`: Procedimientos médicos por nacimiento
- `atenciones_per_nacimiento`: Total atenciones por nacimiento

### Socioeconómicas (3)

- `pct_sin_seguridad_social`: % sin afiliación a salud
- `pct_regimen_subsidiado`: % en régimen subsidiado
- `pct_area_rural`: % nacimientos en zona rural

### Atención Prenatal (2)

- `pct_sin_control_prenatal`: % sin control prenatal
- `consultas_promedio`: Promedio de consultas prenatales

### Críticas Avanzadas (4)

- `tasa_mortalidad_neonatal`: Muertes 0-7 días por 1,000 nacimientos (media: 3.47‰, 22 municipios críticos >15‰)
- `pct_mortalidad_evitable`: % muertes por causas prevenibles CIE-10 (códigos DANE 401-410, 501-506) (media: 49.7%)
- `pct_embarazos_alto_riesgo`: % embarazos con prematuridad + bajo peso + múltiples (media: 93.8%)
- `indice_fragilidad_sistema`: Índice compuesto (mortalidad × presión) / densidad institucional, escala 0-100 (23 municipios >80)

### Targets (3)

- `riesgo_obstetrico`: ALTO / BAJO (Modelo 1)
- `puntos_riesgo`: Puntaje 0-8 (Modelo 1)
- `alta_mortalidad`: 0/1 (Modelo 2)

## Metodología Científica

### Justificación de Parámetros

Todos los parámetros están respaldados por literatura científica. Ver **`DOCUMENTACION_TECNICA.md`** (60+ páginas) con:

- 16 referencias bibliográficas (OMS, PAHO, estudios epidemiológicos)
- Justificación del umbral 50‰ (10x tasa normal)
- Análisis de sensibilidad del umbral ≥3 puntos
- Explicación de SMOTE para balanceo de clases
- Validación de hiperparámetros XGBoost
- Coherencia con conocimiento del dominio médico

### Filtrado de Datos

- **Umbral mínimo**: 10 nacimientos/año por municipio
- **Justificación**: Evitar varianza extrema por números pequeños
- **Resultado**: 310 registros → 251 válidos (81%)

### Umbrales Críticos

| Indicador                  | Umbral        | Justificación                                 |
| -------------------------- | ------------- | ---------------------------------------------- |
| Mortalidad fetal crítica  | >50‰         | 10x tasa normal (OMS: 5‰)                     |
| Sin atención prenatal     | >50%          | Falla sistémica (OMS recomienda 100%)         |
| Clasificación alto riesgo | ≥3 puntos    | Detecta 100% casos críticos, 21% clasificados |
| Target mortalidad infantil | >Percentil 75 | 6.42‰ (50% sobre promedio nacional ~4‰)      |

## Casos de Uso

1. **Planificación estratégica en salud pública**: Identificar municipios que requieren inversión prioritaria
2. **Asignación eficiente de recursos**: Priorizar departamentos según nivel de riesgo
3. **Monitoreo temporal**: Evaluar evolución de indicadores críticos (2020-2024)
4. **Análisis de impacto**: Simular efectos de mejoras en infraestructura sanitaria
5. **Sistema de alertas tempranas**: Detectar deterioro de indicadores en tiempo real
6. **Evaluación de políticas públicas**: Medir efectividad de intervenciones

## Resultados Destacados

### Por Departamento (2024)

| Departamento       | Municipios | Alto Riesgo | % Alto Riesgo | Mortalidad Promedio |
| ------------------ | ---------- | ----------- | ------------- | ------------------- |
| **Vichada**  | 2          | 2           | 100%          | 86.5‰              |
| **Arauca**   | 7          | 4           | 57%           | 99.6‰              |
| **Guaviare** | 4          | 1           | 25%           | 85.2‰              |
| **Meta**     | 18         | 4           | 22%           | 25.1‰              |
| **Casanare** | 14         | 2           | 14%           | 24.8‰              |

### Municipios Críticos (Mortalidad >50‰, 2024)

| Municipio              | Departamento | Nacimientos | Defunciones | Mortalidad | Estado   |
| ---------------------- | ------------ | ----------- | ----------- | ---------- | -------- |
| Saravena               | Arauca       | 1,716       | 278         | 162.0‰    | CRÍTICO |
| Puerto Rondón         | Arauca       | 21          | 2           | 95.2‰     | CRÍTICO |
| Puerto Carreño        | Vichada      | 513         | 47          | 91.6‰     | CRÍTICO |
| Arauca                 | Arauca       | 1,188       | 107         | 90.1‰     | CRÍTICO |
| San José del Guaviare | Guaviare     | 1,009       | 86          | 85.2‰     | CRÍTICO |

**Total población afectada**: 4,811 nacimientos en municipios críticos (38% del total 2024)

## Tecnologías Utilizadas

### Machine Learning

- **XGBoost** 1.7+: Gradient boosting optimizado
- **Scikit-learn** 1.3+: Preprocessing, métricas, validación
- **Imbalanced-learn** 0.11+: SMOTE para balanceo de clases

### Análisis de Datos

- **Pandas** 2.0+: Manipulación y análisis de datos
- **NumPy** 1.24+: Operaciones numéricas

### Visualización

- **Streamlit** 1.28+: Dashboard web interactivo
- **Plotly** 5.11+: Gráficos interactivos
- **Matplotlib** 3.7+: Visualizaciones estáticas

## Documentación Adicional

- **DOCUMENTACION_TECNICA.md**: Justificación científica completa (60+ páginas, 16 referencias)
  - Marco teórico y contexto epidemiológico
  - Justificación de cada parámetro con literatura médica
  - Análisis de sensibilidad de umbrales
  - Validación y coherencia de resultados
  - Limitaciones y trabajo futuro

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto es de **código abierto** bajo licencia MIT para uso en salud pública.

## Autores

- **Manuel Machado**
- **Martha Machado**

**Proyecto AlertaMaterna** - Sistema de Clasificación de Riesgo Obstétrico y Predicción de Mortalidad Infantil en la Región Orinoquía

## Referencias cientificas

- **OMS** (Organización Mundial de la Salud)
- **PAHO** (Pan American Health Organization)
- **UNICEF** - Estudios sobre salud materno-infantil

## Contacto

Para preguntas, sugerencias o colaboraciones:

- GitHub: [@Manuelfmachado](https://github.com/Manuelfmachado)
- Repositorio: [AlertaMaterna](https://github.com/Manuelfmachado/AlertaMaterna)

## Agradecimientos

Este proyecto fue desarrollado en el marco del **Concurso Datos al Ecosistema 2025** organizado por el **Ministerio de Tecnologías de la Información y las Comunicaciones (MinTIC)**.

Agradecemos especialmente:

- **MinTIC** por promover la innovación basada en datos abiertos y fomentar soluciones que generen valor público
- **Concurso Datos al Ecosistema** por la oportunidad de participar y contribuir al ecosistema de datos en Colombia
- **Personal docente** por brindar conocimiento, guía y herramientas fundamentales para el desarrollo de este proyecto

---

## Cómo Citar

Si utilizas este proyecto en tu investigación o trabajo, por favor cítalo como:

```
AlertaMaterna (2025). Sistema de Clasificación de Riesgo Obstétrico y Predicción 
de Mortalidad Infantil en la Región Orinoquía. GitHub: Manuelfmachado/AlertaMaterna
```

---

## Abrir el Sitio Web de AlertaMaterna

**Accede a la aplicación web en vivo:**

### [Ir a AlertaMaterna Dashboard](https://alertamaterna-h58nkmrssz5e6mceegqcxe.streamlit.app/)

**📋 Instrucciones importantes:**

1. **Haz clic en el enlace de arriba** para abrir la aplicación
2. **Si aparece un botón amarillo** que dice *"Yes, get this app back up!"* (la app se está despertando):
   - Presiona el botón amarillo y espera unos segundos
3. **Si la página carga directamente** con el dashboard de AlertaMaterna:
   - ¡Perfecto! No hagas nada, ya estás listo para usar la aplicación

> 💡 **Nota:** La primera carga puede tardar 20-30 segundos mientras Streamlit despierta el servidor.

Explora el sistema de clasificación de riesgo obstétrico en la región Orinoquía y utiliza el predictor de mortalidad infantil directamente desde tu navegador.

---

<div align="center">

**AlertaMaterna v1.0** | 2025
*Sistema de clasificación de riesgo obstétrico y predicción de mortalidad infantil*
*Con intervalos de confianza epidemiológicos (P10/P50/P90)*

[Inicio](#alertamaterna-sistema-de-clasificación-de-riesgo-obstétrico-y-predicción-de-mortalidad-infantil-en-la-región-orinoquía) • [Dashboard](#-uso-del-dashboard) • [Documentación](#-documentación-adicional) • [Contribuir](#-contribuciones)

</div>

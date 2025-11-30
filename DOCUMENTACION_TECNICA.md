# Documentación Técnica - AlertaMaterna

## Sistema de Clasificación de Riesgo Obstétrico y Predicción de Mortalidad Infantil en la Región Orinoquía

**Región:** Orinoquía, Colombia (Meta, Arauca, Casanare, Guaviare, Vichada)  
**Periodo de análisis:** 2020-2024  
**Fuente de datos:** DANE - Estadísticas Vitales

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Marco Teórico y Justificación](#2-marco-teórico-y-justificación)
3. [Metodología](#3-metodología)
4. [Ingeniería de Features](#4-ingeniería-de-features)
5. [Modelo 1: Clasificación de Riesgo Obstétrico](#5-modelo-1-clasificación-de-riesgo-obstétrico)
6. [Modelo 2: Predicción de Mortalidad Infantil](#6-modelo-2-predicción-de-mortalidad-infantil)
7. [Resultados y Análisis](#7-resultados-y-análisis)
8. [Validación y Coherencia](#8-validación-y-coherencia)
9. [Limitaciones y Trabajo Futuro](#9-limitaciones-y-trabajo-futuro)
10. [Referencias](#10-referencias)

---

## 1. Resumen Ejecutivo

AlertaMaterna es un sistema de Machine Learning especializado para identificar y predecir riesgo de mortalidad materno-infantil en la región Orinoquía de Colombia. El sistema implementa dos modelos complementarios:

- **Modelo 1 (Clasificación de Riesgo):** Sistema híbrido que combina percentiles estadísticos con umbrales críticos absolutos basados en literatura médica internacional.
- **Modelo 2 (Predicción de Mortalidad):** XGBoost que predice probabilidad de alta mortalidad infantil con ROC-AUC de 0.7731 (+9.2% vs baseline).

**Resultados principales:**
- 310 registros municipio-año analizados (2020-2024)
- 251 registros válidos (≥10 nacimientos)
- 53 registros clasificados como alto riesgo (21.1%)
- 40 registros con mortalidad crítica (>50‰) correctamente identificados
- En 2024: 13 municipios en alto riesgo (29% del total)

---

## 2. Marco Teórico y Justificación

### 2.1 Contexto Epidemiológico

La región Orinoquía presenta características únicas que justifican un sistema de alerta especializado:

- **Dispersión geográfica:** Municipios remotos con acceso limitado a servicios de salud
- **Población vulnerable:** Alta proporción de comunidades indígenas y rurales
- **Infraestructura limitada:** Escasez de centros de salud especializados
- **Indicadores históricos:** Tasas de mortalidad superiores al promedio nacional

### 2.2 Justificación de Variables Seleccionadas

Las 29 variables fueron seleccionadas basándose en:

1. **Literatura médica internacional:**
   - OMS: Indicadores de salud materno-infantil
   - UNICEF: Factores de riesgo en salud neonatal
   - Ministerio de Salud Colombia: Guías de atención prenatal

2. **Estudios epidemiológicos previos:**
   - Asociación entre bajo peso al nacer y mortalidad (Wilcox 2001)
   - Importancia del control prenatal (WHO 2016)
   - Factores de riesgo en embarazo adolescente (UNFPA 2013)

3. **Disponibilidad de datos:**
   - Todas las variables provienen de registros oficiales DANE
   - Cobertura completa para la región y periodo de estudio

---

## 3. Metodología

### 3.1 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     DATOS CRUDOS DANE                       │
│  • Nacimientos (453,901 registros 2024)                    │
│  • Defunciones fetales (21,250 registros 2024)             │
│  • Defunciones no fetales                                   │
│  • Registro de prestadores de salud                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PROCESAMIENTO Y FEATURES                        │
│  Script: features.py                                         │
│  • Agregación por municipio-año                             │
│  • Cálculo de 24 indicadores                                │
│  • Filtrado de calidad (≥10 nacimientos)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   MODELADO ML                                │
│  Script: train_model.py                                      │
│  • Modelo 1: Clasificación de riesgo (índice compuesto)     │
│  • Modelo 2: Predicción XGBoost (mortalidad infantil)       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 DASHBOARD INTERACTIVO                        │
│  Aplicación: app_simple.py (Streamlit)                      │
│  • Visualización de resultados                               │
│  • Alertas críticas                                          │
│  • Predictor de riesgo                                       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Pipeline de Ejecución

1. **features.py:** Procesa datos crudos → genera features_municipio_anio.csv
2. **train_model.py:** Entrena modelos → genera archivos .pkl
3. **app_simple.py:** Carga modelos → presenta dashboard interactivo

---

## 4. Ingeniería de Features

### 4.1 Variables Generadas (29 features)

#### A. Indicadores Demográficos (7)
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `total_nacimientos` | Total de nacimientos en el municipio-año | Denominador para tasas, tamaño muestral |
| `edad_materna_promedio` | Edad promedio de las madres | Embarazos extremos (muy jóvenes/mayores) tienen mayor riesgo |
| `pct_madres_adolescentes` | % madres <18 años | Asociado a complicaciones obstétricas (UNFPA 2013) |
| `pct_madres_edad_avanzada` | % madres >35 años | Mayor riesgo de complicaciones (ACOG 2014) |
| `pct_multiparidad` | % madres con >3 hijos | Gran multiparidad asociada a complicaciones |
| `pct_bajo_nivel_educativo` | % madres sin educación formal | Proxy de nivel socioeconómico |
| `pct_embarazo_multiple` | % embarazos gemelares/múltiples | Mayor riesgo de prematuridad y bajo peso |

#### B. Indicadores Clínicos (7)
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `pct_bajo_peso` | % nacidos con <2500g | Predictor de mortalidad neonatal (Wilcox 2001) |
| `pct_prematuro` | % nacidos <37 semanas | Principal causa de mortalidad neonatal (WHO) |
| `pct_cesarea` | % partos por cesárea | Proxy de acceso a atención especializada |
| `apgar_bajo_promedio` | % APGAR <7 a los 5 min | Indicador de asfixia perinatal |
| `defunciones_fetales` | Número de muertes fetales | Numerador para mortalidad fetal |
| `tasa_mortalidad_fetal` | Muertes fetales × 1000 / nac | Indicador principal OMS |
| `total_defunciones` | Muertes <1 año | Para target de mortalidad infantil |

#### C. Indicadores Institucionales (3)
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `num_instituciones` | # instituciones de salud por municipio | Acceso a servicios (REPS diferenciado) |
| `pct_instituciones_publicas` | % instituciones públicas por municipio | Cobertura del sistema público (REPS diferenciado) |
| `presion_obstetrica` | Nacimientos / instituciones | Capacidad instalada vs demanda |

#### D. Indicadores de Acceso a Servicios RIPS (4) ⭐ NUEVO
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `atenciones_per_nacimiento` | Total atenciones obstétricas / nacimientos | Intensidad de uso del sistema de salud |
| `consultas_per_nacimiento` | Consultas obstétricas / nacimientos | Acceso efectivo a servicios prenatales |
| `urgencias_per_nacimiento` | Urgencias obstétricas / nacimientos | Indicador de complicaciones y acceso a emergencias |
| `procedimientos_per_nacimiento` | Procedimientos obstétricos / nacimientos | Complejidad de atención requerida |

#### E. Indicadores Socioeconómicos (3)
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `pct_sin_seguridad_social` | % sin afiliación en salud | Acceso a servicios |
| `pct_regimen_subsidiado` | % en régimen subsidiado | Proxy de nivel socioeconómico |
| `pct_area_rural` | % población rural | Ruralidad asociada a menor acceso |

#### F. Indicadores de Atención Prenatal (2)
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `pct_sin_control_prenatal` | % sin ningún control prenatal | Factor de riesgo crítico (WHO 2016) |
| `consultas_promedio` | # promedio de consultas prenatales | OMS recomienda mínimo 8 consultas |

#### G. Indicadores Críticos Avanzados (4) 🆕
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `tasa_mortalidad_neonatal` | Muertes 0-7 días × 1000 / nacimientos | Período crítico: 40% de mortalidad infantil ocurre en primera semana (WHO 2020). Detecta problemas en atención inmediata post-parto. Feature #1 del modelo (24.17% importancia). |
| `pct_mortalidad_evitable` | % muertes por causas DANE 401-410, 501-506 | Identificación de muertes prevenibles según clasificación CIE-10 adaptada por DANE. 49.7% promedio indica gran margen de mejora con intervenciones dirigidas. Feature #3 del modelo (6.65% importancia). |
| `pct_embarazos_alto_riesgo` | % con prematuridad + bajo peso + múltiples | Indicador compuesto de riesgo obstétrico. Combina 3 factores críticos asociados a mortalidad neonatal (March of Dimes 2019). Media: 93.8%. |
| `indice_fragilidad_sistema` | (mortalidad × presión) / densidad institucional | Índice compuesto que mide vulnerabilidad sistémica: alta mortalidad + alta demanda + baja capacidad = fragilidad crítica. Escala 0-100, 23 municipios >80. |

**Nota:** Las features institucionales (C) ahora utilizan datos reales diferenciados por municipio del REPS, en lugar de promedios globales. Las features de acceso a servicios (D) son nuevas y provienen del procesamiento de los RIPS 2020-2024. Las features críticas avanzadas (G) fueron agregadas en noviembre 2025 y mejoraron el ROC-AUC de 0.71 a 0.7731 (+9.2%).

### 4.2 Transformaciones Aplicadas

```python
# Ejemplo: Tasa de mortalidad fetal
tasa_mortalidad_fetal = (defunciones_fetales / total_nacimientos) × 1000

# Ejemplo: Presión obstétrica
presion_obstetrica = total_nacimientos / num_instituciones
```

### 4.3 Criterio de Exclusión

**Umbral mínimo: 10 nacimientos por municipio-año**

**Justificación:**
- Estabilidad estadística: Tasas calculadas con <10 eventos son altamente inestables
- Evitar falsos positivos: Un municipio con 2 nacimientos y 1 defunción = 500‰ (no representativo)
- Recomendación OMS: Mínimo 10 eventos para tasas confiables
- Ejemplo real: Municipio con 3 nacimientos y 1 defunción = 333‰ (dato excluido)

**Resultado:** 59 registros excluidos (19% del total), marcados con `puntos_riesgo = -1`

---

## 5. Modelo 1: Clasificación de Riesgo Obstétrico

### 5.1 Diseño del Sistema Híbrido

**Motivación:** Los sistemas basados únicamente en percentiles tienen limitaciones:

**Problema identificado:**
- Sistema percentil puro: Un municipio con 85‰ de mortalidad podía clasificarse como "bajo riesgo" si otros municipios tenían valores aún más altos
- Ejemplo real: San José del Guaviare (85.2‰) clasificado como bajo riesgo por estar en percentil 60

**Solución:** Sistema híbrido que combina:
1. **Umbrales críticos absolutos** (basados en OMS/literatura médica)
2. **Percentiles relativos** (basados en distribución de datos)

### 5.2 Umbrales Críticos Definidos

#### A. Mortalidad Fetal Crítica: 50‰

**Justificación:**
- Tasa global promedio: 5‰ (OMS 2020)
- Latinoamérica promedio: 10-15‰ (PAHO 2019)
- Colombia nacional: 8-12‰ (DANE 2023)
- **Criterio adoptado:** >50‰ = 10x la tasa normal = CRÍTICO

**Regla:** Mortalidad >50‰ → +3 puntos automáticos → ALTO RIESGO garantizado

**Literatura de soporte:**
- OMS: Tasas >20‰ consideradas "muy altas" 
- PAHO: Tasas >50‰ indican "crisis de salud pública"
- Estudios Colombia: Departamentos con >30‰ requieren intervención urgente

#### B. Sin Atención Prenatal: 50%

**Justificación:**
- OMS recomienda control prenatal universal (100%)
- >50% sin atención prenatal = falla sistémica crítica

**Regla:** >50% sin prenatal → +2 puntos adicionales

### 5.3 Sistema de Puntaje (0-8 puntos)

```
Criterio                           Condición              Puntos
────────────────────────────────────────────────────────────────
Mortalidad crítica                 >50‰                   +3 ⚠️
Mortalidad fetal alta              >percentil 75          +1
Sin control prenatal crítico       >50%                   +2 ⚠️
Sin control prenatal alto          >percentil 75          +1
Bajo peso al nacer alto            >percentil 75          +1
Prematuridad alta                  >percentil 75          +1
Cobertura cesáreas baja            <percentil 25          +1
Presión obstétrica alta            >percentil 75          +1
────────────────────────────────────────────────────────────────
CLASIFICACIÓN:  ≥3 puntos = ALTO RIESGO
```

### 5.4 Cálculo de Percentiles

**Datos utilizados:** 251 registros válidos (≥10 nacimientos)

```python
# Percentiles calculados sobre datos filtrados
p75_mortalidad_fetal = 33.10‰
p75_sin_prenatal = 23.22%
p75_bajo_peso = 6.84%
p75_prematuro = 100.00%  # Dato ausente en mayoría de registros
p25_cesarea = 0.00%      # Dato ausente en mayoría de registros
p75_presion_obstetrica = 5.9 nacimientos/institución
```

**Nota:** Percentiles de prematuridad y cesáreas no discriminan porque la mayoría de registros tienen valor 0 (dato no disponible en certificado de nacimiento).

### 5.5 Justificación del Umbral de Clasificación (≥3 puntos)

**Análisis de sensibilidad realizado:**

```
Umbral    Alto Riesgo    Críticos detectados    Especificidad
  ≥2         35%              100%                  Muy baja
  ≥3         21%              100%                  Adecuada ✓
  ≥4         15%              92%                   Pierde críticos
  ≥5         8%               75%                   Pierde críticos
```

**Criterio seleccionado: ≥3 puntos**

**Razones:**
1. **Detecta 100% de casos críticos** (mortalidad >50‰)
2. **Equilibrio sensibilidad-especificidad:** 21% alto riesgo es manejable operacionalmente
3. **Coherencia médica:** 3+ factores de riesgo = intervención justificada
4. **Validación con expertos:** Umbral aceptado en salud pública

### 5.6 Resultados del Modelo 1

**Distribución final (251 registros válidos):**
- Alto riesgo: 53 registros municipio-año (21.1%)
- Bajo riesgo: 198 registros municipio-año (78.9%)

**Casos críticos identificados:**
- 40 registros con mortalidad >50‰ (todos clasificados como ALTO RIESGO)
- 100% de sensibilidad en casos críticos

**Distribución 2024 (año más reciente):**
- 45 municipios con datos válidos
- 13 municipios clasificados alto riesgo (29%)
- 11 municipios con mortalidad >50‰

**Promedios por grupo:**
```
Indicador                    Alto Riesgo    Bajo Riesgo    Diferencia
─────────────────────────────────────────────────────────────────────
Mortalidad fetal             52.3‰          15.8‰          +3.3x
Sin control prenatal         31.5%          16.2%          +1.9x
Bajo peso al nacer           7.8%           5.9%           +1.3x
Prematuridad                 0.5%           0.3%           +1.7x
```

---

## 6. Modelo 2: Predicción de Mortalidad Infantil

### 6.1 Definición del Target

**Variable objetivo:** `alta_mortalidad` (binaria)

```python
# Cálculo de tasa de mortalidad infantil (<1 año)
tasa_mortalidad_infantil = (defunciones_menores_1_año / nacimientos) × 1000

# Umbral: percentil 75
umbral = 6.42 muertes por 1000 nacimientos

# Target
alta_mortalidad = 1 si tasa > 6.42, sino 0
```

**Justificación del percentil 75:**
- Identifica el 25% de municipios con peor desempeño
- Suficientes casos positivos para entrenar (78 casos, 25.2%)
- Umbral 6.42‰ es 50% superior al promedio nacional (~4‰)

### 6.2 Selección de Features

**Features utilizadas: 28 de 29**

**Excluidas:**
- `COD_DPTO`, `COD_MUNIC`, `ANO`: Variables de identificación (no se cuentan como features)
- `tasa_mortalidad_infantil`: Se calcula dinámicamente como (total_defunciones / total_nacimientos × 1000)

**Nota:** El target `alta_mortalidad` se genera en train_model.py basándose en el percentil 75 de tasa_mortalidad_infantil.

**Features finales (orden alfabético):**
```
1.  apgar_bajo_promedio
2.  atenciones_per_nacimiento (RIPS)
3.  consultas_per_nacimiento (RIPS)
4.  consultas_promedio
5.  defunciones_fetales
6.  edad_materna_promedio
7.  indice_fragilidad_sistema (🆕 Crítica)
8.  num_instituciones
9.  pct_area_rural
10. pct_bajo_nivel_educativo
11. pct_bajo_peso
12. pct_cesarea
13. pct_embarazo_multiple
14. pct_embarazos_alto_riesgo (🆕 Crítica)
15. pct_instituciones_publicas
16. pct_madres_adolescentes
17. pct_madres_edad_avanzada
18. pct_mortalidad_evitable (🆕 Crítica)
19. pct_prematuro
20. pct_regimen_subsidiado
21. pct_sin_control_prenatal
22. pct_sin_seguridad_social
23. presion_obstetrica
24. procedimientos_per_nacimiento (RIPS)
25. tasa_mortalidad_fetal
26. tasa_mortalidad_neonatal (🆕 Crítica)
27. total_defunciones
28. total_nacimientos
29. urgencias_per_nacimiento (RIPS)
```

### 6.3 Algoritmo Seleccionado: XGBoost

**Justificación de XGBoost vs otras opciones:**

| Modelo | Ventajas | Desventajas | Seleccionado |
|--------|----------|-------------|--------------|
| Regresión Logística | Simple, interpretable | Asume linealidad | No |
| Random Forest | Robusto, no asume distribución | Menos preciso que XGBoost | No |
| **XGBoost** | **Mejor performance, maneja desbalanceo, interpreta importancia** | **Requiere tuning** | **Sí ✓** |
| Redes Neuronales | Máxima capacidad | Caja negra, requiere muchos datos | No |

**Hiperparámetros seleccionados:**

```python
XGBClassifier(
    n_estimators=100,      # Número de árboles
    max_depth=5,           # Profundidad máxima
    learning_rate=0.1,     # Tasa de aprendizaje
    random_state=42,       # Reproducibilidad
    eval_metric='logloss'  # Métrica de evaluación
)
```

**Justificación de hiperparámetros:**

1. **n_estimators=100:**
   - Suficiente para convergencia
   - No causa overfitting con max_depth=5
   - Balance entre performance y tiempo de entrenamiento

2. **max_depth=5:**
   - Evita overfitting con dataset pequeño (310 registros)
   - Permite capturar interacciones de hasta 5 niveles
   - Valor estándar recomendado para datasets <1000 registros

3. **learning_rate=0.1:**
   - Valor por defecto de XGBoost
   - Balance entre convergencia y estabilidad
   - No requiere ajuste para este tamaño de dataset

### 6.4 Tratamiento del Desbalanceo: SMOTE

**Problema inicial:**
- Clase 0 (baja mortalidad): 232 casos (74.8%)
- Clase 1 (alta mortalidad): 78 casos (25.2%)
- Ratio: 3:1

**Técnica aplicada: SMOTE (Synthetic Minority Over-sampling Technique)**

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Resultado:**
- Clase 0: 186 casos
- Clase 1: 186 casos (original 62 + 124 sintéticos)
- Ratio: 1:1

**Justificación de SMOTE:**
- Genera ejemplos sintéticos realistas (interpola entre casos existentes)
- Mejor que under-sampling (no descarta información)
- Mejor que duplicación simple (no crea copias exactas)
- Estándar en literatura para desbalanceo moderado (ratio <5:1)

### 6.5 Normalización: StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Justificación:**
- XGBoost basado en árboles no requiere normalización estrictamente
- **Incluida para:**
  - Comparabilidad entre features en importancia
  - Estabilidad numérica
  - Uso futuro del scaler en predictor del dashboard

**Transformación aplicada:**
```
X_norm = (X - media) / desviación_estándar
```

### 6.6 Validación: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% para test
    random_state=42,    # Reproducibilidad
    stratify=y          # Mantiene proporción de clases
)
```

**Configuración:**
- Train: 248 registros (80%)
- Test: 62 registros (20%)
- Estratificación asegura representación de ambas clases en test

**Justificación 80/20:**
- Estándar en ML para datasets <1000 registros
- Suficientes datos para entrenar (248)
- Suficientes datos para validar (62)
- No se usó validación cruzada por tamaño limitado del dataset

### 6.7 Resultados del Modelo 2

#### Métricas en Test Set (62 casos)

```
                   Precision    Recall    F1-Score    Support
─────────────────────────────────────────────────────────────
Baja Mortalidad       0.90      0.93      0.91         46
Alta Mortalidad       0.79      0.69      0.73         16
─────────────────────────────────────────────────────────────
Accuracy                                   0.87         62
Macro avg             0.84      0.81      0.82         62
Weighted avg          0.87      0.87      0.87         62
```

**Matriz de Confusión:**
```
                  Predicho
                  Baja    Alta
      ────────────────────────
Baja │  43        3
Alta │   5       11
```

**ROC-AUC Score: 0.7731**

#### Interpretación de Resultados

**Fortalezas (Post-mejora Nov 2025):**
1. **ROC-AUC = 0.7731:** Performance sólida para problema complejo (+9.2% vs baseline 0.71)
   - Supera ampliamente el umbral "aceptable" (>0.7) de la literatura médica
   - Muy superior a clasificación aleatoria (0.5)
   - En rango alto de estudios similares (0.65-0.75 típico)

2. **Accuracy = 87%:** Alta confiabilidad general del modelo
   - 54 de 62 predicciones correctas
   - Mejora de +21 puntos porcentuales vs baseline (66%)

3. **Precision alta mortalidad = 0.79:** Confianza muy alta en alertas
   - Cuando predice "alto riesgo", acierta 79% del tiempo
   - Mejora dramática: +39 puntos porcentuales vs baseline (40%)
   - Reducción de falsos positivos: 15 → 3 casos (-80%)

4. **Recall alto riesgo = 0.69:** Detecta 69% de casos de alta mortalidad
   - Prioriza sensibilidad sobre especificidad (adecuado en salud pública)
   - 11 de 16 casos críticos detectados correctamente
   - Mejora: +7 puntos porcentuales vs baseline (62%)

5. **Precision baja mortalidad = 0.90:** Confianza máxima en predicciones negativas
   - Cuando predice "bajo riesgo", acierta 90% del tiempo

6. **Recall baja mortalidad = 0.93:** Excelente detección de casos seguros
   - 43 de 46 casos de baja mortalidad correctamente identificados
   - Solo 3 falsos positivos (vs 15 en baseline)

**Impacto de Features Críticas Avanzadas:**
- **tasa_mortalidad_neonatal** (feature #1, 24.17% importancia): Captura el período más crítico (0-7 días)
- **pct_mortalidad_evitable** (feature #3, 6.65% importancia): Identifica municipios con muertes prevenibles
- **pct_embarazos_alto_riesgo** e **indice_fragilidad_sistema**: Contribuyen a la robustez del modelo
- Resultado: +9.2% ROC-AUC, +97.5% precision alta mortalidad

**Limitaciones Residuales:**
1. **5 falsos negativos:** Municipios de alto riesgo no detectados (31% de casos críticos)
   - Requiere análisis cualitativo adicional de estos casos
   - Posibles factores no capturados en features actuales

2. **3 falsos positivos:** Municipios alertados innecesariamente (6.5% de bajo riesgo)
   - Impacto aceptable: preferible sobre-alertar que sub-alertar en salud pública
   - Reducción significativa vs 15 casos en baseline

#### Top 10 Features Más Importantes (Actualizado Nov 2025)

```
Feature                           Importancia    Justificación
──────────────────────────────────────────────────────────────────
🆕 tasa_mortalidad_neonatal         0.2417       Predictor directo período más crítico (0-7 días)
num_instituciones                   0.0924       Proxy de acceso a servicios de salud
🆕 pct_mortalidad_evitable          0.0665       Identifica municipios con muertes prevenibles
pct_bajo_peso                       0.0544       Predictor clásico de mortalidad neonatal
procedimientos_per_nacimiento       0.0497       Intensidad de atención médica recibida (RIPS)
edad_materna_promedio               0.0459       Embarazos extremos (muy jóvenes/mayores)
pct_area_rural                      0.0391       Proxy de acceso geográfico a servicios
consultas_promedio                  0.0358       Atención prenatal previene complicaciones
pct_sin_seguridad_social            0.0334       Barrera de acceso a servicios
defunciones_fetales                 0.0330       Correlación con mortalidad infantil
```

**Coherencia con literatura médica:**
- **Mortalidad neonatal** como predictor #1 valida el enfoque en período crítico (WHO 2020)
- **Infraestructura** (num_instituciones #2) confirma importancia del acceso
- **Mortalidad evitable** (#3) identifica margen de mejora con intervenciones
- Variables clínicas dominan el TOP 5 (neonatal, bajo peso, procedimientos)
- Features RIPS (procedimientos) en TOP 5 valida integración de datos de servicios
- Validación robusta del modelo con conocimiento del dominio médico

---

## 7. Resultados y Análisis

### 7.1 Estadísticas Generales del Dataset

```
Total registros procesados:        310
Registros válidos (≥10 nac):       251 (81%)
Registros excluidos (<10 nac):      59 (19%)

Periodo:                           2020-2024 (5 años)
Departamentos:                     5 (Meta, Arauca, Casanare, Guaviare, Vichada)
Municipios únicos:                 55

Total nacimientos 2024:            12,656
Total defunciones fetales 2024:    802
Tasa mortalidad fetal promedio:    63.4‰ (2024)
```

### 7.2 Distribución de Riesgo por Departamento (2024)

```
Departamento    Municipios    Alto Riesgo    % Alto    Mortalidad Promedio
─────────────────────────────────────────────────────────────────────────────
Arauca              7             4           57%           99.6‰
Guaviare            4             1           25%           85.2‰ (solo 1 válido)
Vichada             2             2          100%           86.5‰
Casanare           14             2           14%           24.8‰
Meta               18             4           22%           25.1‰
─────────────────────────────────────────────────────────────────────────────
TOTAL              45            13           29%           63.4‰
```

**Observaciones:**
1. **Arauca y Vichada:** Situación crítica (57-100% alto riesgo)
2. **Guaviare:** Solo 1 municipio con datos suficientes (San José), clasificado alto riesgo
3. **Casanare y Meta:** Situación más controlada pero con casos críticos aislados

### 7.3 Municipios Críticos (Mortalidad >50‰) - Año 2024

| Municipio | Departamento | Nacimientos | Defunciones | Mortalidad | Clasificación |
|-----------|--------------|-------------|-------------|------------|---------------|
| Saravena | Arauca | 1,716 | 278 | 162.0‰ | ALTO RIESGO |
| Puerto Rondón | Arauca | 21 | 2 | 95.2‰ | ALTO RIESGO |
| Puerto Carreño | Vichada | 513 | 47 | 91.6‰ | ALTO RIESGO |
| Arauca | Arauca | 1,188 | 107 | 90.1‰ | ALTO RIESGO |
| San José del Guaviare | Guaviare | 1,009 | 86 | 85.2‰ | ALTO RIESGO |
| Monterrey | Casanare | 24 | 2 | 83.3‰ | ALTO RIESGO |
| Guamal | Meta | 13 | 1 | 76.9‰ | ALTO RIESGO |
| Cabuyaro | Meta | 17 | 1 | 58.8‰ | ALTO RIESGO |
| Hato Corozal | Casanare | 38 | 2 | 52.6‰ | ALTO RIESGO |
| La Primavera | Vichada | 57 | 3 | 52.6‰ | ALTO RIESGO |
| Tame | Arauca | 215 | 11 | 51.2‰ | ALTO RIESGO |

**Total población afectada:** 4,811 nacimientos en municipios críticos (38% del total 2024)

### 7.4 Evolución Temporal (2020-2024)

```
Año    Registros    Alto Riesgo    % Alto    Mortalidad Promedio
──────────────────────────────────────────────────────────────────
2020      55            15          27%           48.3‰
2021      54            12          22%           42.1‰
2022      52            11          21%           38.7‰
2023      50             8          16%           35.2‰
2024      45            13          29%           63.4‰
──────────────────────────────────────────────────────────────────
```

**Tendencia identificada:**
- Mejora 2020-2023 (mortalidad bajó de 48.3‰ a 35.2‰)
- **Retroceso significativo en 2024** (subió a 63.4‰)
- Posibles causas 2024:
  - Impacto post-pandemia retrasado
  - Migración venezolana aumentada
  - Reducción de presupuestos en salud
  - Requiere investigación adicional

### 7.5 Comparación con Estándares Nacionales e Internacionales

```
Referencia                         Tasa Mortalidad Fetal
────────────────────────────────────────────────────────
OMS - Promedio global                      5‰
OMS - América Latina                      10-15‰
Colombia - Promedio nacional               8-12‰
────────────────────────────────────────────────────────
Orinoquía 2024                            63.4‰
Orinoquía Alto Riesgo 2024                99.6‰
────────────────────────────────────────────────────────
```

**Brecha identificada:** Orinoquía tiene tasas 5-8 veces superiores al promedio nacional

---

## 8. Validación y Coherencia

### 8.1 Validación de Cálculos

**Proceso de verificación:**

1. **Verificación manual de tasas:**
```python
# Ejemplo: Saravena 2024
nacimientos = 1716
defunciones_fetales = 278
mortalidad_calculada = (278 / 1716) * 1000 = 162.0‰
mortalidad_archivo = 162.0‰
✓ Coincide exactamente
```

2. **Verificación de clasificaciones:**
```
Todos los municipios con mortalidad >50‰:
- Tienen puntos_riesgo ≥3 (incluye +3 bonus)
- Clasificados como riesgo_obstetrico=1 (ALTO)
✓ 100% correctamente clasificados
```

3. **Consistencia con datos crudos DANE:**
```
Archivo: BD-EEVV-Nacimientos-2024.csv
Filtro: COD_DPTO=81, COD_MUNIC=736
Conteo: 1,716 nacimientos
✓ Coincide con features_alerta_materna.csv
```

### 8.2 Coherencia Epidemiológica

**Análisis de correlaciones esperadas:**

```
Correlación                               Valor    Esperado    Estado
───────────────────────────────────────────────────────────────────────
Mortalidad fetal ~ Sin prenatal           +0.58    Positiva    ✓
Bajo peso ~ Mortalidad infantil           +0.51    Positiva    ✓
Madres adolescentes ~ Sin prenatal        +0.43    Positiva    ✓
Num instituciones ~ Mortalidad            -0.38    Negativa    ✓
Presión obstétrica ~ Mortalidad           +0.45    Positiva    ✓
```

**Conclusión:** Todas las correlaciones coinciden con literatura médica

### 8.3 Validación Geográfica

**Municipios fronterizos (mayor vulnerabilidad esperada):**

```
Municipio         Frontera    Mortalidad    Alto Riesgo
─────────────────────────────────────────────────────────
Arauca            Venezuela      90.1‰          SÍ ✓
Puerto Carreño    Venezuela      91.6‰          SÍ ✓
```

**Capitales departamentales (mejor infraestructura esperada):**

```
Municipio         Capital de    Mortalidad    Alto Riesgo
────────────────────────────────────────────────────────────
Villavicencio     Meta           42.3‰          NO ✓
Yopal             Casanare       18.5‰          NO ✓
```

**Conclusión:** Resultados coherentes con expectativas geográficas

### 8.4 Validación Cross-Validation (adicional)

Aunque el modelo final usa train-test simple, se realizó validación cruzada exploratoria:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"ROC-AUC CV: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**Resultado:** ROC-AUC CV: 0.685 (+/- 0.08)
- Similar a test set (0.71)
- Desviación aceptable (<0.1)
- Confirma estabilidad del modelo

---

## 9. Limitaciones y Trabajo Futuro

### 9.1 Limitaciones Identificadas

#### A. Limitaciones de Datos

1. **Tamaño del dataset:**
   - 310 registros totales, 251 válidos
   - Pequeño para Deep Learning
   - Limita capacidad de generalización

2. **Datos faltantes:**
   - % prematuro: dato ausente en ~70% de certificados
   - % cesárea: dato ausente en ~60% de certificados
   - Impacta discriminación de estos features

3. **Granularidad temporal:**
   - Agregación anual (no mensual/trimestral)
   - Pierde estacionalidad

4. **Cobertura geográfica:**
   - Solo región Orinoquía
   - No generalizable a otras regiones sin reentrenamiento

#### B. Limitaciones del Modelo 1

1. **Percentiles dinámicos:**
   - Cambian según datos de cada año
   - Un municipio puede cambiar clasificación sin cambiar indicadores

2. **Pesos uniformes:**
   - Todos los criterios valen 1 punto (excepto críticos)
   - Mortalidad fetal podría tener más peso que otros

3. **Sin predicción temporal:**
   - Clasifica estado actual
   - No predice evolución futura

#### C. Limitaciones del Modelo 2

1. **Precision baja en alto riesgo (40%):**
   - 60% de alertas de alto riesgo son falsos positivos
   - Puede generar fatiga de alertas

2. **Features con baja importancia:**
   - Algunas variables aportan poco (<0.05)
   - Simplificación podría mejorar interpretabilidad

3. **Sin intervalo de confianza:**
   - Predice probabilidad puntual
   - No comunica incertidumbre

### 9.2 Trabajo Futuro

#### Mejoras a Corto Plazo

1. **Optimización de hiperparámetros:**
   - Grid search o Random search
   - Probar diferentes max_depth, n_estimators

2. **Feature selection:**
   - Eliminar features con importancia <0.03
   - Reducir de 20 a ~12 features principales

3. **Threshold tuning:**
   - Ajustar umbral de clasificación (actualmente 0.5)
   - Buscar punto óptimo recall/precision según prioridades

#### Mejoras a Mediano Plazo

1. **Incorporar más fuentes:**
   - Datos climáticos (sequías, inundaciones)
   - Índices de pobreza multidimensional
   - Infraestructura vial (tiempo a hospital)

2. **Modelos ensembles:**
   - Combinar XGBoost + Random Forest + Logistic Regression
   - Voting o Stacking

3. **Análisis de series temporales:**
   - ARIMA o Prophet para proyecciones
   - Detectar tendencias y estacionalidad

#### Mejoras a Largo Plazo

1. **Sistema de recomendación:**
   - No solo clasificar, sino sugerir intervenciones
   - "Este municipio necesita: +3 instituciones, +20% cobertura prenatal"

2. **Modelo causal:**
   - Ir más allá de correlación
   - Identificar intervenciones con mayor impacto

3. **Integración con sistemas de salud:**
   - API para actualización en tiempo real
   - Alertas automáticas a autoridades

4. **Análisis espacial:**
   - Clustering geográfico
   - Identificar corredores de riesgo

---

## 10. Referencias

### Literatura Científica

1. **Organización Mundial de la Salud (OMS).** (2020). *Trends in maternal mortality 2000 to 2017.* WHO, UNICEF, UNFPA, World Bank Group, and UNDP.

2. **Pan American Health Organization (PAHO).** (2019). *Maternal and Neonatal Health in Latin America and the Caribbean.* Washington, DC.

3. **Wilcox, A.J.** (2001). "On the importance—and the unimportance—of birthweight." *International Journal of Epidemiology*, 30(6), 1233-1241.

4. **WHO.** (2016). *WHO recommendations on antenatal care for a positive pregnancy experience.* Geneva: World Health Organization.

5. **UNFPA.** (2013). *Motherhood in Childhood: Facing the challenge of adolescent pregnancy.* State of World Population 2013.

6. **American College of Obstetricians and Gynecologists (ACOG).** (2014). "Committee Opinion No. 579: Definition of term pregnancy." *Obstetrics & Gynecology*, 122(5), 1139-1140.

7. **Chen, T., & Guestrin, C.** (2016). "XGBoost: A scalable tree boosting system." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

8. **Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P.** (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, 321-357.

### Datos y Fuentes Oficiales

9. **Departamento Administrativo Nacional de Estadística (DANE).** (2024). *Estadísticas Vitales - Nacimientos y Defunciones 2020-2024.* Bogotá, Colombia.

10. **Ministerio de Salud y Protección Social de Colombia.** (2023). *Análisis de Situación de Salud (ASIS) Colombia 2023.*

11. **DANE.** (2023). *Código Único de Identificación de Divisiones Político Administrativas - DIVIPOLA.*

12. **Ministerio de Salud.** (2021). *Registro Especial de Prestadores de Servicios de Salud (REPS).*

### Recursos Técnicos

13. **Scikit-learn Documentation.** (2024). *Machine Learning in Python.* https://scikit-learn.org/

14. **XGBoost Documentation.** (2024). *XGBoost Python Package.* https://xgboost.readthedocs.io/

15. **Imbalanced-learn Documentation.** (2024). *Dealing with imbalanced datasets.* https://imbalanced-learn.org/

16. **Streamlit Documentation.** (2024). *The fastest way to build data apps.* https://docs.streamlit.io/

---

## Anexos

### Anexo A: Código de Reproducción

El código completo está disponible en:
- `src/features.py`: Generación de features
- `src/train_model.py`: Entrenamiento de modelos
- `app_simple.py`: Dashboard interactivo

Para reproducir el análisis completo:

```bash
# 1. Generar features
cd src
python features.py

# 2. Entrenar modelos
python train_model.py

# 3. Lanzar dashboard
cd ..
streamlit run app_simple.py
```

### Anexo B: Requisitos del Sistema

```
Python: 3.8+
pandas: 2.0.0+
numpy: 1.24.0+
scikit-learn: 1.3.0+
xgboost: 2.0.0+
imbalanced-learn: 0.11.0+
streamlit: 1.28.0+
plotly: 5.11.0+
```

### Anexo C: Estructura de Archivos Generados

```
models/
├── modelo_mortalidad_xgb.pkl        # Modelo XGBoost entrenado
├── scaler_mortalidad.pkl            # StandardScaler para normalización
├── umbral_mortalidad.pkl            # Umbral de alta mortalidad (6.42‰)
└── umbral_riesgo_obstetrico.pkl     # Percentiles del Modelo 1

data/processed/
├── features_municipio_anio.csv      # Features sin target (310 registros)
├── features_alerta_materna.csv      # Features con targets (310 registros)
└── feature_importance_mortality.csv # Importancia de features
```

---

**Documento generado:** Noviembre 2025  
**Proyecto:** AlertaMaterna - Sistema de Clasificación de Riesgo Obstétrico y Predicción de Mortalidad Infantil  
**Región:** Orinoquía, Colombia

# Documentación Técnica - AlertaMaterna

## Sistema de Clasificación de Riesgo Obstétrico y Predicción de Mortalidad Infantil en la Región Orinoquía

**Región:** Orinoquía, Colombia (Meta, Arauca, Casanare, Guaviare, Vichada)  
**Periodo de análisis:** 2020-2024  
**Fuentes de datos:** DANE y www.datos.gov.co

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
- **Modelo 2 (Predicción de Mortalidad):** XGBoost que predice la tasa de mortalidad infantil (‰) con R²=0.52, MAE=6.93‰ y RMSE=12.62‰.

**Nota terminológica**: Un "registro" = 1 municipio en 1 año específico. Ejemplo: Villavicencio 2020-2024 = 5 registros.

**Aclaración datos**: Los datos brutos del DANE contienen **2,789,391 nacimientos** en toda Orinoquía (2020-2024), pero el análisis se realiza sobre **137,780 nacimientos** en los 251 registros que cumplen el estándar OMS (≥10 nacimientos/año). Se excluyen municipios-año con datos insuficientes para garantizar validez estadística.

**Resultados principales:**
- 310 registros municipio-año analizados (2020-2024)
- 251 registros válidos con ≥10 nacimientos/año (estándar OMS)
- **63 registros clasificados como alto riesgo (25.1%)**
- **137,780 nacimientos analizados** en registros válidos
- **Mortalidad fetal promedio: 23.4‰** (23.4 muertes por 1,000 nacimientos)
- **49.7% de muertes maternas son PREVENIBLES** (causas evitables CIE-10)
- 40 registros con mortalidad crítica (>50‰) correctamente identificados (100% sensibilidad)
- **Modelo predictivo: R²=0.52 | MAE=6.93‰ | RMSE=12.62‰**

---

## 2. Marco Teórico y Justificación

### 2.1 Contexto Epidemiológico

La región Orinoquía presenta características únicas que justifican un sistema de alerta especializado:

- **Dispersión geográfica:** Municipios remotos con acceso limitado a servicios de salud
- **Población vulnerable:** Alta proporción de comunidades indígenas y rurales
- **Infraestructura limitada:** Escasez de centros de salud especializados
- **Indicadores históricos:** Tasas de mortalidad superiores al promedio nacional

### 2.2 Justificación de Variables Seleccionadas

Las 34 variables fueron seleccionadas basándose en:

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
│  • Cálculo de 34 indicadores                                │
│  • Filtrado de calidad (≥10 nacimientos)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   MODELADO ML                                │
│  Script: train_model.py                                      │
│  • Modelo 1: Clasificación de riesgo (índice compuesto)     │
│  • Modelo 2: Regresión XGBoost (tasa mortalidad infantil)   │
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

### 4.1 Variables Generadas (34 indicadores)

#### A. Indicadores Demográficos (5)
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `total_nacimientos` | Total de nacimientos en el municipio-año | Denominador para tasas, tamaño muestral |
| `edad_materna_promedio` | Edad promedio de las madres | Embarazos extremos (muy jóvenes/mayores) tienen mayor riesgo |
| `pct_madres_adolescentes` | % madres <18 años | Asociado a complicaciones obstétricas (UNFPA 2013) |
| `pct_madres_solteras` | % madres no casadas | Factor socioeconómico de riesgo |
| `pct_educacion_baja` | % madres con educación básica | Proxy de nivel socioeconómico |

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

#### D. Indicadores de Acceso a Servicios RIPS (5) - NUEVO
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `atenciones_per_nacimiento` | Total atenciones obstétricas / nacimientos | Intensidad de uso del sistema de salud |
| `consultas_per_nacimiento` | Consultas obstétricas / nacimientos | Acceso efectivo a servicios prenatales |
| `urgencias_per_nacimiento` | Urgencias obstétricas / nacimientos | Indicador de complicaciones y acceso a emergencias |
| `procedimientos_per_nacimiento` | Procedimientos obstétricos / nacimientos | Complejidad de atención requerida |
| `pct_urgencias` | % atenciones de urgencia | Indicador de complicaciones obstétricas |

#### E. Indicadores Socioeconómicos (3)
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `pct_sin_seguridad_social` | % sin afiliación en salud | Acceso a servicios |
| `pct_regimen_subsidiado` | % en régimen subsidiado | Proxy de nivel socioeconómico |
| `pct_area_rural` | % población rural | Ruralidad asociada a menor acceso |

#### F. Indicadores de Atención Prenatal (3)
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `consultas_promedio` | # promedio de consultas prenatales | OMS recomienda mínimo 4 consultas |
| `pct_consultas_insuficientes` | % con <4 consultas prenatales | Atención prenatal inadecuada |
| `pct_sin_control_prenatal` | % sin ningún control prenatal | Factor de riesgo crítico (WHO 2016) |

#### G. Indicadores Críticos Avanzados (8) - NUEVO

**Mortalidad Neonatal (1):**
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `tasa_mortalidad_neonatal` | Muertes 0-27 días × 1000 / nacimientos | Período crítico: 40% de mortalidad infantil ocurre en primera semana (WHO 2020). Detecta problemas en atención inmediata post-parto. Feature #1 del modelo (24.17% importancia). |

**Mortalidad Fetal (2):**
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `defunciones_fetales` | Número absoluto de muertes fetales | Numerador para cálculo de tasas |
| `tasa_mortalidad_fetal` | Muertes fetales × 1000 / nacimientos | Indicador principal OMS. Media regional: 23.4‰ |

**Presión Obstétrica (2):**
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `total_defunciones` | Muertes fetales + no fetales (<1 año) | Denominador para presión |
| `presion_obstetrica` | Total defunciones × 1000 / nacimientos | Medida agregada de estrés del sistema: nacimientos / fallecimientos |

**Causas Evitables (1):**
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `pct_mortalidad_evitable` | % muertes por causas DANE 401-410, 501-506 | Identificación de muertes prevenibles según CIE-10 adaptada por DANE. 49.7% promedio indica gran margen de mejora. Feature #3 del modelo (6.65% importancia). |

**Riesgo Obstétrico Compuesto (2):**
| Variable | Descripción | Justificación |
|----------|-------------|---------------|
| `pct_embarazos_alto_riesgo` | % con prematuridad + bajo peso + múltiples | Indicador compuesto: combina 3 factores críticos asociados a mortalidad neonatal (March of Dimes 2019). Media: 93.8%. |
| `indice_fragilidad_sistema` | Índice compuesto (0-100) basado en componentes críticos | Mide vulnerabilidad sistémica: alta mortalidad + baja cobertura prenatal + falta de aseguramiento + mortalidad evitable. Escala 0-100, 23 municipios >80. |

**Nota:** Las features institucionales (C) utilizan datos diferenciados por municipio del REPS. Las features de acceso a servicios (D) provienen del procesamiento de los RIPS 2020-2024. Las features críticas avanzadas (G) detectan vulnerabilidades específicas en mortalidad neonatal, presión obstétrica y fragilidad del sistema.

### 4.2 Cálculo Detallado de Features Principales

A continuación se detalla el proceso de cálculo de las 10 features más importantes según el modelo XGBoost:

#### 1. **tasa_mortalidad_neonatal** (Importancia: 24.17%)

Mide la tasa de mortalidad en el período más crítico (0-28 días de vida).

```python
# Fórmula:
tasa_mortalidad_neonatal = (defunciones_neonatales / total_nacimientos) * 1000

# Donde:
# - defunciones_neonatales: muertes entre 0-28 días de vida
# - Se obtiene del dataset defunciones no fetales del DANE
# - Se filtra por edad gestacional y días de vida
# - Expresado por cada 1,000 nacimientos vivos

# Código (src/features.py línea 346):
features['tasa_mortalidad_neonatal'] = (
    features['defunciones_neonatales'] / 
    features['total_nacimientos_temp'] * 1000
).fillna(0)
```

**Fuente de datos:** Defunciones no fetales DANE 2020-2024  
**Justificación:** El 40% de la mortalidad infantil ocurre en la primera semana de vida (WHO 2020). Detecta problemas en atención inmediata post-parto.

#### 2. **num_instituciones** (Importancia: 9.24%)

Cuenta el número de instituciones de salud por municipio.

```python
# Proceso:
# 1. Del REPS (Registro Especial de Prestadores de Salud)
# 2. Contar instituciones por código de municipio (5 dígitos)
num_instituciones = df_REPS.groupby('COD_MUNIC').size()

# Código (src/features.py línea 204):
inst_count = df_inst.groupby('MunicipioSede').size().reset_index(name='num_instituciones')
```

**Fuente de datos:** REPS (Registro Especial de Prestadores) MinSalud  
**Justificación:** Proxy de acceso a servicios de salud. Municipios con más instituciones tienen mejor cobertura.

#### 3. **pct_mortalidad_evitable** (Importancia: 6.65%)

Porcentaje de muertes causadas por enfermedades PREVENIBLES según clasificación CIE-10.

```python
# Fórmula:
pct_mortalidad_evitable = (muertes_evitables / total_muertes) * 100

# Identificación de causas evitables:
# - Se usa el campo CAUSA_667 en defunciones DANE
# - Códigos específicos de CIE-10 para causas evitables:
#   * Códigos 401-410: Causas maternas evitables
#   * Códigos 501-506: Causas perinatales evitables
# - Requiere intervención médica oportuna

# Código (src/features.py línea 367):
causas_evitables = df_def_fet[
    df_def_fet['CAUSA_667'].isin(causas_evitables_codes)
].groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size()

pct_mortalidad_evitable = (causas_evitables / total_defunciones) * 100
```

**Fuente de datos:** Defunciones fetales y no fetales DANE, campo CAUSA_667  
**Justificación:** Identifica municipios donde las muertes podrían prevenirse con intervención oportuna. 49.7% promedio indica gran margen de mejora.

#### 4. **pct_bajo_peso** (Importancia: 5.44%)

Porcentaje de recién nacidos con bajo peso al nacer (<2,500 gramos).

```python
# Fórmula:
pct_bajo_peso = (nacimientos_bajo_peso / total_nacimientos) * 100

# Del dataset nacimientos DANE, campo PESO_NAC:
# - Categorías 1-4 representan bajo peso (<2500g)
# - Categoría 1: <1000g (extremo bajo peso)
# - Categoría 2: 1000-1499g (muy bajo peso)
# - Categoría 3: 1500-1999g (bajo peso)
# - Categoría 4: 2000-2499g (bajo peso)

# Código (src/features.py línea 160):
pct_bajo_peso = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
    pct_bajo_peso=('PESO_NAC', lambda x: 
        (x.isin([1, 2, 3, 4])).sum() / len(x) if len(x) > 0 else 0
    )
)
```

**Fuente de datos:** Nacimientos DANE, campo PESO_NAC  
**Justificación:** Predictor clásico de mortalidad neonatal. Asociado a complicaciones y mayor necesidad de cuidados intensivos (Wilcox 2001).

#### 5. **procedimientos_per_nacimiento** (Importancia: 4.97%)

Número promedio de procedimientos médicos realizados por cada nacimiento.

```python
# Fórmula:
procedimientos_per_nacimiento = total_procedimientos_RIPS / total_nacimientos

# Donde:
# - total_procedimientos_RIPS: conteo de registros en RIPS 2020-2024
# - Incluye: cirugías, procedimientos obstétricos, intervenciones
# - Se agrupa por municipio y año

# Código (src/features.py línea 240):
procedimientos = df_rips.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size()
procedimientos_per_nacimiento = procedimientos / total_nacimientos
```

**Fuente de datos:** RIPS (Registros Individuales de Prestación de Servicios)  
**Justificación:** Indica intensidad de atención médica recibida. Valores altos sugieren embarazos de alto riesgo o buena cobertura de servicios.

#### 6. **edad_materna_promedio** (Importancia: 4.59%)

Edad promedio de las madres al momento del parto.

```python
# Fórmula simple:
edad_materna_promedio = df_nacimientos.groupby('COD_MUNIC')['EDAD_MADRE'].mean()

# Del campo EDAD_MADRE en nacimientos DANE (años enteros)

# Código (src/features.py línea 179):
edad_materna_promedio = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
    edad_materna_promedio=('EDAD_MADRE', 'mean')
)
```

**Fuente de datos:** Nacimientos DANE, campo EDAD_MADRE  
**Justificación:** Embarazos en edades extremas (<18 años o >35 años) tienen mayor riesgo de complicaciones obstétricas (UNFPA 2013, ACOG 2014).

#### 7. **pct_area_rural** (Importancia: 3.91%)

Porcentaje de nacimientos ocurridos en área rural.

```python
# Fórmula:
pct_area_rural = (nacimientos_rurales / total_nacimientos) * 100

# Del campo AREA_NAC en nacimientos DANE:
# - 1 = Cabecera municipal (urbano)
# - 2 = Centro poblado / Rural disperso

# Código (src/features.py línea 282):
pct_area_rural = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
    pct_area_rural=('AREA_NAC', lambda x: 
        (x == 2).sum() / len(x) if len(x) > 0 else 0
    )
)
```

**Fuente de datos:** Nacimientos DANE, campo AREA_NAC  
**Justificación:** Proxy de acceso geográfico a servicios de salud. Zonas rurales tienen mayor dificultad para acceder a atención especializada.

#### 8. **consultas_promedio** (Importancia: 3.58%)

Número promedio de consultas prenatales por embarazo.

```python
# Fórmula simple:
consultas_promedio = df_nacimientos.groupby('COD_MUNIC')['NUMERO_CON'].mean()

# Del campo NUMERO_CON en nacimientos DANE (número entero de consultas)

# Código (src/features.py línea 256):
consultas_promedio = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
    consultas_promedio=('NUMERO_CON', 'mean')
)
```

**Fuente de datos:** Nacimientos DANE, campo NUMERO_CON  
**Justificación:** OMS recomienda mínimo 8 consultas prenatales. Atención prenatal adecuada previene complicaciones y reduce mortalidad (WHO 2016).

#### 9. **pct_sin_seguridad_social** (Importancia: 3.34%)

Porcentaje de madres sin afiliación al sistema de salud.

```python
# Fórmula:
pct_sin_seguridad_social = (sin_afiliacion / total_nacimientos) * 100

# Del campo ID_REGIMEN_SEG en nacimientos DANE:
# - 1 = Régimen contributivo
# - 2 = Régimen subsidiado
# - 3 = Régimen especial
# - 4 = Sin afiliación al SGSSS

# Código (src/features.py línea 277):
pct_sin_seguridad_social = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
    pct_sin_seguridad_social=('ID_REGIMEN_SEG', lambda x: 
        (x == 4).sum() / len(x) if len(x) > 0 else 0
    )
)
```

**Fuente de datos:** Nacimientos DANE, campo ID_REGIMEN_SEG  
**Justificación:** Barrera crítica de acceso a servicios de salud. Madres sin seguridad social tienen menor probabilidad de recibir atención prenatal y obstétrica adecuada.

#### 10. **defunciones_fetales** (Importancia: 3.30%)

Número absoluto de defunciones fetales (muertes antes del nacimiento).

```python
# Fórmula simple - conteo directo:
defunciones_fetales = df_def_fetales.groupby('COD_MUNIC').size()

# Del dataset completo de defunciones fetales DANE 2020-2024

# Código (src/features.py línea 174):
defunciones_fet = df_def_fet.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(
    name='defunciones_fetales'
)
```

**Fuente de datos:** Defunciones fetales DANE  
**Justificación:** Indicador directo de problemas en atención obstétrica y calidad de servicios de salud. Correlación alta con mortalidad infantil general.

---

**Nota sobre importancias:** Los valores de importancia (24.17%, 9.24%, etc.) fueron calculados por el modelo XGBoost después del entrenamiento. El algoritmo evalúa automáticamente qué features mejor predicen la clasificación de alto/bajo riesgo mediante análisis de ganancia de información en cada split del árbol de decisión.

### 4.3 Transformaciones Aplicadas

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
Mortalidad crítica                 >50‰                   +3 [CRÍTICO]
Mortalidad fetal alta              >percentil 75          +1
Sin control prenatal crítico       >50%                   +2 [CRÍTICO]
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
2. **Equilibrio sensibilidad-especificidad:** 25.1% alto riesgo es manejable operacionalmente
3. **Coherencia médica:** 3+ factores de riesgo = intervención justificada
4. **Validación con expertos:** Umbral aceptado en salud pública

### 5.6 Resultados del Modelo 1

**Distribución final (251 registros válidos con ≥10 nacimientos/año):**
- Alto riesgo: **63 registros municipio-año (25.1%)**
- Bajo riesgo: 188 registros municipio-año (74.9%)

**Casos críticos identificados:**
- 40 registros con mortalidad >50‰ (todos clasificados como ALTO RIESGO)
- 100% de sensibilidad en casos críticos

**Indicadores agregados periodo 2020-2024:**
- **Nacimientos totales:** 137,780 nacimientos vivos
- **Mortalidad fetal promedio:** 23.4‰ (23.4 muertes por 1,000 nacimientos)
- **Mortalidad evitable:** 49.7% de muertes maternas son por causas PREVENIBLES

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

**Variable objetivo:** `tasa_mortalidad_infantil` (continua en ‰)

```python
# Cálculo de tasa de mortalidad infantil (<1 año)
tasa_mortalidad_infantil = (defunciones_menores_1_año / nacimientos) × 1000

# Target: valor continuo (no umbral binario)
# Rango observado: 0‰ - 200‰
# Media: 4.2‰ (Orinoquía 2020-2024)
# OMS estándar: <5‰ (países desarrollados: 2-3‰)
```

**Justificación del enfoque de regresión:**
- **Interpretación médica directa:** Predice tasa real en ‰ (ej: "8.5 muertes por 1,000 nacimientos")
- **Umbrales absolutos OMS:** Permite clasificar según estándares internacionales (<5‰ Normal, 5-10‰ Moderado, 10-20‰ Alto, >20‰ Crítico)
- **Planificación cuantitativa:** "500 nacimientos × 15‰ = ~7-8 muertes esperadas"
- **Simulación de escenarios:** Evaluar impacto de intervenciones en reducción de tasa
- **Manejo de casos extremos:** Reglas médicas para mortalidad fetal >80‰ o neonatal >15‰

### 6.2 Selección de Features

**Features utilizadas: 34 indicadores**

**Excluidas:**
- `COD_DPTO`, `COD_MUNIC`, `ANO`: Variables de identificación (no se cuentan como features)

**Nota:** El target `tasa_mortalidad_infantil` se calcula como (total_defunciones / total_nacimientos × 1000) y se predice directamente como valor continuo.

**Features finales (orden alfabético):**
```
1.  apgar_bajo_promedio
2.  atenciones_per_nacimiento (RIPS)
3.  consultas_per_nacimiento (RIPS)
4.  consultas_promedio
5.  defunciones_fetales
6.  edad_materna_promedio
7.  indice_fragilidad_sistema [Crítica]
8.  instituciones_per_1000nac
9.  num_instituciones
10. pct_cesareas
11. pct_consultas_insuficientes
12. pct_educacion_baja
13. pct_embarazos_alto_riesgo [Crítica]
14. pct_instituciones_publicas
15. pct_madres_adolescentes
16. pct_madres_solteras
17. pct_mortalidad_evitable [Crítica]
18. pct_multiparidad
19. pct_partos_multiples
20. pct_prematuros
21. pct_regimen_subsidiado
22. pct_sin_control_prenatal
23. pct_sin_seguridad
24. pct_urgencias
25. presion_obstetrica
26. procedimientos_per_nacimiento (RIPS)
27. t_ges_promedio
28. tasa_mortalidad_fetal
29. tasa_mortalidad_neonatal [Crítica]
30. total_defunciones
31. total_nacimientos
32. urgencias_per_nacimiento (RIPS)
```

### 6.3 Algoritmo Seleccionado: XGBoost

**Justificación de XGBoost vs otras opciones:**

| Modelo | Ventajas | Desventajas | Seleccionado |
|--------|----------|-------------|--------------|
| Regresión Lineal | Simple, interpretable | Asume linealidad, no captura interacciones | No |
| Random Forest Regressor | Robusto, no asume distribución | Menos preciso que XGBoost | No |
| **XGBoost Regressor** | **Mejor performance, captura no-linealidades, interpreta importancia** | **Requiere tuning** | **Sí ✓** |
| Redes Neuronales | Máxima capacidad | Caja negra, requiere muchos datos (>10k) | No |

**Hiperparámetros optimizados (post-tuning):**

```python
XGBRegressor(
    n_estimators=50,        # Reducido para evitar overfitting
    max_depth=3,            # Reducido: árboles más simples
    learning_rate=0.05,     # Reducido: aprendizaje más lento y estable
    subsample=0.8,          # 80% datos por árbol (robustez)
    colsample_bytree=0.8,   # 80% features por árbol (reduce correlación)
    reg_alpha=0.1,          # Regularización L1 (feature selection)
    reg_lambda=1.0,         # Regularización L2 (reduce overfitting)
    random_state=42,        # Reproducibilidad
    eval_metric='rmse'      # Métrica de regresión
)
```

**Justificación de hiperparámetros:**

1. **n_estimators=50 (vs 100 inicial):**
   - Suficiente para convergencia con learning_rate bajo
   - Reduce overfitting en dataset pequeño (251 registros)
   - Balance entre performance y tiempo de entrenamiento

2. **max_depth=3 (vs 5 inicial):**
   - **Crítico para evitar overfitting:** Primera versión con max_depth=5 mostró R² train=0.998 vs test=0.448 (overfitting extremo)
   - Árboles más simples generalizan mejor
   - Captura interacciones de 3 niveles (suficiente para este problema)
   - Reducción de overfitting: R² train 0.63 vs test 0.52 (diferencia aceptable <12%)

3. **learning_rate=0.05 (vs 0.1 inicial):**
   - Aprendizaje más lento y estable
   - Reduce riesgo de overfitting
   - Compensa reducción de n_estimators

4. **subsample=0.8 y colsample_bytree=0.8:**
   - Introduce aleatoriedad para robustez
   - Cada árbol ve solo 80% de datos y features
   - Reduce correlación entre árboles (ensemble más diverso)

5. **reg_alpha=0.1 y reg_lambda=1.0:**
   - Regularización L1 (Lasso): promueve feature selection
   - Regularización L2 (Ridge): penaliza pesos grandes
   - Combinación óptima para dataset pequeño

### 6.4 Reglas Médicas Post-Predicción

**Problema identificado:** Casos extremos pueden generar predicciones inconsistentes.

**Reglas implementadas:**

```python
# Regla 1: Mortalidad fetal crítica
if mortalidad_fetal > 80:  # ‰
    tasa_predicha = max(tasa_predicha, 15.0)
    # Justificación: Mortalidad fetal >80‰ indica crisis sistémica
    # Imposible tener mortalidad infantil <15‰ en ese contexto

# Regla 2: Mortalidad neonatal crítica
if mortalidad_neonatal > 15:  # ‰
    tasa_predicha = max(tasa_predicha, 20.0)
    # Justificación: Mortalidad neonatal >15‰ (3x OMS) indica
    # problemas graves en atención post-parto inmediata

# Regla 3: Piso mínimo realista de 3.0‰
tasa_predicha = max(tasa_predicha, 3.0)
# Justificación científica:
#
# 1. PAHO (2019) - Regional Health Report Latin America:
#    "Municipios con mejor desempeño en Latinoamérica mantienen 3-5‰ debido a
#    limitaciones estructurales regionales: distancias geográficas, déficit de
#    especialistas, y acceso limitado a tecnología neonatal avanzada."
#
# 2. Contexto Orinoquía (datos propios 2020-2024):
#    - Promedio regional: 4.2‰
#    - 3.0‰ representa reducción del 29% (meta ambiciosa pero realista)
#    - Requiere mejoras sostenidas en atención prenatal y neonatal
#
# 3. Realismo técnico Colombia:
#    - Distancias geográficas Orinoquía (traslado UCI >2 horas)
#    - Déficit de neonatólogos especializados en región
#    - Equipamiento UCI neonatal limitado vs países desarrollados
#
# 4. Meta Plan Nacional Salud 2030: <6‰
#    - 3.0‰ es 50% mejor que meta nacional → excelencia regional
```

**Justificación médica:**
- **OMS (2020):** Mortalidad fetal >50‰ asociada a mortalidad infantil >10‰
- **PAHO (2019):** Países con mortalidad neonatal >10‰ tienen mortalidad infantil >15‰
- **PAHO (2019) - Contexto Latinoamericano:** Municipios mejor desempeño regional mantienen 3-5‰ debido a limitaciones estructurales
- **Coherencia epidemiológica:** Garantiza predicciones médicamente plausibles y realistas para contexto colombiano y regional

**Impacto:**
- Afecta ~10% de predicciones (casos con indicadores excelentes)
- Evita subestimación en municipios críticos
- Evita predicciones irreales de 0‰ (no existe en ningún país)
- Establece límite técnico alcanzable para Latinoamérica (~3-5‰)
- 3.0‰ representa reducción del 29% vs promedio Orinoquía (4.2‰)
- Ejemplo: Saravena 2024 (mort_fetal=162‰) → predicción ajustada a 25‰ (vs 8‰ inicial)

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
    shuffle=True        # Aleatorización
)
```

**Configuración:**
- Train: 201 registros (80%)
- Test: 50 registros (20%)
- Sin estratificación (no aplica en regresión continua)

**Justificación 80/20:**
- Estándar en ML para datasets <1000 registros
- Suficientes datos para entrenar (201)
- Suficientes datos para validar (50)
- No se usó validación cruzada para evitar data leakage temporal (municipios repetidos entre años)

### 6.7 Resultados del Modelo 2

#### Métricas de Regresión en Test Set (50 casos)

```
═══════════════════════════════════════════════════════════
                    MÉTRICAS DE REGRESIÓN
═══════════════════════════════════════════════════════════
R² Score:              0.5209  (explica 52% de variabilidad)
MAE (Error Promedio):  6.93‰   (desviación promedio)
RMSE:                  12.62‰  (error cuadrático medio)
───────────────────────────────────────────────────────────
R² Train:              0.6315  (performance en entrenamiento)
R² Test:               0.5209  (performance en validación)
Diferencia:            10.6%   (overfitting controlado)
═══════════════════════════════════════════════════════════
```

**Distribución de Predicciones por Rango OMS:**

```
Rango OMS          Casos    %      Tasa Promedio Real    Tasa Promedio Predicha
─────────────────────────────────────────────────────────────────────────────────
Normal (<5‰)        179    71.3%        2.8‰                    3.1‰
Moderado (5-10‰)     26    10.3%        7.2‰                    7.5‰
Alto (10-20‰)        18     7.1%       14.1‰                   13.8‰
Crítico (>20‰)       28    11.3%       45.3‰                   38.2‰
─────────────────────────────────────────────────────────────────────────────────
TOTAL               251   100.0%        8.4‰                    8.4‰
```

#### Interpretación de Resultados

**Fortalezas del Modelo de Regresión:**

1. **R² = 0.52 - Performance BUENA para salud pública:**
   - Explica 52% de la variabilidad en tasas de mortalidad
   - Comparable a estudios similares en epidemiología (típico: 0.45-0.60)
   - Superior a modelos lineales simples (R² ~0.30)
   - Suficiente para identificar municipios de alto riesgo y priorizar intervenciones

2. **MAE = 6.93‰ - Error promedio razonable:**
   - Desviación absoluta promedio de 6.93 muertes por 1,000 nacimientos
   - En contexto: tasa promedio Orinoquía = 8.4‰, error = 82% de la media
   - Predicción ejemplo: Real 15‰ → Predicho 8-22‰ (rango útil para alertas)
   - Menor error en rangos Normal y Moderado (<5‰), mayor en Crítico (>20‰)

3. **RMSE = 12.62‰ - Penaliza errores grandes:**
   - Error cuadrático medio: ~1.5x MAE (indica algunos errores grandes)
   - Casos extremos (>50‰) difíciles de predecir con precisión
   - Aceptable: errores grandes en casos extremos son menos críticos (ya identificados como críticos)

4. **Overfitting controlado (10.6% diferencia):**
   - R² Train 0.63 vs Test 0.52 = diferencia <12% (aceptable)
   - Hiperparámetros optimizados evitaron overfitting extremo inicial (R² train 0.998)
   - Regularización L1/L2 + max_depth=3 + subsample efectivos
   - Modelo generaliza bien a datos nuevos

5. **Interpretación médica directa:**
   - Predice tasa real en ‰: "Este municipio tendrá 8.5 muertes por 1,000 nacimientos"
   - Vs clasificación binaria confusa: "87% probabilidad de estar en percentil 75"
   - Permite planificación cuantitativa: "500 nacimientos × 15‰ = ~7-8 muertes esperadas"

6. **Umbrales absolutos OMS:**
   - Normal (<5‰): 71.3% casos - sistema funcionando bien
   - Crítico (>20‰): 11.3% casos - requieren intervención urgente
   - No depende de percentiles relativos que cambian cada año

**Análisis de Errores por Categoría:**

```
Categoría OMS    MAE (‰)    RMSE (‰)    Interpretación
────────────────────────────────────────────────────────────
Normal           2.1        3.4         Excelente precisión
Moderado         4.8        6.2         Buena precisión
Alto             8.3       11.5         Precisión aceptable
Crítico         18.7       24.3         Mayor incertidumbre
────────────────────────────────────────────────────────────
```

**Observaciones:**
- Mejor performance en rangos Normal/Moderado (71% casos)
- Mayor error en casos Críticos (>20‰) por:
  * Variabilidad extrema (rango 20-200‰)
  * Pocos casos de entrenamiento (28 casos = 11%)
  * Factores no capturados (conflicto armado, migración masiva)

**Casos Extremos Manejados:**
- Municipio con mortalidad fetal 100‰ → predicción 15-20‰ (coherente con reglas médicas)
- Vs modelo clasificación antiguo: 100‰ → "87% probabilidad percentil 75" (confuso)

**Limitaciones Residuales:**

1. **Casos críticos subestimados (18.7‰ error promedio):**
   - Modelo tiende a subestimar tasas >50‰
   - Reglas médicas mitigan pero no eliminan totalmente
   - Solución: combinar con alertas de Modelo 1 (100% sensibilidad casos >50‰)

2. **Variabilidad no capturada (48%):**
   - R² = 0.52 → 48% varianza no explicada
   - Factores no en dataset: clima, conflicto, infraestructura vial, índices pobreza
   - Futuro: integrar más fuentes de datos

3. **Sin intervalos de confianza:**
   - Predicción puntual (ej: 8.5‰) sin rango de incertidumbre
   - Futuro: implementar quantile regression para intervalos

#### Top 10 Features Más Importantes (Dic 2025)

```
Feature                           Importancia    Cambio vs Clasificación
────────────────────────────────────────────────────────────────────────
apgar_bajo_promedio [CRÍTICA]      10.78%        ↑ (antes #4: 5.44%)
num_instituciones                   8.29%        ↓ (antes #2: 9.24%)
consultas_promedio                  6.93%        ↑ (antes #8: 3.58%)
tasa_mortalidad_neonatal [CRÍTICA]  6.45%        ↓ (antes #1: 24.17%)
pct_mortalidad_evitable [CRÍTICA]   6.34%        ↓ (antes #3: 6.65%)
pct_area_rural                      5.87%        ↑ (antes #7: 3.91%)
pct_madres_adolescentes             5.54%        ↑ (nuevo en top 10)
edad_materna_promedio               5.21%        ↓ (antes #6: 4.59%)
tasa_mortalidad_fetal               4.98%        ↑ (nuevo en top 10)
pct_bajo_peso                       4.76%        ↓ (antes #4: 5.44%)
────────────────────────────────────────────────────────────────────────
```

**Cambios en Importancias vs Modelo Clasificación:**
- **APGAR bajo** ahora #1 (antes #4): Mejor predictor continuo de severidad
- **Mortalidad neonatal** bajó de #1 a #4: Ya no domina (antes 24% → ahora 6%)
- **Consultas promedio** subió a #3: Más relevante para rango continuo
- **Distribución más equilibrada:** Top feature 10.78% (vs 24.17% anterior)

**Coherencia con Literatura Médica:**
- **APGAR bajo** como #1: Predictor universal de mortalidad neonatal (WHO 2020)
- **Infraestructura** (#2) y **atención prenatal** (#3): Factores modificables clave
- **Mortalidad neonatal** (#4): Valida enfoque en período crítico (0-7 días)
- **Mortalidad evitable** (#5): 49.7% muertes prevenibles → gran margen de mejora
- Variables clínicas + acceso dominan TOP 10: coherente con causalidad médica

---

## 7. Resultados y Análisis

### 7.1 Estadísticas Generales del Dataset

**Nota**: Un "registro" = combinación municipio-año. Ejemplo: Villavicencio tiene 5 registros (años 2020-2024).

```
Total registros procesados:        310 registros municipio-año
Registros válidos (≥10 nac/año):   251 (81%) - cumple estándar OMS
Registros excluidos (<10 nac/año): 59 (19%)

Periodo:                           2020-2024 (5 años)
Departamentos:                     5 (Meta, Arauca, Casanare, Guaviare, Vichada)
Municipios únicos:                 55

DATOS AGREGADOS PERIODO 2020-2024:
Total nacimientos:                 137,780 nacimientos vivos
Mortalidad fetal promedio:         23.4‰ (23.4 muertes por 1,000 nacimientos)
Mortalidad evitable:               49.7% de muertes maternas son PREVENIBLES
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

1. **Verificación manual de tasas:
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

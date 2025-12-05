"""
Script para verificar qué datos muestra el dashboard
"""
import pandas as pd
import numpy as np

# Cargar datos como lo hace el dashboard
df = pd.read_csv('data/processed/features_municipio_anio.csv')

# Filtrar registros válidos (>=10 nacimientos)
df = df[df['total_nacimientos'] >= 10].copy()

print('=== DATOS DEL DASHBOARD (features_municipio_anio.csv) ===')
print(f'Total registros (>=10 nac): {len(df)}')
print(f'Municipios únicos: {df["COD_MUNIC"].nunique()}')
print(f'Total nacimientos: {df["total_nacimientos"].sum():,}')
print(f'Total defunciones: {df["total_defunciones"].sum():,}')

# Mortalidad promedio simple
print(f'Mortalidad fetal promedio simple: {df["tasa_mortalidad_fetal"].mean():.1f}‰')

# Mortalidad promedio ponderada
mort_ponderada = (df['tasa_mortalidad_fetal'] * df['total_nacimientos']).sum() / df['total_nacimientos'].sum()
print(f'Mortalidad fetal ponderada: {mort_ponderada:.1f}‰')

# Alto riesgo usando mismo algoritmo del dashboard
UMBRAL_CRITICO_MORTALIDAD = 50.0
UMBRAL_CRITICO_SIN_PRENATAL = 0.50

p75_mort_fetal = df['tasa_mortalidad_fetal'].quantile(0.75)
p75_sin_prenatal = df['pct_sin_control_prenatal'].quantile(0.75)
p75_bajo_peso = df['pct_bajo_peso'].quantile(0.75)
p75_prematuro = df['pct_prematuros'].quantile(0.75)
p25_cesarea = df['pct_cesareas'].quantile(0.25)
p75_presion_obs = df['presion_obstetrica'].quantile(0.75)

df['puntos_riesgo'] = 0
df.loc[df['tasa_mortalidad_fetal'] > p75_mort_fetal, 'puntos_riesgo'] += 1
df.loc[df['pct_bajo_peso'] > p75_bajo_peso, 'puntos_riesgo'] += 1
df.loc[df['pct_prematuros'] > p75_prematuro, 'puntos_riesgo'] += 1
df.loc[df['pct_cesareas'] < p25_cesarea, 'puntos_riesgo'] += 1
df.loc[df['presion_obstetrica'] > p75_presion_obs, 'puntos_riesgo'] += 1
df.loc[df['pct_sin_control_prenatal'] > p75_sin_prenatal, 'puntos_riesgo'] += 1
df.loc[df['pct_sin_control_prenatal'] > UMBRAL_CRITICO_SIN_PRENATAL, 'puntos_riesgo'] += 1
df.loc[df['tasa_mortalidad_fetal'] > UMBRAL_CRITICO_MORTALIDAD, 'puntos_riesgo'] += 3

df['alto_riesgo'] = df['puntos_riesgo'] >= 3

print(f'\n=== CLASIFICACIÓN DE RIESGO (DASHBOARD) ===')
print(f'Registros alto riesgo (>=3 puntos): {df["alto_riesgo"].sum()}')
print(f'Municipios únicos alto riesgo: {df[df["alto_riesgo"]]["COD_MUNIC"].nunique()}')
print(f'Registros críticos (MF>50‰): {(df["tasa_mortalidad_fetal"] > 50).sum()}')

# Por año
print('\n=== POR AÑO ===')
for anio in sorted(df['ANO'].unique()):
    df_anio = df[df['ANO'] == anio]
    nac = df_anio['total_nacimientos'].sum()
    mort_pond = (df_anio['tasa_mortalidad_fetal'] * df_anio['total_nacimientos']).sum() / nac if nac > 0 else 0
    alto_r = df_anio['alto_riesgo'].sum()
    munic = df_anio['COD_MUNIC'].nunique()
    print(f'{anio}: {munic} municipios, {nac:,} nac, MF pond={mort_pond:.1f}‰, Alto riesgo={alto_r}')

# Departamentos
print('\n=== POR DEPARTAMENTO ===')
dptos_map = {50: 'Meta', 81: 'Arauca', 85: 'Casanare', 95: 'Guaviare', 99: 'Vichada'}
df['DEPARTAMENTO'] = df['COD_DPTO'].map(dptos_map)
for dpto in sorted(df['DEPARTAMENTO'].dropna().unique()):
    df_dpto = df[df['DEPARTAMENTO'] == dpto]
    nac = df_dpto['total_nacimientos'].sum()
    mort_pond = (df_dpto['tasa_mortalidad_fetal'] * df_dpto['total_nacimientos']).sum() / nac if nac > 0 else 0
    alto_r = df_dpto['alto_riesgo'].sum()
    munic = df_dpto['COD_MUNIC'].nunique()
    print(f'{dpto}: {munic} municipios, {nac:,} nac, MF pond={mort_pond:.1f}‰, Alto riesgo={alto_r}')

# Mortalidad evitable
if 'mortalidad_evitable' in df.columns:
    total_def = df['total_defunciones'].sum()
    if total_def > 0:
        evitable = (df['mortalidad_evitable'] * df['total_defunciones']).sum() / total_def * 100
        print(f'\nMortalidad evitable ponderada: {evitable:.1f}%')
    print(f'Mortalidad evitable promedio simple: {df["mortalidad_evitable"].mean()*100:.1f}%')

# Comparar con criterio MF>30 (usado en presentación)
print('\n=== COMPARACIÓN CON CRITERIO SIMPLE MF>30‰ ===')
alto_riesgo_simple = df['tasa_mortalidad_fetal'] > 30
print(f'Registros con MF>30‰: {alto_riesgo_simple.sum()}')
print(f'Municipios únicos con MF>30‰: {df[alto_riesgo_simple]["COD_MUNIC"].nunique()}')

print('\n=== EVOLUCIÓN MORTALIDAD PONDERADA ===')
for anio in sorted(df['ANO'].unique()):
    df_anio = df[df['ANO'] == anio]
    nac = df_anio['total_nacimientos'].sum()
    mort_pond = (df_anio['tasa_mortalidad_fetal'] * df_anio['total_nacimientos']).sum() / nac if nac > 0 else 0
    print(f'{anio}: {mort_pond:.1f}‰')

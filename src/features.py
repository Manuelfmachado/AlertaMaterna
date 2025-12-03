"""
Script para generar features por municipio-año para AlertaMaterna.

Este script procesa los datos del DANE (nacimientos, defunciones fetales y no fetales)
para la región Orinoquía y genera 24 variables que alimentan los modelos de ML.

Proyecto: AlertaMaterna - Sistema de Clasificación de Riesgo Obstétrico 
          y Predicción de Mortalidad Infantil en la Región Orinoquía
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Departamentos de la Orinoquía
DPTOS_ORINOQUIA = ['50', '81', '85', '95', '99']  # Meta, Arauca, Casanare, Guaviare, Vichada

# Rutas de archivos
DATA_DIR = '../data/processed/'
NACIMIENTOS_FILE = f'{DATA_DIR}nacimientos_2020_2024_decoded.csv'
DEFUNCIONES_FETALES_FILE = f'{DATA_DIR}defunciones_fetales_2020_2024_decoded.csv'
DEFUNCIONES_NO_FETALES_FILE = f'{DATA_DIR}defunciones_no_fetales_2020_2024_decoded.csv'
REPS_FILE = f'{DATA_DIR}Registro_Especial_de_Prestadores_y_Sedes_de_Servicios_de_Salud_20251120.csv'
RIPS_FILE = f'{DATA_DIR}Registros_Individuales_de_Prestación_de_Servicios_de_Salud_–_RIPS_20251120.csv'
OUTPUT_FILE = f'{DATA_DIR}features_municipio_anio.csv'

# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def cargar_nacimientos():
    """Carga y filtra datos de nacimientos de la Orinoquía"""
    print("Cargando nacimientos...")
    df = pd.read_csv(NACIMIENTOS_FILE, low_memory=False)
    
    # Filtrar Orinoquía
    df = df[df['COD_DPTO'].astype(str).isin(DPTOS_ORINOQUIA)].copy()
    
    # Convertir a numéricos
    df['ANO'] = pd.to_numeric(df['ANO'], errors='coerce')
    df['COD_MUNIC'] = pd.to_numeric(df['COD_MUNIC'], errors='coerce')
    df['EDAD_MADRE'] = pd.to_numeric(df['EDAD_MADRE'], errors='coerce')
    df['NUMCONSUL'] = pd.to_numeric(df['NUMCONSUL'], errors='coerce')
    df['PESO_NAC'] = pd.to_numeric(df['PESO_NAC'], errors='coerce')
    df['APGAR1'] = pd.to_numeric(df['APGAR1'], errors='coerce')
    df['APGAR2'] = pd.to_numeric(df['APGAR2'], errors='coerce')
    df['T_GES'] = pd.to_numeric(df['T_GES'], errors='coerce')
    df['MUL_PARTO'] = pd.to_numeric(df['MUL_PARTO'], errors='coerce')
    df['TIPO_PARTO'] = pd.to_numeric(df['TIPO_PARTO'], errors='coerce')
    
    print(f"  → {len(df):,} nacimientos cargados")
    return df

def cargar_defunciones_fetales():
    """Carga y filtra defunciones fetales de la Orinoquía"""
    print("Cargando defunciones fetales...")
    df = pd.read_csv(DEFUNCIONES_FETALES_FILE, low_memory=False)
    
    # Filtrar Orinoquía
    df = df[df['COD_DPTO'].astype(str).isin(DPTOS_ORINOQUIA)].copy()
    
    # Convertir a numéricos
    df['ANO'] = pd.to_numeric(df['ANO'], errors='coerce')
    df['COD_MUNIC'] = pd.to_numeric(df['COD_MUNIC'], errors='coerce')
    
    print(f"  → {len(df):,} defunciones fetales cargadas")
    return df

def cargar_defunciones_no_fetales():
    """Carga y filtra defunciones no fetales (menores de 1 año) de la Orinoquía"""
    print("Cargando defunciones no fetales...")
    df = pd.read_csv(DEFUNCIONES_NO_FETALES_FILE, low_memory=False)
    
    # Filtrar Orinoquía
    df = df[df['COD_DPTO'].astype(str).isin(DPTOS_ORINOQUIA)].copy()
    
    # Convertir a numéricos
    df['ANO'] = pd.to_numeric(df['ANO'], errors='coerce')
    df['COD_MUNIC'] = pd.to_numeric(df['COD_MUNIC'], errors='coerce')
    df['GRU_ED1'] = pd.to_numeric(df['GRU_ED1'], errors='coerce')
    
    # Filtrar menores de 1 año (grupos etarios 1-4 según DANE)
    # 1: < 1 hora, 2: 1-23 horas, 3: 1-6 días, 4: 7-27 días, 5: 28-364 días
    df = df[df['GRU_ED1'].isin([1, 2, 3, 4, 5])].copy()
    
    print(f"  → {len(df):,} defunciones < 1 año cargadas")
    return df

def cargar_instituciones():
    """Carga datos de instituciones de salud por municipio"""
    print("Cargando instituciones de salud...")
    df = pd.read_csv(REPS_FILE, low_memory=False)
    
    # Filtrar Orinoquía
    df['DepartamentoSedeDesc'] = df['DepartamentoSedeDesc'].fillna('')
    orinoquia_names = ['Meta', 'Arauca', 'Casanare', 'Guaviare', 'Vichada']
    df = df[df['DepartamentoSedeDesc'].isin(orinoquia_names)].copy()
    
    print(f"  → {len(df):,} instituciones cargadas")
    return df

def cargar_rips():
    """Carga datos de servicios de salud (RIPS) por municipio-año"""
    print("Cargando servicios de salud (RIPS)...")
    df = pd.read_csv(RIPS_FILE, sep=';', encoding='latin1', low_memory=False)
    
    # Limpiar columna vacía
    if 'Unnamed: 6' in df.columns:
        df = df.drop('Unnamed: 6', axis=1)
    
    # Extraer códigos de departamento y municipio
    df['COD_DPTO'] = df['Departamento'].str.extract(r'^(\d+)')[0]
    df['COD_MUNIC'] = df['Municipio'].str.extract(r'^(\d+)')[0]
    
    # La columna 'Año' está mal codificada como 'AÃ±o' por el encoding
    col_ano = [c for c in df.columns if 'o' in c and len(c) < 5][0]  # Buscar columna del año
    df['ANO'] = pd.to_numeric(df[col_ano].astype(str).str.replace(',', ''), errors='coerce').astype('Int64')
    
    # Filtrar Orinoquía y años 2020-2024
    df = df[df['COD_DPTO'].isin(['50', '81', '85', '95', '99'])].copy()
    df = df[(df['ANO'] >= 2020) & (df['ANO'] <= 2024)].copy()
    
    print(f"  → {len(df):,} registros RIPS Orinoquía 2020-2024 cargados")
    return df

# ============================================================================
# FUNCIONES DE FEATURES - DEMOGRÁFICAS (5)
# ============================================================================

def generar_features_demograficas(df_nac):
    """Genera features demográficas por municipio-año"""
    print("\nGenerando features demográficas...")
    
    # DECODIFICAR EDAD_MADRE de códigos a edades reales (punto medio del rango)
    # Códigos DANE: 1=10-14, 2=15-19, 3=20-24, 4=25-29, 5=30-34, 6=35-39, 7=40-44, 8=45-49, 9=50-54, 99=Sin info
    edad_map = {
        1: 12,   # 10-14 años → punto medio 12
        2: 17,   # 15-19 años → punto medio 17
        3: 22,   # 20-24 años → punto medio 22
        4: 27,   # 25-29 años → punto medio 27
        5: 32,   # 30-34 años → punto medio 32
        6: 37,   # 35-39 años → punto medio 37
        7: 42,   # 40-44 años → punto medio 42
        8: 47,   # 45-49 años → punto medio 47
        9: 52,   # 50-54 años → punto medio 52
        99: np.nan  # Sin información
    }
    
    df_nac_temp = df_nac.copy()
    df_nac_temp['edad_madre_real'] = df_nac_temp['EDAD_MADRE'].map(edad_map)
    
    features = df_nac_temp.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        total_nacimientos=('ANO', 'size'),
        edad_materna_promedio=('edad_madre_real', 'mean'),
        pct_madres_adolescentes=('edad_madre_real', lambda x: (x < 20).sum() / len(x) if len(x) > 0 else 0),
        pct_madres_edad_avanzada=('edad_madre_real', lambda x: (x >= 35).sum() / len(x) if len(x) > 0 else 0),
        pct_bajo_nivel_educativo=('NIV_EDUM', lambda x: (x.isin([1, 2, 3, 4, 99])).sum() / len(x) if len(x) > 0 else 0)
    ).reset_index()
    
    # Rellenar NaN en edad_materna_promedio con la mediana
    features['edad_materna_promedio'] = features['edad_materna_promedio'].fillna(features['edad_materna_promedio'].median())
    
    print(f"  → {len(features)} registros municipio-año generados")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - CLÍNICAS (7)
# ============================================================================

def generar_features_clinicas(df_nac, df_def_fet, df_def_no_fet):
    """Genera features clínicas por municipio-año"""
    print("\nGenerando features clínicas...")
    
    # Features de nacimientos
    features_nac = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        pct_bajo_peso=('PESO_NAC', lambda x: (x.isin([1, 2, 3, 4])).sum() / len(x) if len(x) > 0 else 0),
        pct_embarazo_multiple=('MUL_PARTO', lambda x: (x == 2).sum() / len(x) if len(x) > 0 else 0),
        pct_cesarea=('TIPO_PARTO', lambda x: (x == 2).sum() / len(x) if len(x) > 0 else 0),
        pct_prematuro=('T_GES', lambda x: (x < 5).sum() / len(x) if len(x) > 0 else 0),  # < 37 semanas
        apgar_bajo_promedio=('APGAR1', lambda x: (x < 7).sum() / len(x) if len(x) > 0 else 0),
        total_nacimientos_temp=('ANO', 'size')
    ).reset_index()
    
    # Defunciones no fetales (< 1 año)
    defunciones = df_def_no_fet.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='total_defunciones')
    
    # Defunciones fetales
    defunciones_fet = df_def_fet.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='defunciones_fetales')
    
    # Merge
    features = features_nac.merge(defunciones, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.merge(defunciones_fet, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    
    # Rellenar nulos
    features['total_defunciones'] = features['total_defunciones'].fillna(0)
    features['defunciones_fetales'] = features['defunciones_fetales'].fillna(0)
    
    # Calcular tasa de mortalidad fetal (por 1000 nacimientos)
    features['tasa_mortalidad_fetal'] = (features['defunciones_fetales'] / features['total_nacimientos_temp']) * 1000
    features['tasa_mortalidad_fetal'] = features['tasa_mortalidad_fetal'].fillna(0)
    
    # Eliminar columna temporal
    features = features.drop('total_nacimientos_temp', axis=1)
    
    print(f"  → 7 features clínicas generadas")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - INSTITUCIONALES (3)
# ============================================================================

def generar_features_institucionales(df_nac, df_inst):
    """Genera features institucionales por municipio-año"""
    print("\nGenerando features institucionales...")
    
    # Contar instituciones por municipio (MunicipioSede ya tiene código completo 5 dígitos)
    inst_por_mun = df_inst.groupby('MunicipioSede').agg(
        num_instituciones=('CodigoHabilitacionSede', 'nunique'),
        pct_instituciones_publicas=('NaturalezaJuridica', lambda x: (x == 'Pública').sum() / len(x) if len(x) > 0 else 0)
    ).reset_index()
    inst_por_mun.rename(columns={'MunicipioSede': 'COD_MUNIC_COMPLETO'}, inplace=True)
    inst_por_mun['COD_MUNIC_COMPLETO'] = inst_por_mun['COD_MUNIC_COMPLETO'].astype(int)
    
    # Features de nacimientos por municipio-año
    features = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        total_nacimientos_temp=('ANO', 'size')
    ).reset_index()
    
    # Construir código municipal completo (COD_DPTO + COD_MUNIC)
    features['COD_MUNIC_COMPLETO'] = (features['COD_DPTO'].astype(int) * 1000 + features['COD_MUNIC'].astype(int))
    
    # Merge con instituciones (usar datos reales por municipio)
    features = features.merge(inst_por_mun, on='COD_MUNIC_COMPLETO', how='left')
    
    # Para municipios sin match, usar promedios regionales
    promedio_inst = inst_por_mun['num_instituciones'].mean()
    promedio_pub = inst_por_mun['pct_instituciones_publicas'].mean()
    
    features['num_instituciones'] = features['num_instituciones'].fillna(promedio_inst)
    features['pct_instituciones_publicas'] = features['pct_instituciones_publicas'].fillna(promedio_pub)
    
    # Calcular presión obstétrica
    features['presion_obstetrica'] = features['total_nacimientos_temp'] / features['num_instituciones']
    
    # Eliminar columnas temporales
    features = features.drop(['total_nacimientos_temp', 'COD_MUNIC_COMPLETO'], axis=1)
    
    print(f"  → 3 features institucionales generadas")
    return features
    print(f"  → 3 features institucionales generadas")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - ACCESO A SERVICIOS RIPS (4)
# ============================================================================

def generar_features_rips(df_nac, df_rips):
    """Genera features de acceso a servicios de salud (RIPS) por municipio-año"""
    print("\nGenerando features de acceso a servicios (RIPS)...")
    
    # RIPS ya tiene códigos completos (ej. "50001"), construir COD_MUNIC_COMPLETO
    df_rips['COD_MUNIC_COMPLETO'] = df_rips['COD_MUNIC'].astype(int)
    
    # Limpiar y agrupar RIPS por código completo y año
    rips_agg = df_rips.groupby(['COD_MUNIC_COMPLETO', 'ANO']).agg(
        total_atenciones=('NumeroAtenciones', 'sum'),
        total_consultas=('TipoAtencion', lambda x: (x == 'CONSULTAS').sum()),
        total_urgencias=('TipoAtencion', lambda x: (x == 'URGENCIAS').sum()),
        total_procedimientos=('TipoAtencion', lambda x: (x == 'PROCEDIMIENTOS DE SALUD').sum())
    ).reset_index()
    
    # Features de nacimientos con código completo
    nac_count = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='total_nacimientos_temp')
    nac_count['COD_MUNIC_COMPLETO'] = (nac_count['COD_DPTO'].astype(int) * 1000 + nac_count['COD_MUNIC'].astype(int))
    
    # Merge
    features = nac_count.merge(rips_agg, on=['COD_MUNIC_COMPLETO', 'ANO'], how='left')
    
    # Rellenar nulos (municipios sin datos RIPS)
    features[['total_atenciones', 'total_consultas', 'total_urgencias', 'total_procedimientos']] = \
        features[['total_atenciones', 'total_consultas', 'total_urgencias', 'total_procedimientos']].fillna(0)
    
    # Asegurar tipos numéricos antes de calcular ratios
    features['total_nacimientos_temp'] = features['total_nacimientos_temp'].astype(float)
    features['total_atenciones'] = features['total_atenciones'].astype(float)
    features['total_consultas'] = features['total_consultas'].astype(float)
    features['total_urgencias'] = features['total_urgencias'].astype(float)
    features['total_procedimientos'] = features['total_procedimientos'].astype(float)
    
    # VALIDACIÓN: Reemplazar valores extremos en totales ANTES de calcular ratios
    # Problema identificado: total_atenciones puede tener valores absurdos (ej. 3e+218)
    # que generan ratios >1000 incluso sin división por cero
    max_razonable = 100000  # Máximo razonable de atenciones por municipio-año
    features['total_atenciones'] = features['total_atenciones'].clip(upper=max_razonable)
    features['total_consultas'] = features['total_consultas'].clip(upper=max_razonable)
    features['total_urgencias'] = features['total_urgencias'].clip(upper=max_razonable)
    features['total_procedimientos'] = features['total_procedimientos'].clip(upper=max_razonable)
    
    # Calcular ratios per cápita (por nacimiento)
    # Usar np.where para evitar división por cero
    features['atenciones_per_nacimiento'] = np.where(
        features['total_nacimientos_temp'] > 0,
        features['total_atenciones'] / features['total_nacimientos_temp'],
        0
    )
    features['consultas_per_nacimiento'] = np.where(
        features['total_nacimientos_temp'] > 0,
        features['total_consultas'] / features['total_nacimientos_temp'],
        0
    )
    features['urgencias_per_nacimiento'] = np.where(
        features['total_nacimientos_temp'] > 0,
        features['total_urgencias'] / features['total_nacimientos_temp'],
        0
    )
    features['procedimientos_per_nacimiento'] = np.where(
        features['total_nacimientos_temp'] > 0,
        features['total_procedimientos'] / features['total_nacimientos_temp'],
        0
    )
    
    # VALIDACIÓN FINAL: Clip ratios a valores razonables (máx 500 atenciones/nacimiento)
    # Contexto: OMS recomienda ~8 consultas prenatales, pero puede haber múltiples atenciones
    # 500 es extremadamente alto pero posible en casos con complicaciones graves
    features['atenciones_per_nacimiento'] = features['atenciones_per_nacimiento'].clip(upper=500)
    features['consultas_per_nacimiento'] = features['consultas_per_nacimiento'].clip(upper=100)
    features['urgencias_per_nacimiento'] = features['urgencias_per_nacimiento'].clip(upper=50)
    features['procedimientos_per_nacimiento'] = features['procedimientos_per_nacimiento'].clip(upper=100)
    
    # Reemplazar inf/NaN residuales
    features['atenciones_per_nacimiento'] = features['atenciones_per_nacimiento'].replace([float('inf'), -float('inf')], 0).fillna(0)
    features['consultas_per_nacimiento'] = features['consultas_per_nacimiento'].replace([float('inf'), -float('inf')], 0).fillna(0)
    features['urgencias_per_nacimiento'] = features['urgencias_per_nacimiento'].replace([float('inf'), -float('inf')], 0).fillna(0)
    features['procedimientos_per_nacimiento'] = features['procedimientos_per_nacimiento'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # Eliminar columnas temporales
    features = features.drop(['total_nacimientos_temp', 'total_atenciones', 'total_consultas', 
                              'total_urgencias', 'total_procedimientos', 'COD_MUNIC_COMPLETO'], axis=1)
    
    print(f"  → 4 features de acceso a servicios generadas")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - SOCIOECONÓMICAS (3)
# ============================================================================

def generar_features_socioeconomicas(df_nac):
    """Genera features socioeconómicas por municipio-año"""
    print("\nGenerando features socioeconómicas...")
    
    features = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        pct_sin_seguridad_social=('SEG_SOCIAL', lambda x: (x.isin([3, 4, 9])).sum() / len(x) if len(x) > 0 else 0),
        pct_area_rural=('AREANAC', lambda x: (x.isin([2, 3])).sum() / len(x) if len(x) > 0 else 0),
        pct_regimen_subsidiado=('SEG_SOCIAL', lambda x: (x == 2).sum() / len(x) if len(x) > 0 else 0)
    ).reset_index()
    
    print(f"  → 3 features socioeconómicas generadas")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - ATENCIÓN PRENATAL (2)
# ============================================================================

def generar_features_atencion_prenatal(df_nac):
    """Genera features de atención prenatal por municipio-año"""
    print("\nGenerando features de atención prenatal...")
    
    features = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        pct_sin_control_prenatal=('NUMCONSUL', lambda x: (x.isin([0, 99])).sum() / len(x) if len(x) > 0 else 0),
        consultas_promedio=('NUMCONSUL', lambda x: x[~x.isin([99])].mean() if len(x[~x.isin([99])]) > 0 else 0)
    ).reset_index()
    
    print(f"  → 3 features prenatales generadas")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - CRÍTICAS AVANZADAS (4)
# ============================================================================

def generar_features_mortalidad_neonatal(df_nac, df_def_nofet):
    """
    Genera features de mortalidad neonatal temprana (0-7 días).
    Indicador clave OMS de calidad de atención perinatal.
    """
    print("\nGenerando features de mortalidad neonatal...")
    
    # Contar nacimientos por municipio-año
    nac_count = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='total_nacimientos_temp')
    
    # Filtrar defunciones neonatales tempranas (0-7 días): GRU_ED1 = 1, 2, 3
    # 1: < 1 hora, 2: 1-23 horas, 3: 1-6 días
    df_neonatal = df_def_nofet[df_def_nofet['GRU_ED1'].isin([1, 2, 3])].copy()
    
    # Contar defunciones neonatales por municipio-año
    def_neonatal = df_neonatal.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='defunciones_neonatales')
    
    # Merge
    features = nac_count.merge(def_neonatal, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features['defunciones_neonatales'] = features['defunciones_neonatales'].fillna(0)
    
    # Calcular tasa por 1000 nacidos vivos
    features['tasa_mortalidad_neonatal'] = (features['defunciones_neonatales'] / features['total_nacimientos_temp'] * 1000).fillna(0)
    
    # Limpiar columnas temporales
    features = features[['COD_DPTO', 'COD_MUNIC', 'ANO', 'tasa_mortalidad_neonatal']]
    
    print(f"  → 1 feature de mortalidad neonatal generada")
    print(f"     Media nacional: {features['tasa_mortalidad_neonatal'].mean():.2f} por 1000 nacidos vivos")
    return features

def generar_features_causas_evitables(df_def_fet, df_def_nofet, df_nac):
    """
    Genera features de mortalidad por causas evitables.
    Basado en clasificación CIE-10 de causas evitables (usa CAUSA_667).
    Requiere df_nac para crear el esqueleto de todos los municipios-años.
    """
    print("\nGenerando features de causas evitables...")
    
    # Códigos 667 de causas EVITABLES según clasificación DANE
    # 401-410: Causas obstétricas directas evitables
    # 501-506: Causas perinatales evitables
    causas_evitables_667 = list(range(401, 411)) + list(range(501, 507))
    
    # Combinar defunciones fetales y no fetales
    def_fet_causas = df_def_fet[['COD_DPTO', 'COD_MUNIC', 'ANO', 'CAUSA_667']].copy()
    def_nofet_causas = df_def_nofet[['COD_DPTO', 'COD_MUNIC', 'ANO', 'CAUSA_667']].copy() if 'CAUSA_667' in df_def_nofet.columns else pd.DataFrame()
    
    todas_def = pd.concat([def_fet_causas, def_nofet_causas], ignore_index=True)
    
    # Identificar causas evitables
    todas_def['CAUSA_667'] = pd.to_numeric(todas_def['CAUSA_667'], errors='coerce')
    todas_def['es_evitable'] = todas_def['CAUSA_667'].isin(causas_evitables_667).astype(int)
    
    # Agrupar por municipio-año
    features_temp = todas_def.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        total_defunciones_temp=('ANO', 'size'),
        defunciones_evitables=('es_evitable', 'sum')
    ).reset_index()
    
    # Calcular proporción con validación
    features_temp['pct_mortalidad_evitable'] = np.where(
        features_temp['total_defunciones_temp'] > 0,
        (features_temp['defunciones_evitables'] / features_temp['total_defunciones_temp'] * 100),
        0
    )
    
    # VALIDACIÓN: Clip entre 0 y 100%
    features_temp['pct_mortalidad_evitable'] = features_temp['pct_mortalidad_evitable'].clip(0, 100)
    
    # Crear esqueleto con TODOS los municipios-años (desde nacimientos)
    esqueleto = df_nac[['COD_DPTO', 'COD_MUNIC', 'ANO']].drop_duplicates().reset_index(drop=True)
    
    # Merge con esqueleto (left join para mantener todos los municipios)
    features = esqueleto.merge(
        features_temp[['COD_DPTO', 'COD_MUNIC', 'ANO', 'pct_mortalidad_evitable']], 
        on=['COD_DPTO', 'COD_MUNIC', 'ANO'], 
        how='left'
    )
    
    # Rellenar NaN con 0 (municipios sin defunciones = 0% evitable)
    features['pct_mortalidad_evitable'] = features['pct_mortalidad_evitable'].fillna(0)
    
    print(f"  → 1 feature de causas evitables generada")
    print(f"     Media: {features['pct_mortalidad_evitable'].mean():.1f}% de muertes evitables")
    print(f"     Municipios con datos: {(features['pct_mortalidad_evitable'] > 0).sum()}/{len(features)}")
    return features

def generar_features_embarazo_alto_riesgo(df_nac):
    """
    Genera features de embarazos de alto riesgo clínico.
    Combina prematuridad, bajo peso y embarazo múltiple.
    T_GES: 1=<22sem, 2=22-27, 3=28-31, 4=32-36, 5=37-41, 6=42+
    PESO_NAC: 1=<500g, 2=500-999, 3=1000-1499, 4=1500-1999, 5=2000-2499, 6=2500-2999, 7=3000-3499, 8=3500-3999, 9=4000+
    """
    print("\nGenerando features de embarazo alto riesgo...")
    
    # Identificar embarazos de alto riesgo
    df_nac_temp = df_nac.copy()
    
    # T_GES codificado: Prematuro = 1, 2, 3, 4 (< 37 semanas)
    df_nac_temp['T_GES'] = pd.to_numeric(df_nac_temp['T_GES'], errors='coerce')
    df_nac_temp['es_prematuro'] = (df_nac_temp['T_GES'].isin([1, 2, 3, 4])).astype(int)
    
    # PESO_NAC codificado: Bajo peso = 1, 2, 3, 4, 5 (< 2500g)
    df_nac_temp['PESO_NAC'] = pd.to_numeric(df_nac_temp['PESO_NAC'], errors='coerce')
    df_nac_temp['es_bajo_peso'] = (df_nac_temp['PESO_NAC'].isin([1, 2, 3, 4, 5])).astype(int)
    
    # Embarazo múltiple (MUL_PARTO > 1)
    df_nac_temp['MUL_PARTO'] = pd.to_numeric(df_nac_temp['MUL_PARTO'], errors='coerce')
    df_nac_temp['es_multiple'] = (df_nac_temp['MUL_PARTO'] > 1).fillna(0).astype(int)
    
    # Un embarazo es de alto riesgo si cumple AL MENOS UNA condición
    df_nac_temp['es_alto_riesgo'] = (
        (df_nac_temp['es_prematuro'] == 1) | 
        (df_nac_temp['es_bajo_peso'] == 1) | 
        (df_nac_temp['es_multiple'] == 1)
    ).astype(int)
    
    # Agrupar por municipio-año
    features = df_nac_temp.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        total_nacimientos_temp=('ANO', 'size'),
        embarazos_alto_riesgo=('es_alto_riesgo', 'sum')
    ).reset_index()
    
    # Calcular proporción con validación
    features['pct_embarazos_alto_riesgo'] = np.where(
        features['total_nacimientos_temp'] > 0,
        (features['embarazos_alto_riesgo'] / features['total_nacimientos_temp'] * 100),
        0
    )
    
    # VALIDACIÓN: Clip entre 0 y 100% (algunos cálculos pueden dar >100 por errores de datos)
    features['pct_embarazos_alto_riesgo'] = features['pct_embarazos_alto_riesgo'].clip(0, 100).fillna(0)
    
    # Limpiar columnas temporales
    features = features[['COD_DPTO', 'COD_MUNIC', 'ANO', 'pct_embarazos_alto_riesgo']]
    
    print(f"  → 1 feature de embarazo alto riesgo generada")
    print(f"     Media: {features['pct_embarazos_alto_riesgo'].mean():.1f}% embarazos alto riesgo")
    return features

def generar_features_fragilidad_sistema(feat_demograficas, feat_institucionales, feat_mortalidad_neonatal):
    """
    Genera índice de fragilidad del sistema de salud.
    Combina mortalidad neonatal, presión obstétrica y densidad institucional.
    Fórmula: (mortalidad_neonatal × presion_obstetrica) / (num_instituciones + 1)
    """
    print("\nGenerando índice de fragilidad del sistema...")
    
    # Merge de las tres fuentes
    features = feat_demograficas[['COD_DPTO', 'COD_MUNIC', 'ANO', 'total_nacimientos']].copy()
    
    # Agregar presión obstétrica y num_instituciones
    features = features.merge(
        feat_institucionales[['COD_DPTO', 'COD_MUNIC', 'ANO', 'presion_obstetrica', 'num_instituciones']], 
        on=['COD_DPTO', 'COD_MUNIC', 'ANO'], 
        how='left'
    )
    
    # Agregar mortalidad neonatal
    features = features.merge(
        feat_mortalidad_neonatal[['COD_DPTO', 'COD_MUNIC', 'ANO', 'tasa_mortalidad_neonatal']], 
        on=['COD_DPTO', 'COD_MUNIC', 'ANO'], 
        how='left'
    )
    
    # Rellenar nulos
    features['presion_obstetrica'] = features['presion_obstetrica'].fillna(features['presion_obstetrica'].mean())
    features['num_instituciones'] = features['num_instituciones'].fillna(1)
    features['tasa_mortalidad_neonatal'] = features['tasa_mortalidad_neonatal'].fillna(0)
    
    # Calcular densidad institucional (instituciones per capita, ajustado por 1000 nacimientos)
    # VALIDACIÓN: Evitar división por cero
    features['densidad_institucional'] = np.where(
        features['total_nacimientos'] > 0,
        features['num_instituciones'] / (features['total_nacimientos'] / 1000 + 1),
        0
    )
    
    # Calcular índice de fragilidad
    # Normalizar presión obstétrica para evitar valores extremos
    presion_norm = features['presion_obstetrica'] / features['presion_obstetrica'].quantile(0.75)
    presion_norm = presion_norm.clip(upper=3)  # Cap en 3x el percentil 75
    
    # VALIDACIÓN: Evitar división por cero en índice de fragilidad
    features['indice_fragilidad_sistema'] = np.where(
        features['densidad_institucional'] > 0,
        (features['tasa_mortalidad_neonatal'] * presion_norm / (features['densidad_institucional'] + 0.1)),
        0
    )
    
    # VALIDACIÓN: Reemplazar inf/NaN antes de normalizar
    features['indice_fragilidad_sistema'] = features['indice_fragilidad_sistema'].replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # Normalizar a escala 0-100
    max_fragilidad = features['indice_fragilidad_sistema'].quantile(0.95)
    if max_fragilidad > 0:
        features['indice_fragilidad_sistema'] = (features['indice_fragilidad_sistema'] / max_fragilidad * 100).clip(0, 100)
    else:
        features['indice_fragilidad_sistema'] = 0
    
    # Limpiar columnas temporales
    features = features[['COD_DPTO', 'COD_MUNIC', 'ANO', 'indice_fragilidad_sistema']]
    
    print(f"  → 1 feature de fragilidad del sistema generada")
    print(f"     Media: {features['indice_fragilidad_sistema'].mean():.1f} (escala 0-100)")
    print(f"     Municipios críticos (>80): {(features['indice_fragilidad_sistema'] > 80).sum()}")
    return features

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que orquesta la generación de features"""
    print("=" * 80)
    print("GENERACIÓN DE FEATURES - ALERTAMATERNA")
    print("=" * 80)
    
    # 1. Cargar datos
    df_nacimientos = cargar_nacimientos()
    df_def_fetales = cargar_defunciones_fetales()
    df_def_no_fetales = cargar_defunciones_no_fetales()
    df_instituciones = cargar_instituciones()
    df_rips = cargar_rips()
    
    # 2. Generar features por categoría
    feat_demograficas = generar_features_demograficas(df_nacimientos)
    feat_clinicas = generar_features_clinicas(df_nacimientos, df_def_fetales, df_def_no_fetales)
    feat_institucionales = generar_features_institucionales(df_nacimientos, df_instituciones)
    feat_rips = generar_features_rips(df_nacimientos, df_rips)
    feat_socioeconomicas = generar_features_socioeconomicas(df_nacimientos)
    feat_prenatal = generar_features_atencion_prenatal(df_nacimientos)
    
    # 2.1. Generar features CRÍTICAS AVANZADAS
    print("\n" + "=" * 80)
    print("GENERANDO FEATURES CRÍTICAS AVANZADAS")
    print("=" * 80)
    feat_mortalidad_neonatal = generar_features_mortalidad_neonatal(df_nacimientos, df_def_no_fetales)
    feat_causas_evitables = generar_features_causas_evitables(df_def_fetales, df_def_no_fetales, df_nacimientos)
    feat_embarazo_riesgo = generar_features_embarazo_alto_riesgo(df_nacimientos)
    
    # 3. Normalizar tipos de las columnas de merge
    print("\nNormalizando tipos de datos...")
    for df in [feat_demograficas, feat_clinicas, feat_institucionales, feat_rips, feat_socioeconomicas, feat_prenatal,
               feat_mortalidad_neonatal, feat_causas_evitables, feat_embarazo_riesgo]:
        df['COD_DPTO'] = df['COD_DPTO'].astype(str)
        df['COD_MUNIC'] = df['COD_MUNIC'].astype(str).str.zfill(5)
        df['ANO'] = df['ANO'].astype(int)
    
    # 4. Combinar todas las features (incluyendo las básicas primero)
    print("Combinando features básicas...")
    features_final = feat_demograficas
    features_final = features_final.merge(feat_clinicas, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features_final = features_final.merge(feat_institucionales, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features_final = features_final.merge(feat_rips, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features_final = features_final.merge(feat_socioeconomicas, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features_final = features_final.merge(feat_prenatal, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    
    # 4.1. Combinar features críticas avanzadas
    print("Combinando features críticas...")
    features_final = features_final.merge(feat_mortalidad_neonatal, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features_final = features_final.merge(feat_causas_evitables, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features_final = features_final.merge(feat_embarazo_riesgo, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    
    # 4.2. Generar índice de fragilidad (requiere features previas combinadas)
    print("Generando índice de fragilidad del sistema...")
    feat_fragilidad = generar_features_fragilidad_sistema(feat_demograficas, feat_institucionales, feat_mortalidad_neonatal)
    
    # Normalizar tipos
    feat_fragilidad['COD_DPTO'] = feat_fragilidad['COD_DPTO'].astype(str)
    feat_fragilidad['COD_MUNIC'] = feat_fragilidad['COD_MUNIC'].astype(str).str.zfill(5)
    feat_fragilidad['ANO'] = feat_fragilidad['ANO'].astype(int)
    
    # Merge final
    features_final = features_final.merge(feat_fragilidad, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    
    # 5. Reordenar columnas
    columnas_id = ['COD_DPTO', 'COD_MUNIC', 'ANO']
    columnas_features = [col for col in features_final.columns if col not in columnas_id]
    features_final = features_final[columnas_id + sorted(columnas_features)]
    
    # 5. Guardar
    features_final.to_csv(OUTPUT_FILE, index=False)
    
    # 6. Resumen
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"Total de registros: {len(features_final):,}")
    print(f"Total de features: {len(features_final.columns) - 3}")  # -3 por las columnas de ID
    print(f"  • Features básicas: 21")
    print(f"  • Features críticas nuevas: 4")
    print(f"    - tasa_mortalidad_neonatal")
    print(f"    - pct_mortalidad_evitable")
    print(f"    - pct_embarazos_alto_riesgo")
    print(f"    - indice_fragilidad_sistema")
    print(f"Años: {sorted(features_final['ANO'].unique())}")
    print(f"Departamentos: {sorted(features_final['COD_DPTO'].unique())}")
    print(f"Municipios únicos: {features_final['COD_MUNIC'].nunique()}")
    print(f"\nArchivo guardado en: {OUTPUT_FILE}")
    
    # 7. Estadísticas de features críticas
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS DE FEATURES CRÍTICAS")
    print("=" * 80)
    
    if 'tasa_mortalidad_neonatal' in features_final.columns:
        print(f"\n Mortalidad Neonatal:")
        print(f"   Media: {features_final['tasa_mortalidad_neonatal'].mean():.2f} por 1000 nacidos vivos")
        print(f"   Rango: {features_final['tasa_mortalidad_neonatal'].min():.2f} - {features_final['tasa_mortalidad_neonatal'].max():.2f}")
        print(f"   Municipios con tasa >15: {(features_final['tasa_mortalidad_neonatal'] > 15).sum()}")
    
    if 'pct_mortalidad_evitable' in features_final.columns:
        print(f"\n Mortalidad Evitable:")
        print(f"   Media: {features_final['pct_mortalidad_evitable'].mean():.1f}%")
        print(f"   Municipios con >50% evitable: {(features_final['pct_mortalidad_evitable'] > 50).sum()}")
    
    if 'pct_embarazos_alto_riesgo' in features_final.columns:
        print(f"\n Embarazos Alto Riesgo:")
        print(f"   Media: {features_final['pct_embarazos_alto_riesgo'].mean():.1f}%")
        print(f"   Municipios con >30% alto riesgo: {(features_final['pct_embarazos_alto_riesgo'] > 30).sum()}")
    
    if 'indice_fragilidad_sistema' in features_final.columns:
        print(f"\n Índice de Fragilidad:")
        print(f"   Media: {features_final['indice_fragilidad_sistema'].mean():.1f}")
        print(f"   Municipios críticos (>80): {(features_final['indice_fragilidad_sistema'] > 80).sum()}")
        print(f"   Municipios muy críticos (>90): {(features_final['indice_fragilidad_sistema'] > 90).sum()}")
    
    # 7. Mostrar primeras filas
    print("\nPrimeras filas:")
    print(features_final.head())
    
    print("\nProceso completado exitosamente!")

if __name__ == "__main__":
    main()

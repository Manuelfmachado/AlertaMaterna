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
    
    features = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        total_nacimientos=('ANO', 'size'),
        edad_materna_promedio=('EDAD_MADRE', 'mean'),
        pct_madres_adolescentes=('EDAD_MADRE', lambda x: (x < 20).sum() / len(x)),
        pct_madres_edad_avanzada=('EDAD_MADRE', lambda x: (x >= 35).sum() / len(x)),
        pct_bajo_nivel_educativo=('NIV_EDUM', lambda x: (x.isin([1, 2, 3, 4, 99])).sum() / len(x))
    ).reset_index()
    
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
    
    # Extraer código de municipio del REPS (asegurar formato string)
    df_inst['COD_MUNIC'] = df_inst['MunicipioSede'].astype(str).str.zfill(5)
    
    # Contar instituciones por municipio
    inst_por_mun = df_inst.groupby('COD_MUNIC').agg(
        num_instituciones=('CodigoHabilitacionSede', 'nunique'),
        pct_instituciones_publicas=('NaturalezaJuridica', lambda x: (x == 'Pública').sum() / len(x) if len(x) > 0 else 0)
    ).reset_index()
    
    # Features de nacimientos por municipio-año (asegurar COD_MUNIC como string)
    features = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        total_nacimientos_temp=('ANO', 'size')
    ).reset_index()
    features['COD_MUNIC'] = features['COD_MUNIC'].astype(str).str.zfill(5)
    
    # Merge con instituciones (usar datos reales por municipio)
    features = features.merge(inst_por_mun, on='COD_MUNIC', how='left')
    
    # Para municipios sin match, usar promedios regionales
    promedio_inst = inst_por_mun['num_instituciones'].mean()
    promedio_pub = inst_por_mun['pct_instituciones_publicas'].mean()
    
    features['num_instituciones'] = features['num_instituciones'].fillna(promedio_inst)
    features['pct_instituciones_publicas'] = features['pct_instituciones_publicas'].fillna(promedio_pub)
    
    # Calcular presión obstétrica
    features['presion_obstetrica'] = features['total_nacimientos_temp'] / features['num_instituciones']
    
    # Eliminar columna temporal
    features = features.drop('total_nacimientos_temp', axis=1)
    
    print(f"  → 3 features institucionales generadas")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - ACCESO A SERVICIOS RIPS (4)
# ============================================================================

def generar_features_rips(df_nac, df_rips):
    """Genera features de acceso a servicios de salud (RIPS) por municipio-año"""
    print("\nGenerando features de acceso a servicios (RIPS)...")
    
    # Asegurar tipos consistentes
    df_rips['COD_DPTO'] = df_rips['COD_DPTO'].astype(str)
    df_rips['COD_MUNIC'] = df_rips['COD_MUNIC'].astype(str).str.zfill(5)
    
    # Limpiar y agrupar RIPS por municipio-año
    rips_agg = df_rips.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        total_atenciones=('NumeroAtenciones', 'sum'),
        total_consultas=('TipoAtencion', lambda x: (x == 'CONSULTAS').sum()),
        total_urgencias=('TipoAtencion', lambda x: (x == 'URGENCIAS').sum()),
        total_procedimientos=('TipoAtencion', lambda x: (x == 'PROCEDIMIENTOS DE SALUD').sum())
    ).reset_index()
    
    # Features de nacimientos con tipos consistentes
    nac_count = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='total_nacimientos_temp')
    nac_count['COD_DPTO'] = nac_count['COD_DPTO'].astype(str)
    nac_count['COD_MUNIC'] = nac_count['COD_MUNIC'].astype(str).str.zfill(5)
    
    # Merge
    features = nac_count.merge(rips_agg, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    
    # Rellenar nulos (municipios sin datos RIPS)
    features[['total_atenciones', 'total_consultas', 'total_urgencias', 'total_procedimientos']] = \
        features[['total_atenciones', 'total_consultas', 'total_urgencias', 'total_procedimientos']].fillna(0)
    
    # Calcular ratios per cápita (por nacimiento)
    features['atenciones_per_nacimiento'] = features['total_atenciones'] / features['total_nacimientos_temp']
    features['consultas_per_nacimiento'] = features['total_consultas'] / features['total_nacimientos_temp']
    features['urgencias_per_nacimiento'] = features['total_urgencias'] / features['total_nacimientos_temp']
    features['procedimientos_per_nacimiento'] = features['total_procedimientos'] / features['total_nacimientos_temp']
    
    # Eliminar columnas temporales
    features = features.drop(['total_nacimientos_temp', 'total_atenciones', 'total_consultas', 
                              'total_urgencias', 'total_procedimientos'], axis=1)
    
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
    
    print(f"  → 2 features de atención prenatal generadas")
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
    
    # 3. Normalizar tipos de las columnas de merge
    print("\nNormalizando tipos de datos...")
    for df in [feat_demograficas, feat_clinicas, feat_institucionales, feat_rips, feat_socioeconomicas, feat_prenatal]:
        df['COD_DPTO'] = df['COD_DPTO'].astype(str)
        df['COD_MUNIC'] = df['COD_MUNIC'].astype(str).str.zfill(5)
        df['ANO'] = df['ANO'].astype(int)
    
    # 4. Combinar todas las features
    print("Combinando features...")
    features_final = feat_demograficas
    features_final = features_final.merge(feat_clinicas, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features_final = features_final.merge(feat_institucionales, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features_final = features_final.merge(feat_rips, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features_final = features_final.merge(feat_socioeconomicas, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features_final = features_final.merge(feat_prenatal, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    
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
    print(f"Años: {sorted(features_final['ANO'].unique())}")
    print(f"Departamentos: {sorted(features_final['COD_DPTO'].unique())}")
    print(f"Municipios únicos: {features_final['COD_MUNIC'].nunique()}")
    print(f"\nArchivo guardado en: {OUTPUT_FILE}")
    
    # 7. Mostrar primeras filas
    print("\nPrimeras filas:")
    print(features_final.head())
    
    print("\nProceso completado exitosamente!")

if __name__ == "__main__":
    main()

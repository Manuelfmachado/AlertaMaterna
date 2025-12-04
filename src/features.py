"""
Script para generar features por municipio-año para AlertaMaterna.

ENFOQUE: Usa archivos ORIGINALES (códigos numéricos) para cálculos precisos.
Los archivos decodificados son solo para visualización/reportes.

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

# Rutas de archivos - USAR ORIGINALES (códigos numéricos)
DATA_DIR = '../data/processed/'
NACIMIENTOS_FILE = f'{DATA_DIR}nacimientos_2020_2024.csv'
DEFUNCIONES_FETALES_FILE = f'{DATA_DIR}defunciones_fetales_2020_2024.csv'
DEFUNCIONES_NO_FETALES_FILE = f'{DATA_DIR}defunciones_no_fetales_2020_2024.csv'
REPS_FILE = f'{DATA_DIR}Registro_Especial_de_Prestadores_y_Sedes_de_Servicios_de_Salud_20251120.csv'
RIPS_FILE = f'{DATA_DIR}Registros_Individuales_de_Prestación_de_Servicios_de_Salud_–_RIPS_20251204.csv'
OUTPUT_FILE = f'{DATA_DIR}features_municipio_anio.csv'

# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def cargar_nacimientos():
    """Carga y filtra datos de nacimientos de la Orinoquía"""
    print("Cargando nacimientos (códigos numéricos)...")
    df = pd.read_csv(NACIMIENTOS_FILE, low_memory=False)
    
    # Filtrar Orinoquía
    df['COD_DPTO'] = df['COD_DPTO'].astype(str).str.strip()
    df = df[df['COD_DPTO'].isin(DPTOS_ORINOQUIA)].copy()
    
    # Convertir a numéricos las columnas críticas
    numeric_cols = ['ANO', 'COD_MUNIC', 'EDAD_MADRE', 'NUMCONSUL', 'PESO_NAC', 
                   'APGAR1', 'APGAR2', 'T_GES', 'MUL_PARTO', 'TIPO_PARTO', 
                   'N_HIJOSV', 'N_EMB', 'SEG_SOCIAL', 'EST_CIVM', 'NIV_EDUM']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"  → {len(df):,} nacimientos cargados")
    return df

def cargar_defunciones_fetales():
    """Carga y filtra defunciones fetales de la Orinoquía"""
    print("Cargando defunciones fetales (códigos numéricos)...")
    df = pd.read_csv(DEFUNCIONES_FETALES_FILE, low_memory=False)
    
    # Filtrar Orinoquía
    df['COD_DPTO'] = df['COD_DPTO'].astype(str).str.strip()
    df = df[df['COD_DPTO'].isin(DPTOS_ORINOQUIA)].copy()
    
    # Convertir a numéricos
    df['ANO'] = pd.to_numeric(df['ANO'], errors='coerce')
    df['COD_MUNIC'] = pd.to_numeric(df['COD_MUNIC'], errors='coerce')
    df['CAUSA_667'] = pd.to_numeric(df['CAUSA_667'], errors='coerce')
    
    print(f"  → {len(df):,} defunciones fetales cargadas")
    return df

def cargar_defunciones_no_fetales():
    """Carga y filtra defunciones no fetales (menores de 1 año) de la Orinoquía"""
    print("Cargando defunciones no fetales (códigos numéricos)...")
    df = pd.read_csv(DEFUNCIONES_NO_FETALES_FILE, low_memory=False)
    
    # Filtrar Orinoquía
    df['COD_DPTO'] = df['COD_DPTO'].astype(str).str.strip()
    df = df[df['COD_DPTO'].isin(DPTOS_ORINOQUIA)].copy()
    
    # Convertir a numéricos
    df['ANO'] = pd.to_numeric(df['ANO'], errors='coerce')
    df['COD_MUNIC'] = pd.to_numeric(df['COD_MUNIC'], errors='coerce')
    df['GRU_ED1'] = pd.to_numeric(df['GRU_ED1'], errors='coerce')
    df['CAUSA_667'] = pd.to_numeric(df['CAUSA_667'], errors='coerce')
    
    # Filtrar menores de 1 año (GRU_ED1: 1=<1h, 2=1-23h, 3=1-6d, 4=7-27d, 5=28d-11m, 6=1-4a)
    df = df[df['GRU_ED1'].isin([1, 2, 3, 4, 5])].copy()
    
    print(f"  → {len(df):,} defunciones < 1 año cargadas")
    return df

def cargar_instituciones():
    """Carga datos de instituciones de salud por municipio"""
    print("Cargando instituciones de salud...")
    df = pd.read_csv(REPS_FILE, sep=';', encoding='latin1', low_memory=False)
    
    # Filtrar Orinoquía por nombre de departamento
    df['DepartamentoSedeDesc'] = df['DepartamentoSedeDesc'].fillna('')
    orinoquia_names = ['Meta', 'Arauca', 'Casanare', 'Guaviare', 'Vichada']
    df = df[df['DepartamentoSedeDesc'].isin(orinoquia_names)].copy()
    
    print(f"  → {len(df):,} instituciones cargadas")
    return df

def cargar_rips():
    """Carga datos de servicios de salud (RIPS) por municipio-año"""
    print("Cargando servicios de salud (RIPS)...")
    df = pd.read_csv(RIPS_FILE, sep=';', encoding='latin1', low_memory=False)
    
    # Renombrar para consistencia
    df = df.rename(columns={'COD_DEP': 'COD_DPTO', 'COD_MUN': 'COD_MUNIC'})
    
    # Convertir a tipos apropiados
    df['COD_DPTO'] = df['COD_DPTO'].astype(str).str.strip().str.zfill(2)
    df['COD_MUNIC'] = pd.to_numeric(df['COD_MUNIC'], errors='coerce')
    df['ANO'] = pd.to_numeric(df['ANO'], errors='coerce')
    df['NumeroAtenciones'] = pd.to_numeric(df['NumeroAtenciones'], errors='coerce')
    
    # Filtrar Orinoquía y años 2020-2024
    df = df[df['COD_DPTO'].isin(DPTOS_ORINOQUIA)].copy()
    df = df[(df['ANO'] >= 2020) & (df['ANO'] <= 2024)].copy()
    
    print(f"  → {len(df):,} registros RIPS cargados")
    return df

# ============================================================================
# FUNCIONES DE FEATURES - DEMOGRÁFICAS (5)
# ============================================================================

def generar_features_demograficas(df_nac):
    """Genera features demográficas por municipio-año"""
    print("\nGenerando features demográficas...")
    
    # EDAD_MADRE ya viene como código numérico: 1=10-14, 2=15-19, 3=20-24, etc.
    # Convertir a edad promedio (punto medio del rango)
    edad_map = {
        1: 12, 2: 17, 3: 22, 4: 27, 5: 32, 
        6: 37, 7: 42, 8: 47, 9: 52, 99: np.nan
    }
    df_nac['edad_real'] = df_nac['EDAD_MADRE'].map(edad_map)
    
    # EST_CIVM (estado civil): 1=No casada, 2=Casada, 3=Viuda, 4=Separada, 5=Unión libre, 9=Sin info
    df_nac['madre_soltera'] = (df_nac['EST_CIVM'].isin([1, 4])).astype(int)
    
    # NIV_EDUM (educación madre): 1=Ninguno, 2=Preescolar, 3=Básica primaria, 4=Básica secundaria, 
    #                              5=Media académica, 6=Técnico, 7=Tecnológico, 8=Profesional, 9=Posgrado
    df_nac['educacion_baja'] = (df_nac['NIV_EDUM'].isin([1, 2, 3])).astype(int)
    
    # Agrupar por municipio-año
    features = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        total_nacimientos=('ANO', 'size'),
        edad_materna_promedio=('edad_real', 'mean'),
        pct_madres_adolescentes=('EDAD_MADRE', lambda x: (x.isin([1, 2])).sum() / len(x) * 100),
        pct_madres_solteras=('madre_soltera', lambda x: x.sum() / len(x) * 100),
        pct_educacion_baja=('educacion_baja', lambda x: x.sum() / len(x) * 100)
    ).reset_index()
    
    print(f"  → 5 features demográficas generadas para {len(features)} municipio-años")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - CLÍNICAS (7)
# ============================================================================

def generar_features_clinicas(df_nac):
    """Genera features clínicas por municipio-año"""
    print("\nGenerando features clínicas...")
    
    # T_GES (edad gestacional): 1=<22sem, 2=22-27, 3=28-31, 4=32-36, 5=37-41, 6=42+, 99=Sin info
    df_nac['prematuro'] = (df_nac['T_GES'].isin([1, 2, 3, 4])).astype(int)
    
    # PESO_NAC: 1=<500g, 2=500-999, 3=1000-1499, 4=1500-1999, 5=2000-2499, 6=2500-2999, 
    #           7=3000-3499, 8=3500-3999, 9=4000+, 99=Sin info
    df_nac['bajo_peso'] = (df_nac['PESO_NAC'].isin([1, 2, 3, 4, 5])).astype(int)
    
    # APGAR1 y APGAR2: 0-3=Severamente deprimido, 4-6=Moderadamente deprimido, 7-10=Normal
    df_nac['apgar1_bajo'] = (df_nac['APGAR1'] <= 6).astype(int)
    df_nac['apgar2_bajo'] = (df_nac['APGAR2'] <= 6).astype(int)
    
    # MUL_PARTO: 1=Simple, 2=Doble, 3=Triple, 4=Cuádruple o más
    df_nac['parto_multiple'] = (df_nac['MUL_PARTO'] > 1).astype(int)
    
    # TIPO_PARTO: 1=Espontáneo, 2=Ayudado, 3=Cesárea
    df_nac['cesarea'] = (df_nac['TIPO_PARTO'] == 3).astype(int)
    
    # Agrupar por municipio-año
    features = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        pct_prematuros=('prematuro', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0),
        pct_bajo_peso=('bajo_peso', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0),
        pct_apgar_bajo=('apgar1_bajo', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0),
        apgar_bajo_promedio=('apgar2_bajo', 'mean'),
        pct_partos_multiples=('parto_multiple', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0),
        pct_cesareas=('cesarea', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0),
        t_ges_promedio=('T_GES', lambda x: x[x != 99].mean() if (x != 99).sum() > 0 else np.nan)
    ).reset_index()
    
    print(f"  → 7 features clínicas generadas")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - INSTITUCIONALES (3)
# ============================================================================

def generar_features_institucionales(df_nac, df_inst):
    """Genera features institucionales por municipio"""
    print("\nGenerando features institucionales...")
    
    # Construir código municipal completo en REPS
    df_inst['COD_MUNIC_COMPLETO'] = (df_inst['COD_DEP'].astype(int) * 1000 + 
                                      df_inst['COD_MUN'].astype(int))
    
    # Contar instituciones por municipio
    inst_por_mun = df_inst.groupby('COD_MUNIC_COMPLETO').agg(
        num_instituciones=('NombreSede', 'nunique'),
        pct_instituciones_publicas=('NaturalezaJuridica', 
                                     lambda x: (x.str.contains('blica', case=False, na=False)).sum() / len(x) * 100 if len(x) > 0 else 0)
    ).reset_index()
    
    # Crear código completo en nacimientos
    df_nac['COD_MUNIC_COMPLETO'] = (df_nac['COD_DPTO'].astype(int) * 1000 + 
                                     df_nac['COD_MUNIC'].astype(int))
    
    # Agrupar nacimientos
    nac_por_mun = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO', 'COD_MUNIC_COMPLETO']).size().reset_index(name='total_nacimientos')
    
    # Merge
    features = nac_por_mun.merge(inst_por_mun, on='COD_MUNIC_COMPLETO', how='left')
    features['num_instituciones'] = features['num_instituciones'].fillna(0)
    features['pct_instituciones_publicas'] = features['pct_instituciones_publicas'].fillna(0)
    
    # Calcular instituciones per capita (por 1000 nacimientos)
    features['instituciones_per_1000nac'] = (features['num_instituciones'] / features['total_nacimientos'] * 1000).fillna(0)
    
    # Seleccionar columnas
    features = features[['COD_DPTO', 'COD_MUNIC', 'ANO', 'num_instituciones', 
                        'pct_instituciones_publicas', 'instituciones_per_1000nac']]
    
    print(f"  → 3 features institucionales generadas")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - ACCESO A SERVICIOS (4)
# ============================================================================

def generar_features_acceso_servicios(df_nac, df_rips):
    """Genera features de acceso a servicios de salud usando RIPS"""
    print("\nGenerando features de acceso a servicios...")
    
    # Agrupar RIPS por municipio-año
    rips_mun = df_rips.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        total_atenciones=('NumeroAtenciones', 'sum'),
        atenciones_urgencias=('TipoAtencion', lambda x: (x.str.contains('Urgencias', case=False, na=False)).sum()),
        atenciones_consulta=('TipoAtencion', lambda x: (x.str.contains('Consulta', case=False, na=False)).sum()),
        atenciones_procedimiento=('TipoAtencion', lambda x: (x.str.contains('Procedimiento', case=False, na=False)).sum())
    ).reset_index()
    
    # Contar nacimientos por municipio-año
    nac_count = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='total_nacimientos')
    
    # Merge
    features = nac_count.merge(rips_mun, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.fillna(0)
    
    # Calcular ratios
    features['atenciones_per_nacimiento'] = (features['total_atenciones'] / features['total_nacimientos']).fillna(0)
    features['urgencias_per_nacimiento'] = (features['atenciones_urgencias'] / features['total_nacimientos']).fillna(0)
    features['consultas_per_nacimiento'] = (features['atenciones_consulta'] / features['total_nacimientos']).fillna(0)
    features['procedimientos_per_nacimiento'] = (features['atenciones_procedimiento'] / features['total_nacimientos']).fillna(0)
    features['pct_urgencias'] = (features['atenciones_urgencias'] / features['total_atenciones'] * 100).fillna(0)
    
    # Seleccionar columnas
    features = features[['COD_DPTO', 'COD_MUNIC', 'ANO', 'atenciones_per_nacimiento', 
                        'urgencias_per_nacimiento', 'consultas_per_nacimiento', 
                        'procedimientos_per_nacimiento', 'pct_urgencias']]
    
    print(f"  → 5 features de acceso a servicios generadas")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - SOCIOECONÓMICAS (3)
# ============================================================================

def generar_features_socioeconomicas(df_nac):
    """Genera features socioeconómicas por municipio-año"""
    print("\nGenerando features socioeconómicas...")
    
    # SEG_SOCIAL (seguridad social): 1=Contributivo, 2=Subsidiado, 3=No asegurado, 4=Especial, 5=Excepción
    df_nac['sin_seguridad'] = (df_nac['SEG_SOCIAL'].isin([3])).astype(int)
    df_nac['subsidiado'] = (df_nac['SEG_SOCIAL'] == 2).astype(int)
    
    # N_HIJOSV (número de hijos vivos): Multiparidad ≥ 4 hijos
    df_nac['multiparidad'] = (df_nac['N_HIJOSV'] >= 4).astype(int)
    
    # Agrupar por municipio-año
    features = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        pct_sin_seguridad=('sin_seguridad', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0),
        pct_regimen_subsidiado=('subsidiado', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0),
        pct_multiparidad=('multiparidad', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0)
    ).reset_index()
    
    print(f"  → 3 features socioeconómicas generadas")
    return features

# ============================================================================
# FUNCIONES DE FEATURES - ATENCIÓN PRENATAL (3)
# ============================================================================

def generar_features_atencion_prenatal(df_nac):
    """Genera features de atención prenatal por municipio-año"""
    print("\nGenerando features de atención prenatal...")
    
    # NUMCONSUL (número de consultas prenatales): OMS recomienda mínimo 4
    df_nac['consultas_insuficientes'] = (df_nac['NUMCONSUL'] < 4).astype(int)
    
    # Agrupar por municipio-año
    features = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        consultas_promedio=('NUMCONSUL', lambda x: x[x != 99].mean() if (x != 99).sum() > 0 else 0),
        pct_consultas_insuficientes=('consultas_insuficientes', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0),
        pct_sin_control_prenatal=('NUMCONSUL', lambda x: ((x == 0) | (x == 99)).sum() / len(x) * 100 if len(x) > 0 else 0)
    ).reset_index()
    
    print(f"  → 3 features de atención prenatal generadas")
    return features

# ============================================================================
# FUNCIONES DE FEATURES CRÍTICAS AVANZADAS (4)
# ============================================================================

def generar_features_mortalidad_neonatal(df_nac, df_def_nofet):
    """Genera feature de tasa de mortalidad neonatal (0-27 días)"""
    print("\nGenerando features de mortalidad neonatal...")
    
    # Contar nacimientos por municipio-año
    nac_count = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='total_nacimientos')
    
    # Contar defunciones neonatales (GRU_ED1: 1=<1h, 2=1-23h, 3=1-6d, 4=7-27d)
    def_neonatal = df_def_nofet[df_def_nofet['GRU_ED1'].isin([1, 2, 3, 4])].copy()
    def_neonatal_count = def_neonatal.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='defunciones_neonatales')
    
    # Merge
    features = nac_count.merge(def_neonatal_count, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features['defunciones_neonatales'] = features['defunciones_neonatales'].fillna(0)
    
    # Calcular tasa por 1000 nacidos vivos
    features['tasa_mortalidad_neonatal'] = (features['defunciones_neonatales'] / features['total_nacimientos'] * 1000).fillna(0)
    
    # Seleccionar columnas
    features = features[['COD_DPTO', 'COD_MUNIC', 'ANO', 'tasa_mortalidad_neonatal']]
    
    print(f"  → Tasa promedio: {features['tasa_mortalidad_neonatal'].mean():.2f} por 1000 nacidos vivos")
    return features

def generar_features_mortalidad_fetal(df_nac, df_def_fet):
    """Genera feature de tasa de mortalidad fetal"""
    print("\nGenerando features de mortalidad fetal...")
    
    # Contar nacimientos por municipio-año
    nac_count = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='total_nacimientos')
    
    # Contar defunciones fetales
    def_fetal_count = df_def_fet.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='defunciones_fetales')
    
    # Merge
    features = nac_count.merge(def_fetal_count, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features['defunciones_fetales'] = features['defunciones_fetales'].fillna(0)
    
    # Calcular tasa por 1000 nacidos vivos
    features['tasa_mortalidad_fetal'] = (features['defunciones_fetales'] / features['total_nacimientos'] * 1000).fillna(0)
    
    # Seleccionar columnas
    features = features[['COD_DPTO', 'COD_MUNIC', 'ANO', 'tasa_mortalidad_fetal', 'defunciones_fetales']]
    
    print(f"  → Tasa promedio: {features['tasa_mortalidad_fetal'].mean():.2f} por 1000 nacidos vivos")
    return features

def generar_presion_obstetrica(df_nac, df_def_fet, df_def_nofet):
    """Genera feature de presión obstétrica (total defunciones / nacimientos)"""
    print("\nGenerando presión obstétrica...")
    
    # Contar nacimientos
    nac_count = df_nac.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='total_nacimientos')
    
    # Contar TODAS las defunciones (fetales + no fetales < 1 año)
    def_fet_count = df_def_fet.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='def_fetales')
    def_nofet_count = df_def_nofet.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).size().reset_index(name='def_nofetales')
    
    # Merge
    features = nac_count.merge(def_fet_count, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.merge(def_nofet_count, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.fillna(0)
    
    # Total defunciones
    features['total_defunciones'] = features['def_fetales'] + features['def_nofetales']
    
    # Presión obstétrica = total defunciones / nacimientos * 1000
    features['presion_obstetrica'] = (features['total_defunciones'] / features['total_nacimientos'] * 1000).fillna(0)
    
    # Seleccionar columnas
    features = features[['COD_DPTO', 'COD_MUNIC', 'ANO', 'presion_obstetrica', 'total_defunciones']]
    
    print(f"  → Presión promedio: {features['presion_obstetrica'].mean():.2f} por 1000 nacimientos")
    return features

def generar_features_causas_evitables(df_def_fet, df_def_nofet, df_nac):
    """Genera feature de % mortalidad por causas evitables"""
    print("\nGenerando features de causas evitables...")
    
    # Códigos CAUSA_667 evitables (401-410: obstétricas directas, 501-506: perinatales)
    causas_evitables = list(range(401, 411)) + list(range(501, 507))
    
    # Combinar defunciones
    def_fet_temp = df_def_fet[['COD_DPTO', 'COD_MUNIC', 'ANO', 'CAUSA_667']].copy()
    def_nofet_temp = df_def_nofet[['COD_DPTO', 'COD_MUNIC', 'ANO', 'CAUSA_667']].copy()
    todas_def = pd.concat([def_fet_temp, def_nofet_temp], ignore_index=True)
    
    # Identificar causas evitables
    todas_def['es_evitable'] = todas_def['CAUSA_667'].isin(causas_evitables).astype(int)
    
    # Agrupar
    def_count = todas_def.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        total_defunciones=('ANO', 'size'),
        defunciones_evitables=('es_evitable', 'sum')
    ).reset_index()
    
    # Calcular porcentaje
    def_count['pct_mortalidad_evitable'] = (def_count['defunciones_evitables'] / def_count['total_defunciones'] * 100).fillna(0).clip(0, 100)
    
    # Crear esqueleto con todos los municipios-años
    esqueleto = df_nac[['COD_DPTO', 'COD_MUNIC', 'ANO']].drop_duplicates()
    
    # Merge (municipios sin defunciones = 0% evitable)
    features = esqueleto.merge(def_count[['COD_DPTO', 'COD_MUNIC', 'ANO', 'pct_mortalidad_evitable']], 
                               on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features['pct_mortalidad_evitable'] = features['pct_mortalidad_evitable'].fillna(0)
    
    print(f"  → Promedio: {features['pct_mortalidad_evitable'].mean():.1f}% de muertes evitables")
    return features

def generar_features_embarazo_alto_riesgo(df_nac):
    """Genera feature de % embarazos de alto riesgo"""
    print("\nGenerando features de embarazo alto riesgo...")
    
    # Alto riesgo = prematuro O bajo peso O múltiple
    df_temp = df_nac.copy()
    df_temp['prematuro'] = df_temp['T_GES'].isin([1, 2, 3, 4])
    df_temp['bajo_peso'] = df_temp['PESO_NAC'].isin([1, 2, 3, 4, 5])
    df_temp['multiple'] = df_temp['MUL_PARTO'] > 1
    
    df_temp['alto_riesgo'] = (df_temp['prematuro'] | df_temp['bajo_peso'] | df_temp['multiple']).astype(int)
    
    # Agrupar
    features = df_temp.groupby(['COD_DPTO', 'COD_MUNIC', 'ANO']).agg(
        pct_embarazos_alto_riesgo=('alto_riesgo', lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0)
    ).reset_index()
    
    print(f"  → Promedio: {features['pct_embarazos_alto_riesgo'].mean():.1f}% embarazos alto riesgo")
    return features

def generar_indice_fragilidad(df_features):
    """Genera índice de fragilidad del sistema de salud (0-100)"""
    print("\nGenerando índice de fragilidad del sistema...")
    
    # Normalizar y combinar indicadores críticos
    df_temp = df_features.copy()
    
    # Componentes (mayor valor = mayor fragilidad)
    componentes = []
    
    # 1. Acceso (instituciones bajas = fragilidad alta)
    if 'instituciones_per_1000nac' in df_temp.columns:
        df_temp['frag_instituciones'] = 100 - (df_temp['instituciones_per_1000nac'] / df_temp['instituciones_per_1000nac'].max() * 100).fillna(0).clip(0, 100)
        componentes.append('frag_instituciones')
    
    # 2. Consultas prenatales bajas
    if 'pct_consultas_insuficientes' in df_temp.columns:
        componentes.append('pct_consultas_insuficientes')
    
    # 3. Sin seguridad social
    if 'pct_sin_seguridad' in df_temp.columns:
        componentes.append('pct_sin_seguridad')
    
    # 4. Mortalidad evitable
    if 'pct_mortalidad_evitable' in df_temp.columns:
        componentes.append('pct_mortalidad_evitable')
    
    # Calcular promedio de componentes
    if len(componentes) > 0:
        df_temp['indice_fragilidad_sistema'] = df_temp[componentes].mean(axis=1).fillna(0).clip(0, 100)
    else:
        df_temp['indice_fragilidad_sistema'] = 0
    
    print(f"  → Promedio: {df_temp['indice_fragilidad_sistema'].mean():.1f}")
    print(f"  → Municipios críticos (>80): {(df_temp['indice_fragilidad_sistema'] > 80).sum()}")
    
    return df_temp[['COD_DPTO', 'COD_MUNIC', 'ANO', 'indice_fragilidad_sistema']]

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que orquesta la generación de features"""
    
    print("=" * 80)
    print("GENERACIÓN DE FEATURES - ALERTAMATERNA")
    print("=" * 80)
    
    # 1. CARGAR DATOS
    df_nac = cargar_nacimientos()
    df_def_fet = cargar_defunciones_fetales()
    df_def_nofet = cargar_defunciones_no_fetales()
    df_inst = cargar_instituciones()
    df_rips = cargar_rips()
    
    # 2. GENERAR FEATURES BÁSICAS
    print("\n" + "=" * 80)
    print("GENERANDO FEATURES BÁSICAS")
    print("=" * 80)
    
    feat_demograficas = generar_features_demograficas(df_nac)
    feat_clinicas = generar_features_clinicas(df_nac)
    feat_institucionales = generar_features_institucionales(df_nac, df_inst)
    feat_acceso = generar_features_acceso_servicios(df_nac, df_rips)
    feat_socioeconomicas = generar_features_socioeconomicas(df_nac)
    feat_prenatal = generar_features_atencion_prenatal(df_nac)
    
    # 3. GENERAR FEATURES CRÍTICAS AVANZADAS
    print("\n" + "=" * 80)
    print("GENERANDO FEATURES CRÍTICAS AVANZADAS")
    print("=" * 80)
    
    feat_mortalidad = generar_features_mortalidad_neonatal(df_nac, df_def_nofet)
    feat_mortalidad_fetal = generar_features_mortalidad_fetal(df_nac, df_def_fet)
    feat_presion = generar_presion_obstetrica(df_nac, df_def_fet, df_def_nofet)
    feat_evitables = generar_features_causas_evitables(df_def_fet, df_def_nofet, df_nac)
    feat_alto_riesgo = generar_features_embarazo_alto_riesgo(df_nac)
    
    # 4. COMBINAR TODAS LAS FEATURES
    print("\n" + "=" * 80)
    print("COMBINANDO FEATURES")
    print("=" * 80)
    
    # Merge secuencial
    features = feat_demograficas
    features = features.merge(feat_clinicas, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.merge(feat_institucionales, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.merge(feat_acceso, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.merge(feat_socioeconomicas, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.merge(feat_prenatal, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.merge(feat_mortalidad, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.merge(feat_mortalidad_fetal, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.merge(feat_presion, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.merge(feat_evitables, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    features = features.merge(feat_alto_riesgo, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    
    # 5. GENERAR ÍNDICE DE FRAGILIDAD (usa todas las features)
    feat_fragilidad = generar_indice_fragilidad(features)
    features = features.merge(feat_fragilidad, on=['COD_DPTO', 'COD_MUNIC', 'ANO'], how='left')
    
    # 6. APLICAR FILTRO OMS (≥ 10 nacimientos/año)
    print("\nAplicando filtro OMS (≥ 10 nacimientos/año)...")
    features_filtrado = features[features['total_nacimientos'] >= 10].copy()
    print(f"  → Registros antes del filtro: {len(features)}")
    print(f"  → Registros después del filtro: {len(features_filtrado)}")
    print(f"  → Registros excluidos: {len(features) - len(features_filtrado)}")
    
    # 7. GUARDAR ARCHIVO
    features_filtrado.to_csv(OUTPUT_FILE, index=False)
    
    # 8. RESUMEN FINAL
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print(f"Total de registros: {len(features_filtrado)}")
    print(f"Total de features: {len(features_filtrado.columns) - 3}")  # Excluyendo COD_DPTO, COD_MUNIC, ANO
    print(f"Años: {sorted(features_filtrado['ANO'].unique())}")
    print(f"Departamentos: {sorted(features_filtrado['COD_DPTO'].unique())}")
    print(f"Municipios únicos: {features_filtrado['COD_MUNIC'].nunique()}")
    print(f"\nArchivo guardado en: {OUTPUT_FILE}")
    
    # Estadísticas clave
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS CLAVE")
    print("=" * 80)
    print(f"\n📊 Mortalidad Neonatal:")
    print(f"   Media: {features_filtrado['tasa_mortalidad_neonatal'].mean():.2f} por 1000 nacidos vivos")
    print(f"   Rango: {features_filtrado['tasa_mortalidad_neonatal'].min():.2f} - {features_filtrado['tasa_mortalidad_neonatal'].max():.2f}")
    print(f"   Municipios con tasa >15: {(features_filtrado['tasa_mortalidad_neonatal'] > 15).sum()}")
    
    print(f"\n📊 Mortalidad Evitable:")
    print(f"   Media: {features_filtrado['pct_mortalidad_evitable'].mean():.1f}%")
    print(f"   Municipios con >50% evitable: {(features_filtrado['pct_mortalidad_evitable'] > 50).sum()}")
    
    print(f"\n📊 Embarazos Alto Riesgo:")
    print(f"   Media: {features_filtrado['pct_embarazos_alto_riesgo'].mean():.1f}%")
    print(f"   Municipios con >30% alto riesgo: {(features_filtrado['pct_embarazos_alto_riesgo'] > 30).sum()}")
    
    print(f"\n📊 Índice de Fragilidad:")
    print(f"   Media: {features_filtrado['indice_fragilidad_sistema'].mean():.1f}")
    print(f"   Municipios críticos (>80): {(features_filtrado['indice_fragilidad_sistema'] > 80).sum()}")
    
    print("\n✅ Proceso completado exitosamente!")
    print("\nPrimeras filas:")
    print(features_filtrado.head())

if __name__ == "__main__":
    main()

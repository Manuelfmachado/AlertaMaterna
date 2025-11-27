"""
Configuración del proyecto AlertaMaterna
"""

from pathlib import Path

# Rutas del proyecto
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / 'BACKUP_ARCHIVOS_PROCESO' / 'datos_normalizados'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
PREDICTIONS_DIR = BASE_DIR / 'data' / 'predictions'
MODELS_DIR = BASE_DIR / 'models'

# Departamentos objetivo
DEPARTAMENTOS_OBJETIVO = {
    '50': 'Meta',
    '81': 'Arauca',
    '85': 'Casanare',
    '99': 'Vichada',
    '95': 'Guaviare'
}

# Archivos de datos
ARCHIVOS_DATOS = {
    'defunciones': DATA_DIR / 'defunciones_fetales_2015_2018_normalizado.csv',
    'nacimientos': DATA_DIR / 'nacimientos_2015_2018_normalizado.csv',
    'indicadores': DATA_DIR / 'indicadores_mortalidad_morbilidad.csv',
    'prestadores': DATA_DIR / 'registro_prestadores_salud.csv',
    'rips': DATA_DIR / 'rips_servicios_salud.csv'
}

# Parámetros del modelo
MODELO_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'class_weight': 'balanced',
        'random_state': 42
    }
}

# Features del modelo
FEATURES = [
    # Demográficas
    'total_nacimientos',
    'total_defunciones',
    'pct_madres_adolescentes',
    'pct_madres_edad_avanzada',
    'pct_bajo_nivel_educativo',
    'edad_materna_promedio',
    
    # Clínicas
    'tasa_mortalidad_fetal',
    'pct_bajo_peso',
    'pct_embarazo_multiple',
    'pct_cesarea',
    'pct_prematuro',
    'apgar_bajo_promedio',
    
    # Institucionales
    'presion_obstetrica',
    'num_instituciones',
    'pct_instituciones_publicas',
    'camas_per_capita',
    
    # Socioeconómicas
    'pct_sin_seguridad_social',
    'pct_area_rural',
    'pct_regimen_subsidiado',
    
    # Atención prenatal
    'pct_sin_control_prenatal',
    'consultas_promedio'
]

# Umbrales para clasificación de riesgo
UMBRALES_RIESGO = {
    'tasa_mortalidad_fetal_percentil': 75,
    'presion_obstetrica_max': 100,
    'pct_sin_seguridad_social_max': 30,
    'pct_madres_adolescentes_max': 20,
    'camas_per_capita_percentil': 25
}

# Configuración del dashboard
DASHBOARD_CONFIG = {
    'titulo': '🤰 AlertaMaterna - Anticipación del riesgo obstétrico en la región Orinoquía',
    'mapa_centro': [4.0, -72.0],
    'mapa_zoom': 6,
    'colores': {
        'alto_riesgo': '#d32f2f',
        'bajo_riesgo': '#388e3c',
        'medio_riesgo': '#f57c00'
    }
}

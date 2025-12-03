"""
Script de diagnóstico de calidad de datos para AlertaMaterna.

Verifica que las features generadas estén en rangos razonables y no contengan
valores extremos que puedan causar predicciones incoherentes.

Proyecto: AlertaMaterna - Sistema de Clasificación de Riesgo Obstétrico 
          y Predicción de Mortalidad Infantil en la Región Orinoquía
"""

import pandas as pd
import numpy as np
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Obtener ruta absoluta del directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
FEATURES_FILE = os.path.join(DATA_DIR, 'features_municipio_anio.csv')

# Rangos razonables para cada tipo de feature
RANGOS_ESPERADOS = {
    # Features de RIPS (por nacimiento)
    'atenciones_per_nacimiento': (0, 500),
    'consultas_per_nacimiento': (0, 100),
    'urgencias_per_nacimiento': (0, 50),
    'procedimientos_per_nacimiento': (0, 100),
    
    # Porcentajes (0-100%)
    'pct_area_rural': (0, 100),
    'pct_bajo_nivel_educativo': (0, 100),
    'pct_bajo_peso': (0, 50),
    'pct_cesarea': (0, 100),
    'pct_embarazo_multiple': (0, 10),
    'pct_embarazos_alto_riesgo': (0, 100),
    'pct_instituciones_publicas': (0, 100),
    'pct_madres_adolescentes': (0, 50),
    'pct_madres_edad_avanzada': (0, 50),
    'pct_mortalidad_evitable': (0, 100),
    'pct_prematuro': (0, 50),
    'pct_regimen_subsidiado': (0, 100),
    'pct_sin_control_prenatal': (0, 100),
    'pct_sin_seguridad_social': (0, 100),
    
    # Tasas de mortalidad (por 1000)
    'tasa_mortalidad_fetal': (0, 200),
    'tasa_mortalidad_neonatal': (0, 100),
    
    # Otros
    'edad_materna_promedio': (15, 45),
    'apgar_bajo_promedio': (0, 50),
    'presion_obstetrica': (0, 1000),
    'num_instituciones': (0, 100),
    'indice_fragilidad_sistema': (0, 100),
    'total_nacimientos': (0, 20000),
}

# ============================================================================
# FUNCIONES DE DIAGNÓSTICO
# ============================================================================

def diagnosticar_features():
    """Diagnostica calidad de datos en el archivo de features"""
    print("="*80)
    print("DIAGNÓSTICO DE CALIDAD DE DATOS - AlertaMaterna")
    print("="*80)
    
    # Cargar datos
    print(f"\nCargando {FEATURES_FILE}...")
    df = pd.read_csv(FEATURES_FILE)
    print(f"✓ {len(df)} registros cargados")
    
    # 1. DETECTAR VALORES INFINITOS
    print("\n" + "="*80)
    print("1. DETECCIÓN DE VALORES INFINITOS")
    print("="*80)
    
    inf_count = 0
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            inf_mask = np.isinf(df[col])
            if inf_mask.any():
                inf_count += inf_mask.sum()
                print(f"❌ {col}: {inf_mask.sum()} valores infinitos")
    
    if inf_count == 0:
        print("✅ No se encontraron valores infinitos")
    else:
        print(f"\n⚠️  TOTAL: {inf_count} valores infinitos detectados")
    
    # 2. DETECTAR VALORES NaN
    print("\n" + "="*80)
    print("2. DETECCIÓN DE VALORES NaN")
    print("="*80)
    
    nan_count = 0
    for col in df.columns:
        if df[col].isna().any():
            nan_count += df[col].isna().sum()
            print(f"⚠️  {col}: {df[col].isna().sum()} valores NaN")
    
    if nan_count == 0:
        print("✅ No se encontraron valores NaN")
    else:
        print(f"\n⚠️  TOTAL: {nan_count} valores NaN detectados")
    
    # 3. DETECTAR VALORES FUERA DE RANGO
    print("\n" + "="*80)
    print("3. DETECCIÓN DE VALORES FUERA DE RANGO")
    print("="*80)
    
    fuera_rango_count = 0
    features_problematicas = []
    
    for col, (min_val, max_val) in RANGOS_ESPERADOS.items():
        if col in df.columns:
            fuera_min = (df[col] < min_val).sum()
            fuera_max = (df[col] > max_val).sum()
            
            if fuera_min > 0 or fuera_max > 0:
                fuera_rango_count += fuera_min + fuera_max
                features_problematicas.append(col)
                print(f"⚠️  {col}:")
                print(f"     Rango esperado: [{min_val}, {max_val}]")
                print(f"     Rango real: [{df[col].min():.2f}, {df[col].max():.2f}]")
                if fuera_min > 0:
                    print(f"     {fuera_min} valores < {min_val}")
                if fuera_max > 0:
                    print(f"     {fuera_max} valores > {max_val}")
                    # Mostrar ejemplos de valores extremos
                    extremos = df[df[col] > max_val][col].head(3)
                    print(f"     Ejemplos: {extremos.tolist()}")
    
    if fuera_rango_count == 0:
        print("✅ Todas las features están en rangos razonables")
    else:
        print(f"\n⚠️  TOTAL: {fuera_rango_count} valores fuera de rango en {len(features_problematicas)} features")
    
    # 4. ESTADÍSTICAS DESCRIPTIVAS DE FEATURES CRÍTICAS
    print("\n" + "="*80)
    print("4. ESTADÍSTICAS DE FEATURES CRÍTICAS")
    print("="*80)
    
    features_criticas = [
        'tasa_mortalidad_fetal',
        'tasa_mortalidad_neonatal',
        'pct_embarazos_alto_riesgo',
        'pct_mortalidad_evitable',
        'indice_fragilidad_sistema',
        'atenciones_per_nacimiento'
    ]
    
    for feat in features_criticas:
        if feat in df.columns:
            print(f"\n{feat}:")
            print(f"  Media: {df[feat].mean():.2f}")
            print(f"  Mediana: {df[feat].median():.2f}")
            print(f"  Desv. Est: {df[feat].std():.2f}")
            print(f"  Min: {df[feat].min():.2f}")
            print(f"  Max: {df[feat].max():.2f}")
            print(f"  P95: {df[feat].quantile(0.95):.2f}")
    
    # 5. RESUMEN FINAL
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    
    problemas_totales = inf_count + nan_count + fuera_rango_count
    
    if problemas_totales == 0:
        print("✅ ¡EXCELENTE! No se detectaron problemas de calidad de datos")
        print("   Los datos están listos para entrenar el modelo")
    else:
        print(f"⚠️  Se detectaron {problemas_totales} problemas de calidad:")
        print(f"   - {inf_count} valores infinitos")
        print(f"   - {nan_count} valores NaN")
        print(f"   - {fuera_rango_count} valores fuera de rango")
        print("\n🔧 RECOMENDACIÓN:")
        print("   1. Regenerar features con: python src/features.py")
        print("   2. Verificar correcciones aplicadas en generar_features_rips()")
        print("   3. Reentrenar modelo con: python src/train_model.py")
    
    return problemas_totales == 0

if __name__ == "__main__":
    todo_ok = diagnosticar_features()
    exit(0 if todo_ok else 1)

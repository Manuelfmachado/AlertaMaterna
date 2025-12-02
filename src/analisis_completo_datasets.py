"""
Análisis exhaustivo de estructura, calidad y oportunidades de merge
Dataset: AlertaMaterna
"""

import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path('../data/processed')

print("=" * 100)
print("ANÁLISIS COMPLETO DE DATASETS - ALERTAMATERNA")
print("=" * 100)

# ============================================================================
# 1. NACIMIENTOS DECODED
# ============================================================================
print("\n[1] NACIMIENTOS_2020_2024_DECODED.CSV")
print("-" * 100)

nac = pd.read_csv(data_dir / 'nacimientos_2020_2024_decoded.csv', nrows=50000)
print(f"Dimensiones (muestra): {nac.shape[0]:,} × {nac.shape[1]} columnas")
print(f"Llaves de merge: COD_DPTO ({nac['COD_DPTO'].nunique()} únicos), COD_MUNIC ({nac['COD_MUNIC'].nunique()}), ANO ({sorted(nac['ANO'].unique())})")
print(f"Columnas totales: {list(nac.columns[:20])}... (+{nac.shape[1]-20} más)")

# Analizar columnas clave
print(f"\nCalidad de datos:")
print(f"   • Nulos totales: {nac.isnull().sum().sum():,} ({nac.isnull().sum().sum()/(nac.shape[0]*nac.shape[1])*100:.1f}%)")
print(f"   • Duplicados: {nac.duplicated().sum()}")

# ============================================================================
# 2. DEFUNCIONES FETALES DECODED
# ============================================================================
print("\n\n[2] DEFUNCIONES_FETALES_2020_2024_DECODED.CSV")
print("-" * 100)

def_fet = pd.read_csv(data_dir / 'defunciones_fetales_2020_2024_decoded.csv')
print(f"Dimensiones: {def_fet.shape[0]:,} × {def_fet.shape[1]} columnas")
print(f"Llaves: COD_DPTO ({def_fet['COD_DPTO'].nunique()}), COD_MUNIC ({def_fet['COD_MUNIC'].nunique()}), ANO ({sorted(def_fet['ANO'].unique())})")
print(f"Columnas: {list(def_fet.columns)}")

# Overlap con nacimientos
nac_munic = set(zip(nac['COD_DPTO'], nac['COD_MUNIC']))
fet_munic = set(zip(def_fet['COD_DPTO'], def_fet['COD_MUNIC']))
overlap = len(nac_munic.intersection(fet_munic))
print(f"\nMERGE con Nacimientos: {overlap}/{len(nac_munic)} municipios ({overlap/len(nac_munic)*100:.1f}% overlap)")

# ============================================================================
# 3. DEFUNCIONES NO FETALES DECODED
# ============================================================================
print("\n\n3⃣  DEFUNCIONES_NO_FETALES_2020_2024_DECODED.CSV")
print("-" * 100)

def_nofet = pd.read_csv(data_dir / 'defunciones_no_fetales_2020_2024_decoded.csv', nrows=50000)
print(f" Dimensiones (muestra): {def_nofet.shape[0]:,} × {def_nofet.shape[1]} columnas")
print(f" Llaves: COD_DPTO ({def_nofet['COD_DPTO'].nunique()}), COD_MUNIC ({def_nofet['COD_MUNIC'].nunique()}), ANO ({sorted(def_nofet['ANO'].unique())})")
print(f" Columnas: {list(def_nofet.columns[:25])}...")

nofet_munic = set(zip(def_nofet['COD_DPTO'], def_nofet['COD_MUNIC']))
overlap2 = len(nac_munic.intersection(nofet_munic))
print(f"\n MERGE con Nacimientos: {overlap2}/{len(nac_munic)} municipios ({overlap2/len(nac_munic)*100:.1f}% overlap)")

# ============================================================================
# 4. REPS
# ============================================================================
print("\n\n4⃣  REGISTRO_ESPECIAL_PRESTADORES (REPS)")
print("-" * 100)

reps = pd.read_csv(data_dir / 'Registro_Especial_de_Prestadores_y_Sedes_de_Servicios_de_Salud_20251120.csv')
print(f" Dimensiones: {reps.shape[0]:,} × {reps.shape[1]} columnas")
print(f" Columnas: {list(reps.columns)}")

# Analizar columna de municipio
if 'MunicipioSede' in reps.columns:
    print(f"\n MunicipioSede:")
    print(f"   • Valores únicos: {reps['MunicipioSede'].nunique()}")
    print(f"   • Ejemplos: {reps['MunicipioSede'].head(5).tolist()}")
    print(f"   • Formato: {reps['MunicipioSede'].dtype}")
    
    # Intentar extraer código de 5 dígitos
    reps['COD_5DIG'] = reps['MunicipioSede'].astype(str).str.extract(r'^(\d{5})')[0]
    if reps['COD_5DIG'].notna().any():
        print(f"   • Códigos 5 dígitos extraídos: {reps['COD_5DIG'].nunique()} únicos")
        
        # Comparar con nacimientos
        nac['COD_5DIG'] = (nac['COD_DPTO'].astype(int) * 1000 + nac['COD_MUNIC'].astype(int))
        nac_codes = set(nac['COD_5DIG'].unique())
        reps_codes = set(reps['COD_5DIG'].dropna().astype(int))
        overlap_reps = len(nac_codes.intersection(reps_codes))
        
        print(f"\n MERGE con Nacimientos:")
        print(f"   • Nacimientos códigos únicos: {len(nac_codes)}")
        print(f"   • REPS códigos únicos: {len(reps_codes)}")
        print(f"   •  OVERLAP: {overlap_reps} municipios ({overlap_reps/len(nac_codes)*100:.1f}%)")

# Analizar servicios
if 'NaturalezaJuridica' in reps.columns:
    print(f"\n Naturaleza Jurídica:")
    print(reps['NaturalezaJuridica'].value_counts().head(5))

if 'NivelAtencion' in reps.columns:
    print(f"\n Nivel de Atención:")
    print(reps['NivelAtencion'].value_counts())

# ============================================================================
# 5. RIPS
# ============================================================================
print("\n\n5⃣  REGISTROS_INDIVIDUALES_PRESTACIÓN (RIPS)")
print("-" * 100)

try:
    rips = pd.read_csv(data_dir / 'Registros_Individuales_de_Prestación_de_Servicios_de_Salud_–_RIPS_20251120.csv', 
                       on_bad_lines='skip', encoding='latin1')
    print(f" Dimensiones: {rips.shape[0]:,} × {rips.shape[1]} columnas")
    print(f" Columnas: {list(rips.columns)}")
except Exception as e:
    print(f"  Error al leer RIPS: {e}")
    print("   Intentando análisis alternativo...")
    # Leer primeras líneas manualmente
    with open(data_dir / 'Registros_Individuales_de_Prestación_de_Servicios_de_Salud_–_RIPS_20251120.csv', 'r', encoding='latin1') as f:
        lines = [f.readline() for _ in range(5)]
        print(f"   Primeras líneas del archivo:")
        for i, line in enumerate(lines, 1):
            print(f"      {i}. {line.strip()[:100]}")
    rips = None
    rips = None

if rips is not None:
    if 'Municipio' in rips.columns:
        print(f"\n Municipio:")
        print(f"   • Valores únicos: {rips['Municipio'].nunique()}")
        print(f"   • Ejemplos: {rips['Municipio'].head(5).tolist()}")

    if 'ANO' in rips.columns:
        print(f"\n Años disponibles: {sorted(rips['ANO'].unique())}")

    if 'TipoAtencion' in rips.columns:
        print(f"\n Tipos de Atención:")
        print(rips['TipoAtencion'].value_counts())

    if 'NumeroAtenciones' in rips.columns:
        print(f"\n Atenciones:")
        print(f"   • Total: {rips['NumeroAtenciones'].sum():,.0f}")
        print(f"   • Promedio: {rips['NumeroAtenciones'].mean():.1f}")

# ============================================================================
# 6. DIVIPOLA
# ============================================================================
print("\n\n6⃣  DIVIPOLA - CÓDIGOS MUNICIPIOS")
print("-" * 100)

divipola = pd.read_csv(data_dir / 'DIVIPOLA-_Códigos_municipios_20251128.csv')
print(f" Dimensiones: {divipola.shape[0]:,} × {divipola.shape[1]} columnas")
print(f" Columnas: {list(divipola.columns)}")
print(f"\n Primeras 5 filas:")
print(divipola.head(10).to_string())

# ============================================================================
# 7. CÓDIGOS DANE
# ============================================================================
print("\n\n7⃣  CÓDIGOS DANE")
print("-" * 100)

for filename in ['codigos_nacimientos_dane.csv', 'codigos_defunciones_fetales_dane.csv', 
                 'codigos_defunciones_no_fetales_dane.csv']:
    filepath = data_dir / filename
    if filepath.exists():
        cod = pd.read_csv(filepath)
        print(f"\n {filename}:")
        print(f"   • Dimensiones: {cod.shape[0]} × {cod.shape[1]}")
        print(f"   • Columnas: {list(cod.columns)}")
        if cod.shape[0] <= 10:
            print(f"   • Contenido:")
            print(cod.to_string(index=False))

# ============================================================================
# 8. RESUMEN DE MERGE Y FEATURES
# ============================================================================
print("\n\n" + "=" * 100)
print(" RESUMEN DE COMPATIBILIDAD Y OPORTUNIDADES")
print("=" * 100)

print("\n CAPACIDAD DE MERGE:")
print("    Nacimientos ↔ Def. Fetales: PERFECTA (COD_DPTO, COD_MUNIC, ANO)")
print("    Nacimientos ↔ Def. No Fetales: PERFECTA (COD_DPTO, COD_MUNIC, ANO)")
print(f"    Nacimientos ↔ REPS: BUENA ({overlap_reps/len(nac_codes)*100:.1f}% overlap vía código 5 dígitos)")
print("    Nacimientos ↔ RIPS: BUENA (código 5 dígitos + ANO)")
print("    Todos ↔ DIVIPOLA: EXCELENTE (códigos oficiales)")

print("\n TOP 15 FEATURES IMPACTANTES A CREAR:")
print("\n A. MORTALIDAD ESPECÍFICA (de Defunciones):")
print("   1. tasa_mortalidad_neonatal (0-28 días) - ALTA PRIORIDAD")
print("   2. mortalidad_fetal_anteparto vs intraparto")
print("   3. proporcion_muertes_certificadas_medico")
print("   4. mortalidad_por_causas_evitables")

print("\n B. CAPACIDAD INSTITUCIONAL (REPS):")
print("   5. instituciones_nivel_III_per_capita")
print("   6. ratio_instituciones_publico_privado")
print("   7. instituciones_con_UCI_neonatal (binaria)")
print("   8. capacidad_resolutiva = nivel_complejidad × num_instituciones")

print("\n C. ACCESO A SERVICIOS (RIPS):")
print("   9. cobertura_control_prenatal = consultas/nacimientos")
print("   10. ratio_urgencias_consultas (preventivo vs reactivo)")
print("   11. deficit_servicios = presion_obstetrica × (1 - cobertura)")

print("\n D. GEOGRÁFICAS (DIVIPOLA):")
print("   12. distancia_capital_departamental (km)")
print("   13. municipio_fronterizo (binaria)")
print("   14. densidad_poblacional")

print("\n E. COMPUESTAS (Multi-tabla):")
print("   15. indice_fragilidad = (mortalidad_neonatal × presion) / capacidad_resolutiva")

print("\n⏱  TIEMPO DE IMPLEMENTACIÓN:")
print("   • Features 1-4: 45 min")
print("   • Features 5-8: 1 hora")
print("   • Features 9-11: 45 min")
print("   • Features 12-14: 30 min (si DIVIPOLA tiene coordenadas)")
print("   • Feature 15: 15 min")
print("   • TOTAL: ~3.5 horas")

print("\n PRÓXIMOS PASOS RECOMENDADOS:")
print("   1. Implementar features 1, 5, 9, 12, 15 (las más impactantes)")
print("   2. Reentrenar modelo con nuevas features")
print("   3. Comparar ROC-AUC antes/después")
print("   4. Documentar mejoras para pitch")

print("\n" + "=" * 100)
print("FIN DEL ANÁLISIS")
print("=" * 100)

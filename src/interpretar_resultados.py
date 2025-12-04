"""
Script para decodificar y interpretar resultados de AlertaMaterna
Convierte códigos numéricos a descripciones legibles para el dashboard

Proyecto: AlertaMaterna
"""

import pandas as pd
import numpy as np

# Rutas
DATA_DIR = '../data/processed/'
FEATURES_FILE = f'{DATA_DIR}features_municipio_anio.csv'
OUTPUT_FILE = f'{DATA_DIR}features_municipio_anio_interpretado.csv'

# ============================================================================
# MAPEOS DE DECODIFICACIÓN (basados en DANE)
# ============================================================================

# Mapeo de edades de madre (códigos DANE)
EDAD_MADRE_MAP = {
    1: "10-14 años",
    2: "15-19 años",
    3: "20-24 años",
    4: "25-29 años",
    5: "30-34 años",
    6: "35-39 años",
    7: "40-44 años",
    8: "45-49 años",
    9: "50-54 años",
    99: "Sin información"
}

# Rango de edad materna real (para interpretar promedios)
def interpretar_edad_materna(edad_promedio):
    """Interpreta el promedio de edad materna"""
    if edad_promedio < 18:
        return "Adolescentes (muy alto riesgo)"
    elif edad_promedio < 25:
        return "Jóvenes (bajo riesgo)"
    elif edad_promedio < 35:
        return "Adultas jóvenes (óptimo)"
    else:
        return "Edad avanzada (alto riesgo)"

# Interpretación de tasas
def interpretar_tasa_mortalidad(tasa, tipo="infantil"):
    """Interpreta tasas de mortalidad según estándares OMS"""
    if tipo == "infantil":
        if tasa < 5:
            return "🟢 Normal (OMS: <5‰)"
        elif tasa < 10:
            return "🟡 Moderado (5-10‰)"
        elif tasa < 20:
            return "🟠 Alto (10-20‰)"
        else:
            return "🔴 Crítico (>20‰)"
    elif tipo == "fetal":
        if tasa < 10:
            return "🟢 Bajo"
        elif tasa < 20:
            return "🟡 Moderado"
        elif tasa < 50:
            return "🟠 Alto"
        else:
            return "🔴 Crítico"

# Interpretación de porcentajes
def interpretar_porcentaje(valor, variable):
    """Interpreta porcentajes según variable"""
    if variable == "bajo_peso":
        if valor < 8:
            return "🟢 Bajo"
        elif valor < 12:
            return "🟡 Moderado"
        elif valor < 15:
            return "🟠 Alto"
        else:
            return "🔴 Muy alto"
    
    elif variable == "prematuros":
        if valor < 8:
            return "🟢 Bajo"
        elif valor < 12:
            return "🟡 Moderado"
        elif valor < 15:
            return "🟠 Alto"
        else:
            return "🔴 Muy alto"
    
    elif variable == "sin_control_prenatal":
        if valor < 5:
            return "🟢 Excelente cobertura"
        elif valor < 10:
            return "🟡 Buena cobertura"
        elif valor < 20:
            return "🟠 Cobertura deficiente"
        else:
            return "🔴 Cobertura crítica"
    
    elif variable == "cesareas":
        if valor < 15:
            return "🟡 Bajo (riesgo de subutilización)"
        elif valor < 45:
            return "🟢 Óptimo (OMS: 10-15%)"
        else:
            return "🔴 Alto (OMS: máx 15%)"
    
    elif variable == "adolescentes":
        if valor < 10:
            return "🟢 Bajo"
        elif valor < 20:
            return "🟡 Moderado"
        elif valor < 30:
            return "🟠 Alto"
        else:
            return "🔴 Muy alto"
    
    else:
        if valor < 33:
            return "🟢 Bajo"
        elif valor < 66:
            return "🟡 Moderado"
        else:
            return "🔴 Alto"

def decodificar_features(df):
    """Decodifica features para interpretación"""
    print("Decodificando features para interpretación...")
    
    df_output = df.copy()
    
    # Crear columna de interpretación de edad materna
    df_output['edad_materna_categoria'] = df_output['edad_materna_promedio'].apply(
        interpretar_edad_materna
    )
    
    # Crear interpretaciones de tasas de mortalidad
    df_output['tasa_mortalidad_fetal_categoria'] = df_output['tasa_mortalidad_fetal'].apply(
        lambda x: interpretar_tasa_mortalidad(x, "fetal")
    )
    
    df_output['tasa_mortalidad_neonatal_categoria'] = df_output['tasa_mortalidad_neonatal'].apply(
        lambda x: interpretar_tasa_mortalidad(x, "infantil")
    )
    
    # Crear interpretaciones de porcentajes
    df_output['bajo_peso_categoria'] = df_output['pct_bajo_peso'].apply(
        lambda x: interpretar_porcentaje(x, "bajo_peso")
    )
    
    df_output['prematuros_categoria'] = df_output['pct_prematuro'].apply(
        lambda x: interpretar_porcentaje(x, "prematuros")
    )
    
    df_output['sin_prenatal_categoria'] = df_output['pct_sin_control_prenatal'].apply(
        lambda x: interpretar_porcentaje(x, "sin_control_prenatal")
    )
    
    df_output['cesareas_categoria'] = df_output['pct_cesarea'].apply(
        lambda x: interpretar_porcentaje(x, "cesareas")
    )
    
    df_output['madres_adolescentes_categoria'] = df_output['pct_madres_adolescentes'].apply(
        lambda x: interpretar_porcentaje(x, "adolescentes")
    )
    
    # Clasificación de fragilidad
    def interpretar_fragilidad(indice):
        if indice < 25:
            return "🟢 Sistema fuerte"
        elif indice < 50:
            return "🟡 Sistema moderadamente frágil"
        elif indice < 75:
            return "🟠 Sistema frágil"
        else:
            return "🔴 Sistema muy frágil"
    
    df_output['fragilidad_categoria'] = df_output['indice_fragilidad_sistema'].apply(
        interpretar_fragilidad
    )
    
    # Presión obstétrica
    def interpretar_presion(presion):
        if presion < 10:
            return "🟢 Baja (buena capacidad)"
        elif presion < 30:
            return "🟡 Moderada"
        elif presion < 50:
            return "🟠 Alta"
        else:
            return "🔴 Muy alta (saturación)"
    
    df_output['presion_obstetrica_categoria'] = df_output['presion_obstetrica'].apply(
        interpretar_presion
    )
    
    return df_output

def main():
    """Función principal"""
    print("=" * 80)
    print("DECODIFICADOR DE FEATURES - ALERTAMATERNA")
    print("=" * 80)
    print()
    
    # Cargar features
    print("Cargando features...")
    df = pd.read_csv(FEATURES_FILE)
    print(f"  ✓ {len(df)} registros cargados")
    print(f"  ✓ {len(df.columns)} columnas")
    print()
    
    # Decodificar
    df_interpretado = decodificar_features(df)
    
    # Guardar
    df_interpretado.to_csv(OUTPUT_FILE, index=False)
    print()
    print("=" * 80)
    print(f"Archivo guardado: {OUTPUT_FILE}")
    print("=" * 80)
    print()
    
    # Mostrar ejemplos
    print("EJEMPLOS DE INTERPRETACIÓN:")
    print()
    
    sample_cols = [
        'COD_DPTO', 'COD_MUNIC', 'ANO',
        'edad_materna_promedio', 'edad_materna_categoria',
        'tasa_mortalidad_fetal', 'tasa_mortalidad_fetal_categoria',
        'pct_bajo_peso', 'bajo_peso_categoria',
        'pct_sin_control_prenatal', 'sin_prenatal_categoria',
        'indice_fragilidad_sistema', 'fragilidad_categoria'
    ]
    
    available_cols = [c for c in sample_cols if c in df_interpretado.columns]
    print(df_interpretado[available_cols].head(10).to_string())
    print()
    print("✅ Proceso completado exitosamente!")

if __name__ == "__main__":
    main()

"""
Análisis detallado de integración REPS/RIPS y propuestas de nuevas variables
"""

import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('../data/processed/features_alerta_materna.csv')

print("=" * 80)
print("ANÁLISIS DE INTEGRACIÓN REPS Y RIPS")
print("=" * 80)

print("\n1. FEATURES ACTUALES DE REPS (Instituciones):")
print("-" * 80)
print(f"   • num_instituciones:")
print(f"     - Rango: {df['num_instituciones'].min():.0f} - {df['num_instituciones'].max():.0f}")
print(f"     - Media: {df['num_instituciones'].mean():.1f}")
print(f"     - Mediana: {df['num_instituciones'].median():.1f}")
print(f"     - Municipios con 0 instituciones: {(df['num_instituciones']==0).sum()}")

print(f"\n   • pct_instituciones_publicas:")
print(f"     - Rango: {df['pct_instituciones_publicas'].min():.2%} - {df['pct_instituciones_publicas'].max():.2%}")
print(f"     - Media: {df['pct_instituciones_publicas'].mean():.2%}")

print(f"\n   • presion_obstetrica (nacimientos/institución):")
print(f"     - Rango: {df['presion_obstetrica'].min():.1f} - {df['presion_obstetrica'].max():.1f}")
print(f"     - Media: {df['presion_obstetrica'].mean():.1f}")
print(f"     - Percentil 75: {df['presion_obstetrica'].quantile(0.75):.1f}")

print("\n\n2. FEATURES ACTUALES DE RIPS (Servicios de Salud):")
print("-" * 80)
print(f"   • urgencias_per_nacimiento:")
print(f"     - Municipios con datos: {(df['urgencias_per_nacimiento']>0).sum()} / {len(df)} ({(df['urgencias_per_nacimiento']>0).sum()/len(df)*100:.1f}%)")
print(f"     - Media (cuando >0): {df[df['urgencias_per_nacimiento']>0]['urgencias_per_nacimiento'].mean():.3f}")

print(f"\n   • consultas_per_nacimiento:")
print(f"     - Municipios con datos: {(df['consultas_per_nacimiento']>0).sum()} / {len(df)} ({(df['consultas_per_nacimiento']>0).sum()/len(df)*100:.1f}%)")
print(f"     - Media (cuando >0): {df[df['consultas_per_nacimiento']>0]['consultas_per_nacimiento'].mean():.3f}")

print(f"\n   • procedimientos_per_nacimiento:")
print(f"     - Municipios con datos: {(df['procedimientos_per_nacimiento']>0).sum()} / {len(df)} ({(df['procedimientos_per_nacimiento']>0).sum()/len(df)*100:.1f}%)")
print(f"     - Media (cuando >0): {df[df['procedimientos_per_nacimiento']>0]['procedimientos_per_nacimiento'].mean():.3f}")

print("\n\n3. CORRELACIONES CON TARGET (alta_mortalidad):")
print("-" * 80)
corr = df.corr()['alta_mortalidad'].sort_values(ascending=False)
features_reps_rips = ['presion_obstetrica', 'num_instituciones', 'pct_instituciones_publicas',
                      'urgencias_per_nacimiento', 'consultas_per_nacimiento', 'procedimientos_per_nacimiento']

for feat in features_reps_rips:
    print(f"   • {feat:40s}: {corr[feat]:+.4f}")

print("\n\n4. COMPARACIÓN CON MEJORES PREDICTORES:")
print("-" * 80)
top_corr = corr.head(10)
for feat, val in top_corr.items():
    marker = " REPS/RIPS" if feat in features_reps_rips else ""
    print(f"   {feat:40s}: {val:+.4f}  {marker}")

print("\n\n5. ANÁLISIS POR DEPARTAMENTO (REPS):")
print("-" * 80)
dept_analysis = df.groupby('COD_DPTO').agg({
    'num_instituciones': 'mean',
    'presion_obstetrica': 'mean',
    'alta_mortalidad': 'mean'
}).round(2)
dept_analysis.index = dept_analysis.index.map({50: 'Meta', 81: 'Arauca', 85: 'Casanare', 
                                                95: 'Guaviare', 99: 'Vichada'})
print(dept_analysis)

print("\n\n6. PROPUESTAS DE NUEVAS VARIABLES COMBINADAS:")
print("=" * 80)

print("\n A. Variables de CAPACIDAD vs DEMANDA:")
print("-" * 80)
print("    presion_obstetrica (YA EXISTE): nacimientos/instituciones")
print("   → NUEVA: ratio_servicios_instituciones = (consultas+urgencias+procedimientos)/num_instituciones")
print("     Mide: Servicios totales por institución disponible")
print("   → NUEVA: cobertura_servicios = (consultas+urgencias)/total_nacimientos")
print("     Mide: Cobertura real de servicios obstétricos")

print("\n B. Variables de CALIDAD DE ATENCIÓN:")
print("-" * 80)
print("   → NUEVA: ratio_urgencias_consultas = urgencias/consultas")
print("     Mide: Proporción de atención de emergencia vs preventiva")
print("   → NUEVA: completitud_atencion = (consultas + urgencias + procedimientos) / presion_obstetrica")
print("     Mide: Calidad de atención considerando carga institucional")

print("\n C. Variables de ACCESO y EQUIDAD:")
print("-" * 80)
print("   → NUEVA: deficit_institucional = presion_obstetrica > percentil_75")
print("     Mide: Municipios con sobrecarga institucional")
print("   → NUEVA: brecha_atencion = (pct_sin_control_prenatal * presion_obstetrica)")
print("     Mide: Riesgo por falta de acceso combinado con sobrecarga")
print("   → NUEVA: vulnerabilidad_institucional = (pct_instituciones_publicas * pct_regimen_subsidiado)")
print("     Mide: Dependencia del sistema público en población vulnerable")

print("\n D. Variables de RIESGO COMPUESTO:")
print("-" * 80)
print("   → NUEVA: indice_fragilidad_sistema = (presion_obstetrica/percentil_75) * (1 - cobertura_servicios)")
print("     Mide: Fragilidad del sistema de salud local")
print("   → NUEVA: multiplicador_riesgo = tasa_mortalidad_fetal * (1 + presion_obstetrica/100)")
print("     Mide: Mortalidad amplificada por presión institucional")

print("\n E. Variables TEMPORALES:")
print("-" * 80)
print("   → NUEVA: tendencia_presion = presion_obstetrica_año_actual - presion_obstetrica_año_anterior")
print("     Mide: Empeoramiento/mejora de la capacidad")
print("   → NUEVA: volatilidad_servicios = std(urgencias) últimos 3 años")
print("     Mide: Inestabilidad del sistema de salud")

print("\n\n7. LIMITACIONES ACTUALES:")
print("=" * 80)
print(f"     RIPS solo tiene datos en {(df['urgencias_per_nacimiento']>0).sum()} de {len(df)} registros ({(df['urgencias_per_nacimiento']>0).sum()/len(df)*100:.1f}%)")
print(f"     {(df['num_instituciones']==0).sum()} municipios con 0 instituciones (imputados con promedio)")
print(f"     Presión obstétrica tiene outliers (max: {df['presion_obstetrica'].max():.0f} nacimientos/institución)")

print("\n\n8. RECOMENDACIONES:")
print("=" * 80)
print("   1⃣  Crear las 10 nuevas variables propuestas (categorías A-E)")
print("   2⃣  Normalizar presion_obstetrica con log1p para manejar outliers")
print("   3⃣  Crear variable binaria para municipios SIN datos RIPS (puede ser predictor)")
print("   4⃣  Probar interacciones: presion_obstetrica * pct_sin_control_prenatal")
print("   5⃣  Agregar feature de DISTANCIA a capital más cercana (proxy acceso)")

print("\n" + "=" * 80)
print("FIN DEL ANÁLISIS")
print("=" * 80)

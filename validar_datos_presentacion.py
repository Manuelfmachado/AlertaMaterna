"""
Script para validar los datos que aparecen en la presentación
"""
import pandas as pd

# Cargar datos
feat = pd.read_csv('data/processed/features_municipio_anio_interpretado.csv')

print('=' * 60)
print('VALIDACION DATOS PRESENTACION - AlertaMaterna')
print('=' * 60)
print()
print('ALCANCE GEOGRAFICO: ORINOQUIA (5 departamentos)')
print('PERIODO: 2020-2024')
print()

# 1. Nacimientos analizados en Orinoquía
nac_total_ori = feat['total_nacimientos'].sum()
print(f'1. Nacimientos analizados Orinoquía: {int(nac_total_ori):,}')
print()

# 2. Municipios únicos
mun_total = feat.groupby(['COD_DPTO','COD_MUNIC']).ngroups
print(f'2. Municipios únicos Orinoquía: {mun_total}')
print()

# 3. Registros municipio-año (251)
print(f'3. Registros municipio-año totales: {len(feat)}')
print()

# 4. Alto riesgo según mortalidad neonatal
# Según OMS: >10‰ es alto, >20‰ es crítico
alto_riesgo_registros = feat[feat['tasa_mortalidad_neonatal'] > 10]
mun_alto_riesgo = alto_riesgo_registros.groupby(['COD_DPTO','COD_MUNIC']).ngroups
print(f'4. ALTO RIESGO (MN > 10‰):')
print(f'   - Registros municipio-año: {len(alto_riesgo_registros)}')
print(f'   - Municipios únicos: {mun_alto_riesgo}')
print()

# 5. Críticos según mortalidad neonatal
criticos_registros = feat[feat['tasa_mortalidad_neonatal'] > 20]
mun_criticos = criticos_registros.groupby(['COD_DPTO','COD_MUNIC']).ngroups if len(criticos_registros) > 0 else 0
print(f'5. CRÍTICOS (MN > 20‰):')
print(f'   - Registros municipio-año: {len(criticos_registros)}')
print(f'   - Municipios únicos: {mun_criticos}')
print()

# 6. Muy críticos (>50‰)
muy_criticos = feat[feat['tasa_mortalidad_neonatal'] > 50]
mun_muy_criticos = muy_criticos.groupby(['COD_DPTO','COD_MUNIC']).ngroups if len(muy_criticos) > 0 else 0
print(f'6. MUY CRÍTICOS (MN > 50‰):')
print(f'   - Registros municipio-año: {len(muy_criticos)}')
print(f'   - Municipios únicos: {mun_muy_criticos}')
print()

# 7. Estadísticas por año
print('7. POR AÑO:')
for ano in sorted(feat['ANO'].unique()):
    df_ano = feat[feat['ANO']==ano]
    nac_ano = int(df_ano['total_nacimientos'].sum())
    alto_ano = len(df_ano[df_ano['tasa_mortalidad_neonatal'] > 10])
    print(f'   {ano}: {len(df_ano)} municipios, {nac_ano:,} nacimientos, {alto_ano} alto riesgo')
print()

# 8. Nacimientos 2024 para "embarazos monitoreados"
df2024 = feat[feat['ANO']==2024]
nac_2024 = int(df2024['total_nacimientos'].sum())
print(f'8. Nacimientos 2024 (Orinoquía): {nac_2024:,}')
print()

# 9. Muertes evitables
pct_evitable = feat['pct_mortalidad_evitable'].mean()
print(f'9. % Mortalidad evitable (promedio): {pct_evitable:.1f}%')
print()

# 10. Municipio más crítico
print('10. MUNICIPIO MÁS CRÍTICO:')
max_mn = feat.loc[feat['tasa_mortalidad_neonatal'].idxmax()]
print(f'    Código: {int(max_mn["COD_DPTO"])}-{int(max_mn["COD_MUNIC"])}')
print(f'    Año: {int(max_mn["ANO"])}')
print(f'    Tasa MN: {max_mn["tasa_mortalidad_neonatal"]:.1f}‰')
print(f'    Nacimientos: {int(max_mn["total_nacimientos"])}')
print()

# 11. Verificar departamentos
print('11. DEPARTAMENTOS ORINOQUÍA:')
# Mapear códigos de departamento
cod_dpto_map = {50: 'Meta', 81: 'Arauca', 85: 'Casanare', 95: 'Guaviare', 99: 'Vichada'}
for cod in sorted(feat['COD_DPTO'].unique()):
    df_dpto = feat[feat['COD_DPTO']==cod]
    mun_dpto = df_dpto.groupby('COD_MUNIC').ngroups
    nac_dpto = int(df_dpto['total_nacimientos'].sum())
    nombre = cod_dpto_map.get(cod, f'Dpto {cod}')
    print(f'    {nombre} ({cod}): {mun_dpto} municipios, {nac_dpto:,} nacimientos')
print()

# 12. Tasa mortalidad por año (evolución)
print('12. EVOLUCIÓN TASA MORTALIDAD NEONATAL (promedio ponderado):')
for ano in sorted(feat['ANO'].unique()):
    df_ano = feat[feat['ANO']==ano]
    # Promedio ponderado por nacimientos
    if df_ano['total_nacimientos'].sum() > 0:
        tasa_pond = (df_ano['tasa_mortalidad_neonatal'] * df_ano['total_nacimientos']).sum() / df_ano['total_nacimientos'].sum()
    else:
        tasa_pond = 0
    print(f'    {ano}: {tasa_pond:.1f}‰')

print()

# 13. Evolución mortalidad FETAL
print('13. EVOLUCIÓN TASA MORTALIDAD FETAL (promedio ponderado):')
for ano in sorted(feat['ANO'].unique()):
    df_ano = feat[feat['ANO']==ano]
    if df_ano['total_nacimientos'].sum() > 0:
        tasa_pond = (df_ano['tasa_mortalidad_fetal'] * df_ano['total_nacimientos']).sum() / df_ano['total_nacimientos'].sum()
    else:
        tasa_pond = 0
    print(f'    {ano}: {tasa_pond:.1f}‰')
print()

# 14. Críticos mortalidad fetal >50‰
print('14. CRÍTICOS MORTALIDAD FETAL (>50‰):')
criticos_fetal = feat[feat['tasa_mortalidad_fetal'] > 50]
mun_criticos_fetal = criticos_fetal.groupby(['COD_DPTO','COD_MUNIC']).ngroups
print(f'    - Registros municipio-año: {len(criticos_fetal)}')
print(f'    - Municipios únicos: {mun_criticos_fetal}')
print()

# 15. Alto riesgo mortalidad fetal >30‰
print('15. ALTO RIESGO MORTALIDAD FETAL (>30‰):')
alto_fetal = feat[feat['tasa_mortalidad_fetal'] > 30]
mun_alto_fetal = alto_fetal.groupby(['COD_DPTO','COD_MUNIC']).ngroups
print(f'    - Registros municipio-año: {len(alto_fetal)}')
print(f'    - Municipios únicos: {mun_alto_fetal}')
print()

# 16. Saravena datos
print('16. SARAVENA (Arauca) - Datos reales:')
saravena = feat[(feat['COD_DPTO']==81) & (feat['COD_MUNIC']==736)]
for _, row in saravena.iterrows():
    print(f'    {int(row["ANO"])}: MF={row["tasa_mortalidad_fetal"]:.1f}‰, MN={row["tasa_mortalidad_neonatal"]:.1f}‰, Nac={int(row["total_nacimientos"])}')
print()

print('=' * 60)
print('FIN VALIDACIÓN')
print('=' * 60)

"""
Script para analizar la correspondencia entre columnas de datos y códigos DANE.
Genera un reporte de qué columnas tienen códigos disponibles y cómo se deben reemplazar.
"""

import pandas as pd
import os

# Rutas de archivos
data_dir = "data/processed"
codigos_nacimientos = os.path.join(data_dir, "codigos_nacimientos_dane.csv")
codigos_defunciones_fetales = os.path.join(data_dir, "codigos_defunciones_fetales_dane.csv")
codigos_defunciones_no_fetales = os.path.join(data_dir, "codigos_defunciones_no_fetales_dane.csv")

nacimientos_file = os.path.join(data_dir, "nacimientos_2020_2024.csv")
defunciones_fetales_file = os.path.join(data_dir, "defunciones_fetales_2020_2024.csv")
defunciones_no_fetales_file = os.path.join(data_dir, "defunciones_no_fetales_2020_2024.csv")

def cargar_codigos(archivo):
    """Carga archivo de códigos y devuelve diccionario por variable."""
    df = pd.read_csv(archivo)
    codigos_dict = {}
    
    for variable in df['variable'].unique():
        df_var = df[df['variable'] == variable]
        # Crear diccionario codigo -> descripcion
        codigos_dict[variable] = dict(zip(
            df_var['codigo'].astype(str), 
            df_var['descripcion']
        ))
    
    return codigos_dict

def obtener_columnas_dataset(archivo, nrows=1000):
    """Obtiene las columnas de un dataset grande."""
    df = pd.read_csv(archivo, nrows=nrows)
    return df.columns.tolist(), df

def analizar_correspondencia():
    """Analiza la correspondencia entre columnas y códigos."""
    
    print("="*80)
    print("ANÁLISIS DE CORRESPONDENCIA ENTRE COLUMNAS Y CÓDIGOS DANE")
    print("="*80)
    
    # Cargar códigos
    print("\n📁 Cargando archivos de códigos...")
    cod_nac = cargar_codigos(codigos_nacimientos)
    cod_def_fet = cargar_codigos(codigos_defunciones_fetales)
    cod_def_no_fet = cargar_codigos(codigos_defunciones_no_fetales)
    
    print(f"   ✓ Códigos Nacimientos: {len(cod_nac)} variables")
    print(f"   ✓ Códigos Defunciones Fetales: {len(cod_def_fet)} variables")
    print(f"   ✓ Códigos Defunciones No Fetales: {len(cod_def_no_fet)} variables")
    
    # Analizar cada dataset
    datasets = [
        ("NACIMIENTOS", nacimientos_file, cod_nac),
        ("DEFUNCIONES FETALES", defunciones_fetales_file, cod_def_fet),
        ("DEFUNCIONES NO FETALES", defunciones_no_fetales_file, cod_def_no_fet)
    ]
    
    resultados = {}
    
    for nombre, archivo, codigos in datasets:
        print(f"\n{'='*80}")
        print(f"📊 ANALIZANDO: {nombre}")
        print(f"{'='*80}")
        
        columnas, df_muestra = obtener_columnas_dataset(archivo)
        print(f"\n   Total de columnas: {len(columnas)}")
        
        con_codigo = []
        sin_codigo = []
        
        for col in columnas:
            if col in codigos:
                con_codigo.append(col)
            else:
                sin_codigo.append(col)
        
        print(f"\n   ✅ Columnas CON código disponible: {len(con_codigo)}")
        print(f"   ❌ Columnas SIN código disponible: {len(sin_codigo)}")
        
        # Mostrar columnas con código
        if con_codigo:
            print(f"\n   📋 COLUMNAS QUE SE PUEDEN DECODIFICAR ({len(con_codigo)}):")
            for col in sorted(con_codigo):
                num_codigos = len(codigos[col])
                # Mostrar valores únicos en la muestra
                valores_muestra = df_muestra[col].dropna().unique()[:5]
                print(f"      • {col:20s} ({num_codigos} códigos) - Valores ejemplo: {valores_muestra}")
        
        # Mostrar columnas sin código
        if sin_codigo:
            print(f"\n   📋 COLUMNAS QUE NO SE DECODIFICAN ({len(sin_codigo)}):")
            for col in sorted(sin_codigo):
                print(f"      • {col}")
        
        resultados[nombre] = {
            'total': len(columnas),
            'con_codigo': con_codigo,
            'sin_codigo': sin_codigo,
            'codigos': codigos
        }
    
    # Resumen general
    print(f"\n{'='*80}")
    print("📈 RESUMEN GENERAL")
    print(f"{'='*80}")
    
    for nombre, datos in resultados.items():
        porcentaje = (len(datos['con_codigo']) / datos['total']) * 100
        print(f"\n{nombre}:")
        print(f"   Total columnas: {datos['total']}")
        print(f"   Con código: {len(datos['con_codigo'])} ({porcentaje:.1f}%)")
        print(f"   Sin código: {len(datos['sin_codigo'])} ({100-porcentaje:.1f}%)")
    
    # Ejemplo de cómo hacer el reemplazo
    print(f"\n{'='*80}")
    print("💡 EJEMPLO DE CÓMO REEMPLAZAR LOS CÓDIGOS")
    print(f"{'='*80}")
    
    print("""
MÉTODO 1: Usar pandas.map() para cada columna
-----------------------------------------------
# Cargar datos
df = pd.read_csv('nacimientos_2020_2024.csv')

# Cargar códigos
codigos = pd.read_csv('codigos_nacimientos_dane.csv')

# Crear diccionarios de mapeo por variable
mapeos = {}
for variable in codigos['variable'].unique():
    df_var = codigos[codigos['variable'] == variable]
    mapeos[variable] = dict(zip(
        df_var['codigo'].astype(str), 
        df_var['descripcion']
    ))

# Reemplazar cada columna que tenga código
for columna in df.columns:
    if columna in mapeos:
        # Crear nueva columna con descripción
        df[f'{columna}_DESC'] = df[columna].astype(str).map(mapeos[columna])

# Ahora tienes columnas duplicadas:
# - COD_DPTO (código numérico original)
# - COD_DPTO_DESC (descripción texto)

MÉTODO 2: Merge con el archivo de códigos
------------------------------------------
# Para una variable específica (por ejemplo, COD_DPTO)
codigos_dpto = codigos[codigos['variable'] == 'COD_DPTO'][['codigo', 'descripcion']]
codigos_dpto = codigos_dpto.rename(columns={'codigo': 'COD_DPTO', 'descripcion': 'DEPARTAMENTO'})

# Hacer merge
df = df.merge(codigos_dpto, on='COD_DPTO', how='left')

MÉTODO 3: Crear un diccionario global
--------------------------------------
# Si quieres mantener los códigos originales pero poder consultarlos
codigo_to_desc = {}
for variable in codigos['variable'].unique():
    df_var = codigos[codigos['variable'] == variable]
    for _, row in df_var.iterrows():
        codigo_to_desc[(variable, str(row['codigo']))] = row['descripcion']

# Uso:
descripcion = codigo_to_desc[('COD_DPTO', '11')]  # Returns: 'Bogotá'
""")
    
    # Mostrar ejemplo práctico
    print(f"\n{'='*80}")
    print("🔍 EJEMPLO PRÁCTICO CON DATOS REALES")
    print(f"{'='*80}")
    
    # Tomar muestra de nacimientos
    df_nac_sample = pd.read_csv(nacimientos_file, nrows=3)
    cod_nac_df = pd.read_csv(codigos_nacimientos)
    
    print("\n📄 DATOS ORIGINALES (primeras 3 filas de NACIMIENTOS):")
    print(df_nac_sample[['COD_DPTO', 'SEXO', 'PESO_NAC', 'EDAD_MADRE']].to_string())
    
    # Crear mapeos
    mapeos = {}
    for variable in ['COD_DPTO', 'SEXO', 'PESO_NAC', 'EDAD_MADRE']:
        df_var = cod_nac_df[cod_nac_df['variable'] == variable]
        mapeos[variable] = dict(zip(
            df_var['codigo'].astype(str), 
            df_var['descripcion']
        ))
    
    # Aplicar mapeos
    df_decoded = df_nac_sample.copy()
    for col in ['COD_DPTO', 'SEXO', 'PESO_NAC', 'EDAD_MADRE']:
        df_decoded[f'{col}_DESC'] = df_decoded[col].astype(str).map(mapeos[col])
    
    print("\n📄 DATOS DECODIFICADOS (con descripciones):")
    print(df_decoded[['COD_DPTO', 'COD_DPTO_DESC', 'SEXO', 'SEXO_DESC', 'PESO_NAC', 'PESO_NAC_DESC', 'EDAD_MADRE', 'EDAD_MADRE_DESC']].to_string())
    
    return resultados

if __name__ == "__main__":
    resultados = analizar_correspondencia()
    
    print(f"\n{'='*80}")
    print("✅ ANÁLISIS COMPLETADO")
    print(f"{'='*80}\n")

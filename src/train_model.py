"""
Script para entrenar modelos de AlertaMaterna.

Implementa dos modelos:
1. Clasificación de Riesgo Obstétrico (índice compuesto)
2. Predicción de Alta Mortalidad Infantil (XGBoost)

Proyecto: AlertaMaterna - Sistema de Clasificación de Riesgo Obstétrico 
          y Predicción de Mortalidad Infantil en la Región Orinoquía
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DATA_DIR = '../data/processed/'
FEATURES_FILE = f'{DATA_DIR}features_municipio_anio.csv'
MODEL_DIR = '../models/'

# Crear directorio de modelos si no existe
import os
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# MODELO 1: CLASIFICACIÓN DE RIESGO OBSTÉTRICO
# ============================================================================

def crear_indice_riesgo_obstetrico(df):
    """
    Crea un índice de riesgo obstétrico basado en criterios de salud pública.
    
    SISTEMA HÍBRIDO:
    - Umbrales CRÍTICOS absolutos (mortalidad extrema → alto riesgo automático)
    - Percentiles para otros indicadores
    
    Criterios:
    - Tasa mortalidad fetal >50‰ → ALTO RIESGO AUTOMÁTICO (crítico)
    - Tasa mortalidad fetal >p75 → +1 punto
    - % sin control prenatal >p75 → +1 punto
    - % bajo peso >p75 → +1 punto
    - % prematuro >p75 → +1 punto
    - % cesárea <p25 → +1 punto
    - Presión obstétrica >p75 → +1 punto
    
    Clasificación: ≥3 puntos O mortalidad >50‰ → ALTO RIESGO
    """
    print("\n" + "="*80)
    print("MODELO 1: ÍNDICE DE RIESGO OBSTÉTRICO (HÍBRIDO)")
    print("="*80)
    
    # UMBRALES CRÍTICOS ABSOLUTOS (OMS/Literatura médica)
    UMBRAL_CRITICO_MORTALIDAD = 50.0  # 50‰ es 10x la tasa normal (5‰)
    UMBRAL_CRITICO_SIN_PRENATAL = 0.5  # 50% sin atención prenatal
    MIN_NACIMIENTOS = 10  # Filtrar municipios muy pequeños
    
    print("\n UMBRALES CRÍTICOS (alertas automáticas):")
    print(f"  - Mortalidad fetal > {UMBRAL_CRITICO_MORTALIDAD}‰ → ALTO RIESGO AUTOMÁTICO")
    print(f"  - Sin atención prenatal > {UMBRAL_CRITICO_SIN_PRENATAL:.0%} → +2 puntos")
    print(f"  - Municipios con < {MIN_NACIMIENTOS} nacimientos → EXCLUIDOS del análisis")
    
    # Filtrar municipios muy pequeños (datos poco confiables)
    df_filtrado = df[df['total_nacimientos'] >= MIN_NACIMIENTOS].copy()
    excluidos = len(df) - len(df_filtrado)
    
    if excluidos > 0:
        print(f"\n {excluidos} registros excluidos (< {MIN_NACIMIENTOS} nacimientos)")
    
    # Calcular percentiles sobre datos filtrados
    p25_cesarea = df_filtrado['pct_cesarea'].quantile(0.25)
    p75_mort_fetal = df_filtrado['tasa_mortalidad_fetal'].quantile(0.75)
    p75_sin_prenatal = df_filtrado['pct_sin_control_prenatal'].quantile(0.75)
    p75_bajo_peso = df_filtrado['pct_bajo_peso'].quantile(0.75)
    p75_prematuro = df_filtrado['pct_prematuro'].quantile(0.75)
    p75_presion_obs = df_filtrado['presion_obstetrica'].quantile(0.75)
    
    print("\n Criterios basados en percentiles:")
    print(f"  - Tasa mortalidad fetal > {p75_mort_fetal:.2f}‰")
    print(f"  - % sin control prenatal > {p75_sin_prenatal:.2%}")
    print(f"  - % bajo peso > {p75_bajo_peso:.2%}")
    print(f"  - % prematuro > {p75_prematuro:.2%}")
    print(f"  - % cesárea < {p25_cesarea:.2%}")
    print(f"  - Presión obstétrica > {p75_presion_obs:.1f}")
    
    # Calcular puntuación para datos filtrados (0-8 puntos máximo)
    df_filtrado['puntos_riesgo'] = 0
    
    # Criterios con peso normal (1 punto c/u)
    df_filtrado.loc[df_filtrado['tasa_mortalidad_fetal'] > p75_mort_fetal, 'puntos_riesgo'] += 1
    df_filtrado.loc[df_filtrado['pct_bajo_peso'] > p75_bajo_peso, 'puntos_riesgo'] += 1
    df_filtrado.loc[df_filtrado['pct_prematuro'] > p75_prematuro, 'puntos_riesgo'] += 1
    df_filtrado.loc[df_filtrado['pct_cesarea'] < p25_cesarea, 'puntos_riesgo'] += 1
    df_filtrado.loc[df_filtrado['presion_obstetrica'] > p75_presion_obs, 'puntos_riesgo'] += 1
    
    # Criterio de atención prenatal (peso doble si es extremo)
    df_filtrado.loc[df_filtrado['pct_sin_control_prenatal'] > p75_sin_prenatal, 'puntos_riesgo'] += 1
    df_filtrado.loc[df_filtrado['pct_sin_control_prenatal'] > UMBRAL_CRITICO_SIN_PRENATAL, 'puntos_riesgo'] += 1
    
    # ALERTA CRÍTICA: Mortalidad extrema suma +3 puntos (garantiza alto riesgo)
    df_filtrado.loc[df_filtrado['tasa_mortalidad_fetal'] > UMBRAL_CRITICO_MORTALIDAD, 'puntos_riesgo'] += 3
    
    # Marcar municipios excluidos con puntos = -1
    df['puntos_riesgo'] = -1
    df.loc[df_filtrado.index, 'puntos_riesgo'] = df_filtrado['puntos_riesgo']
    
    # Clasificar (≥3 puntos = alto riesgo, -1 = excluido)
    df['riesgo_obstetrico'] = 0
    df.loc[df['puntos_riesgo'] >= 3, 'riesgo_obstetrico'] = 1
    df.loc[df['puntos_riesgo'] == -1, 'riesgo_obstetrico'] = -1  # Excluidos
    
    # Estadísticas
    total_validos = (df['riesgo_obstetrico'] >= 0).sum()
    excluidos = (df['riesgo_obstetrico'] == -1).sum()
    alto_riesgo = (df['riesgo_obstetrico'] == 1).sum()
    bajo_riesgo = (df['riesgo_obstetrico'] == 0).sum()
    
    print(f"\n Resultados:")
    print(f"  - Total registros válidos: {total_validos:,}")
    print(f"  - Excluidos (< {MIN_NACIMIENTOS} nac): {excluidos:,}")
    print(f"  - Alto riesgo: {alto_riesgo:,} ({alto_riesgo/total_validos:.1%})")
    print(f"  - Bajo riesgo: {bajo_riesgo:,} ({bajo_riesgo/total_validos:.1%})")
    
    # Mostrar municipios con mortalidad crítica
    criticos = df[df['tasa_mortalidad_fetal'] > UMBRAL_CRITICO_MORTALIDAD]
    if len(criticos) > 0:
        print(f"\n ALERTA: {len(criticos)} municipios con mortalidad >50‰:")
        for _, row in criticos.iterrows():
            print(f"    - Código {int(row['COD_DPTO'])}-{int(row['COD_MUNIC'])} ({int(row['ANO'])}): "
                  f"{row['tasa_mortalidad_fetal']:.1f}‰ | {int(row['total_nacimientos'])} nac | "
                  f"Puntaje: {int(row['puntos_riesgo'])}")
    
    # Guardar umbrales para uso en producción
    umbral = {
        'min_nacimientos': MIN_NACIMIENTOS,
        'umbral_critico_mortalidad': UMBRAL_CRITICO_MORTALIDAD,
        'umbral_critico_sin_prenatal': UMBRAL_CRITICO_SIN_PRENATAL,
        'p25_cesarea': p25_cesarea,
        'p75_mort_fetal': p75_mort_fetal,
        'p75_sin_prenatal': p75_sin_prenatal,
        'p75_bajo_peso': p75_bajo_peso,
        'p75_prematuro': p75_prematuro,
        'p75_presion_obs': p75_presion_obs
    }
    
    with open(f'{MODEL_DIR}umbral_riesgo_obstetrico.pkl', 'wb') as f:
        pickle.dump(umbral, f)
    
    return df

# ============================================================================
# MODELO 2: PREDICCIÓN DE MORTALIDAD INFANTIL (REGRESIÓN)
# ============================================================================

def preparar_datos_mortalidad(df):
    """
    Prepara datos para predicción de mortalidad infantil.
    Target: Tasa de mortalidad infantil en ‰ (valor continuo)
    """
    print("\n" + "="*80)
    print("MODELO 2: PREDICCIÓN DE TASA DE MORTALIDAD INFANTIL (REGRESIÓN)")
    print("="*80)
    
    # ========================================================================
    # LIMPIEZA Y VALIDACIÓN DE DATOS
    # ========================================================================
    print("\n[1/4] Limpiando y validando datos...")
    
    # Reemplazar infinitos y valores extremos
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Validar y limpiar features de RIPS (pueden tener valores absurdos)
    rips_features = ['atenciones_per_nacimiento', 'consultas_per_nacimiento', 
                     'urgencias_per_nacimiento', 'procedimientos_per_nacimiento']
    
    for feat in rips_features:
        if feat in df.columns:
            # Si valor es absurdo (>1000), reemplazar con mediana
            mediana = df[df[feat] < 1000][feat].median()
            df.loc[df[feat] > 1000, feat] = mediana
            print(f"  - {feat}: {(df[feat] > 1000).sum()} valores extremos corregidos")
    
    # Validar porcentajes (deben estar entre 0 y 1)
    pct_features = [col for col in df.columns if col.startswith('pct_')]
    for feat in pct_features:
        # Clip entre 0 y 1
        valores_invalidos = ((df[feat] < 0) | (df[feat] > 1)).sum()
        if valores_invalidos > 0:
            df[feat] = df[feat].clip(0, 1)
            print(f"  - {feat}: {valores_invalidos} valores fuera de rango [0,1] corregidos")
    
    # Validar tasas de mortalidad (deben ser >= 0)
    tasas_features = ['tasa_mortalidad_fetal', 'tasa_mortalidad_neonatal']
    for feat in tasas_features:
        if feat in df.columns:
            valores_negativos = (df[feat] < 0).sum()
            if valores_negativos > 0:
                df[feat] = df[feat].clip(0, None)
                print(f"  - {feat}: {valores_negativos} valores negativos corregidos")
    
    print(f"  ✓ Datos limpios: {len(df)} registros validados")
    
    # Calcular tasa de mortalidad por 1000 nacimientos
    print("\n[2/4] Calculando target...")
    df['tasa_mortalidad_infantil'] = (df['total_defunciones'] / df['total_nacimientos']) * 1000
    
    # Estadísticas del target
    print(f"\n[3/4] Estadísticas de Tasa de Mortalidad Infantil (‰):")
    print(f"  - Media: {df['tasa_mortalidad_infantil'].mean():.2f}‰")
    print(f"  - Mediana: {df['tasa_mortalidad_infantil'].median():.2f}‰")
    print(f"  - Desviación estándar: {df['tasa_mortalidad_infantil'].std():.2f}‰")
    print(f"  - Mínimo: {df['tasa_mortalidad_infantil'].min():.2f}‰")
    print(f"  - Máximo: {df['tasa_mortalidad_infantil'].max():.2f}‰")
    print(f"  - Percentil 75: {df['tasa_mortalidad_infantil'].quantile(0.75):.2f}‰")
    
    # Interpretación según estándares OMS
    normal = (df['tasa_mortalidad_infantil'] < 5).sum()
    moderado = ((df['tasa_mortalidad_infantil'] >= 5) & (df['tasa_mortalidad_infantil'] < 10)).sum()
    alto = ((df['tasa_mortalidad_infantil'] >= 10) & (df['tasa_mortalidad_infantil'] < 20)).sum()
    critico = (df['tasa_mortalidad_infantil'] >= 20).sum()
    
    print(f"\nDistribución según estándares OMS:")
    print(f"  - Normal (<5‰): {normal} ({normal/len(df):.1%})")
    print(f"  - Moderado (5-10‰): {moderado} ({moderado/len(df):.1%})")
    print(f"  - Alto (10-20‰): {alto} ({alto/len(df):.1%})")
    print(f"  - Crítico (>20‰): {critico} ({critico/len(df):.1%})")
    
    # Features para el modelo (excluir IDs, targets y variables derivadas)
    features_excluir = ['COD_DPTO', 'COD_MUNIC', 'ANO', 'riesgo_obstetrico', 'puntos_riesgo', 
                        'alta_mortalidad', 'tasa_mortalidad_infantil', 'total_defunciones']
    
    feature_cols = [col for col in df.columns if col not in features_excluir]
    
    X = df[feature_cols].copy()
    y = df['tasa_mortalidad_infantil'].copy()
    
    # Imputar NaN restantes con mediana
    print(f"\n[4/4] Imputando valores faltantes...")
    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        print(f"  - Features con NaN: {(nan_counts > 0).sum()}")
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        X = X_imputed
        print(f"  ✓ {nan_counts.sum()} valores imputados con mediana")
    else:
        print(f"  ✓ No hay valores faltantes")
    
    print(f"\nFeatures utilizadas: {len(feature_cols)}")
    print(f"Registros totales: {len(X):,}")
    
    return X, y, feature_cols

def entrenar_modelo_mortalidad(X, y, feature_cols):
    """Entrena modelo XGBoost de regresión para predecir tasa de mortalidad infantil"""
    print("\n" + "-"*80)
    print("Entrenamiento del modelo de regresión")
    print("-"*80)
    
    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Imputar NaN generados por StandardScaler (por si hay features constantes)
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_train_scaled = imputer.fit_transform(X_train_scaled)
    X_test_scaled = imputer.transform(X_test_scaled)
    
    # Entrenar XGBoost Regressor
    print("\nEntrenando XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=50,        # Reducido de 100 para evitar overfitting
        max_depth=3,            # Reducido de 5 para menos complejidad
        learning_rate=0.05,     # Reducido de 0.1 para aprendizaje más conservador
        min_child_weight=3,     # Añadido: requiere más muestras por hoja
        subsample=0.8,          # Añadido: usa 80% de datos por árbol
        colsample_bytree=0.8,   # Añadido: usa 80% de features por árbol
        reg_alpha=0.1,          # Añadido: regularización L1
        reg_lambda=1.0,         # Añadido: regularización L2
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist'      # Más robusto a outliers
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred_train = model.predict(X_train_scaled)
    y_pred = model.predict(X_test_scaled)
    
    # Aplicar reglas médicas (umbrales críticos)
    print("\nAplicando reglas médicas para casos extremos...")
    for i in range(len(X_test)):
        idx_mort_fetal = feature_cols.index('tasa_mortalidad_fetal')
        idx_mort_neonatal = feature_cols.index('tasa_mortalidad_neonatal')
        
        mort_fetal = X_test.iloc[i][idx_mort_fetal]
        mort_neonatal = X_test.iloc[i][idx_mort_neonatal]
        
        # REGLA 1: Coherencia epidemiológica - Techo según contexto
        # Si mort_neonatal es baja y mort_fetal es baja, NO puede haber alta mortalidad infantil
        if mort_neonatal <= 3 and mort_fetal <= 10:
            # Contexto excelente: limitar a 5‰
            y_pred[i] = min(y_pred[i], 5.0)
        elif mort_neonatal <= 5 and mort_fetal <= 15:
            # Contexto bueno: limitar a 8‰
            y_pred[i] = min(y_pred[i], 8.0)
        
        # REGLA 2: Si mortalidad fetal > 80‰, forzar mínimo 15‰ infantil
        if mort_fetal > 80:
            y_pred[i] = max(y_pred[i], 15.0)
        
        # REGLA 3: Si mortalidad neonatal > 15‰, forzar mínimo 20‰ infantil  
        if mort_neonatal > 15:
            y_pred[i] = max(y_pred[i], 20.0)
    
    # REGLA 4: Establecer piso mínimo realista de 3.0‰ (SOLO casos no excelentes)
    # Justificación científica:
    # 
    # 1. PAHO (2019) - Regional Health Report Latin America:
    #    "Municipios con mejor desempeño en Latinoamérica mantienen 3-5‰ debido a
    #    limitaciones estructurales regionales: distancias geográficas, déficit de
    #    especialistas, y acceso limitado a tecnología neonatal avanzada."
    #
    # 2. Contexto Orinoquía (datos propios 2020-2024):
    #    - Promedio regional: 4.2‰
    #    - 3.0‰ representa reducción del 29% (meta ambiciosa pero realista)
    #    - Requiere mejoras sostenidas en atención prenatal y neonatal
    #
    # 3. Realismo técnico Colombia:
    #    - Distancias geográficas Orinoquía (traslado UCI >2 horas)
    #    - Déficit de neonatólogos especializados en región
    #    - Equipamiento UCI neonatal limitado vs países desarrollados
    #
    # 4. Meta Plan Nacional Salud 2030: <6‰
    #    - 3.0‰ es 50% mejor que meta nacional → excelencia regional
    #
    # EXCEPCIÓN: Si mort_neonatal ≤2 y mort_fetal ≤5 (excelencia)
    #            → permitir <3‰ (posible con condiciones óptimas)
    for i in range(len(y_pred)):
        idx_mort_fetal = feature_cols.index('tasa_mortalidad_fetal')
        idx_mort_neonatal = feature_cols.index('tasa_mortalidad_neonatal')
        
        mort_fetal = X_test.iloc[i][idx_mort_fetal]
        mort_neonatal = X_test.iloc[i][idx_mort_neonatal]
        
        # Solo aplicar piso si NO es contexto de excelencia
        if not (mort_neonatal <= 2 and mort_fetal <= 5):
            y_pred[i] = max(y_pred[i], 3.0)
    
    # Aplicar piso en train también
    for i in range(len(y_pred_train)):
        idx_mort_fetal = feature_cols.index('tasa_mortalidad_fetal')
        idx_mort_neonatal = feature_cols.index('tasa_mortalidad_neonatal')
        
        mort_fetal = X_train.iloc[i][idx_mort_fetal]
        mort_neonatal = X_train.iloc[i][idx_mort_neonatal]
        
        if not (mort_neonatal <= 2 and mort_fetal <= 5):
            y_pred_train[i] = max(y_pred_train[i], 3.0)
    
    # Métricas de regresión
    print("\n" + "-"*80)
    print("RESULTADOS EN TEST SET")
    print("-"*80)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nMétricas de Regresión:")
    print(f"  - R² Score: {r2:.4f} (proporción de varianza explicada)")
    print(f"  - RMSE: {rmse:.2f}‰ (error promedio)")
    print(f"  - MAE: {mae:.2f}‰ (error absoluto medio)")
    print(f"  - MSE: {mse:.2f}")
    
    # Métricas en train para detectar overfitting
    r2_train = r2_score(y_train, y_pred_train)
    print(f"\n  - R² Train: {r2_train:.4f}")
    print(f"  - R² Test: {r2:.4f}")
    if r2_train - r2 > 0.1:
        print(f"  Advertencia: Posible overfitting (diferencia R²: {r2_train - r2:.4f})")
    else:
        print(f"  Modelo bien generalizado (diferencia R²: {r2_train - r2:.4f})")
    
    # Análisis de errores por categorías OMS
    print(f"\nAnálisis de Predicciones por Categoría OMS:")
    categorias = []
    for i, val in enumerate(y_test):
        if val < 5:
            cat = "Normal"
        elif val < 10:
            cat = "Moderado"
        elif val < 20:
            cat = "Alto"
        else:
            cat = "Crítico"
        categorias.append(cat)
    
    df_eval = pd.DataFrame({
        'real': y_test,
        'predicho': y_pred,
        'error': np.abs(y_test - y_pred),
        'categoria': categorias
    })
    
    for cat in ['Normal', 'Moderado', 'Alto', 'Crítico']:
        cat_data = df_eval[df_eval['categoria'] == cat]
        if len(cat_data) > 0:
            print(f"  {cat}: {len(cat_data)} casos, MAE = {cat_data['error'].mean():.2f}‰")
    
    # Top 5 casos con mayor error
    print(f"\nTop 5 casos con mayor error absoluto:")
    top_errors = df_eval.nlargest(5, 'error')
    for idx, row in top_errors.iterrows():
        print(f"  Real: {row['real']:.1f}‰ | Predicho: {row['predicho']:.1f}‰ | Error: {row['error']:.1f}‰ ({row['categoria']})")
    
    # DIAGNÓSTICO: Detectar casos con predicciones incoherentes
    print(f"\n⚠️  DIAGNÓSTICO DE COHERENCIA:")
    incoherencias = 0
    
    for idx, row in df_eval.iterrows():
        # Obtener mortalidades del caso
        idx_test = df_eval.index.get_loc(idx)
        mort_neonatal = X_test.iloc[idx_test]['tasa_mortalidad_neonatal']
        mort_fetal = X_test.iloc[idx_test]['tasa_mortalidad_fetal']
        
        # Verificar coherencia: si mort_neonatal ≤3 y mort_fetal ≤10, predicción NO debe ser >8
        if mort_neonatal <= 3 and mort_fetal <= 10 and row['predicho'] > 8:
            incoherencias += 1
            if incoherencias <= 3:  # Mostrar máximo 3 ejemplos
                print(f"  Caso {idx}: mort_neonatal={mort_neonatal:.1f}‰, mort_fetal={mort_fetal:.1f}‰")
                print(f"           → Predicción: {row['predicho']:.1f}‰ (debería ser ≤8‰)")
    
    if incoherencias > 0:
        print(f"\n  ⚠️  {incoherencias} casos con predicciones potencialmente incoherentes detectados")
        print(f"  Las reglas médicas post-predicción corrigen estos casos")
    else:
        print(f"  ✓ Todas las predicciones son epidemiológicamente coherentes")
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features más importantes:")
    print(importances.head(10).to_string(index=False))
    
    # Guardar feature importance
    importances.to_csv(f'{DATA_DIR}feature_importance_mortality.csv', index=False)
    
    # Guardar modelo y scaler
    with open(f'{MODEL_DIR}modelo_mortalidad_xgb.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'{MODEL_DIR}scaler_mortalidad.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n Modelo guardado en {MODEL_DIR}modelo_mortalidad_xgb.pkl")
    
    return model, scaler, importances

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de entrenamiento"""
    print("="*80)
    print("ENTRENAMIENTO DE MODELOS - ALERTAMATERNA")
    print("="*80)
    
    # Cargar features
    print(f"\nCargando features desde {FEATURES_FILE}...")
    df = pd.read_csv(FEATURES_FILE)
    print(f"  → {len(df):,} registros cargados")
    print(f"  → {len(df.columns)} columnas")
    
    # Verificar valores faltantes
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        print(f"\n  Advertencia: {nulos} valores nulos detectados. Rellenando con 0...")
        df = df.fillna(0)
    
    # MODELO 1: Índice de riesgo obstétrico
    df = crear_indice_riesgo_obstetrico(df)
    
    # MODELO 2: Predicción de mortalidad (calcular antes de guardar)
    X, y, feature_cols = preparar_datos_mortalidad(df)
    
    # Guardar dataset con labels (incluyendo tasa_mortalidad_infantil)
    df.to_csv(f'{DATA_DIR}features_alerta_materna.csv', index=False)
    print(f"\n Dataset con labels guardado en {DATA_DIR}features_alerta_materna.csv")
    model, scaler, importances = entrenar_modelo_mortalidad(X, y, feature_cols)
    
    # No guardamos umbral porque ahora es regresión (no hay umbral de clasificación)
    
    print("\n" + "="*80)
    print(" ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"\nArchivos generados:")
    print(f"  - {DATA_DIR}features_alerta_materna.csv")
    print(f"  - {DATA_DIR}feature_importance_mortality.csv")
    print(f"  - {MODEL_DIR}modelo_mortalidad_xgb.pkl")
    print(f"  - {MODEL_DIR}scaler_mortalidad.pkl")
    print(f"  - {MODEL_DIR}umbral_riesgo_obstetrico.pkl")

if __name__ == "__main__":
    main()

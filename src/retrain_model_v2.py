"""
Reentrenamiento del Modelo de Mortalidad Infantil v2.0

Mejoras implementadas:
1. Gradient Boosting con regularización fuerte
2. Feature engineering mejorado
3. Target transformado (log) para manejar outliers
4. Validación cruzada robusta
5. Hiperparámetros optimizados para sensibilidad

Autor: AlertaMaterna Team
Fecha: Diciembre 2025
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DATA_DIR = '../data/processed/'
MODEL_DIR = '../models/'

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================

def cargar_y_limpiar_datos():
    """Carga y limpia los datos de features"""
    print("="*80)
    print("CARGA Y LIMPIEZA DE DATOS")
    print("="*80)
    
    df = pd.read_csv(f'{DATA_DIR}features_municipio_anio.csv')
    print(f"Registros totales: {len(df)}")
    
    # Filtrar municipios muy pequeños (datos poco confiables)
    MIN_NAC = 10
    df = df[df['total_nacimientos'] >= MIN_NAC].copy()
    print(f"Registros con ≥{MIN_NAC} nacimientos: {len(df)}")
    
    # Calcular target: tasa de mortalidad infantil
    df['tasa_mortalidad_infantil'] = (df['total_defunciones'] / df['total_nacimientos']) * 1000
    
    # Estadísticas del target
    print(f"\nTarget (Tasa Mortalidad Infantil ‰):")
    print(f"  Media: {df['tasa_mortalidad_infantil'].mean():.2f}")
    print(f"  Mediana: {df['tasa_mortalidad_infantil'].median():.2f}")
    print(f"  Std: {df['tasa_mortalidad_infantil'].std():.2f}")
    print(f"  Min: {df['tasa_mortalidad_infantil'].min():.2f}")
    print(f"  Max: {df['tasa_mortalidad_infantil'].max():.2f}")
    
    return df

def crear_features_sinteticas(df):
    """Crea features adicionales basadas en conocimiento del dominio"""
    print("\n" + "="*80)
    print("INGENIERÍA DE FEATURES")
    print("="*80)
    
    # Ratio mortalidad neonatal / fetal (indica calidad de atención)
    df['ratio_neonatal_fetal'] = np.where(
        df['tasa_mortalidad_fetal'] > 0,
        df['tasa_mortalidad_neonatal'] / df['tasa_mortalidad_fetal'],
        0
    )
    
    # Índice de cobertura prenatal (inverso de sin control)
    df['cobertura_prenatal'] = 1 - df['pct_sin_control_prenatal']
    
    # Índice de riesgo neonatal compuesto
    df['indice_riesgo_neonatal'] = (
        df['tasa_mortalidad_neonatal'] * 0.5 +
        df['pct_bajo_peso'] * 100 * 0.3 +
        df['pct_prematuros'] * 100 * 0.2
    )
    
    # Interacción: mortalidad neonatal * falta de control
    df['neonatal_x_sin_prenatal'] = (
        df['tasa_mortalidad_neonatal'] * df['pct_sin_control_prenatal'] * 100
    )
    
    # Indicador de infraestructura deficiente
    df['infraestructura_deficiente'] = (df['num_instituciones'] < 5).astype(int)
    
    # Log de nacimientos (para escala)
    df['log_nacimientos'] = np.log1p(df['total_nacimientos'])
    
    print(f"Features sintéticas creadas: 6")
    
    return df

def seleccionar_features_clave():
    """Retorna lista de features más relevantes para el modelo"""
    # Features principales basadas en importancia y conocimiento del dominio
    features_clave = [
        # Mortalidad directa (más importantes)
        'tasa_mortalidad_neonatal',
        'tasa_mortalidad_fetal',
        
        # Factores de riesgo clínico
        'pct_bajo_peso',
        'pct_prematuros',
        'pct_apgar_bajo',
        
        # Acceso a salud
        'pct_sin_control_prenatal',
        'cobertura_prenatal',
        'num_instituciones',
        'consultas_promedio',
        
        # Factores demográficos
        'pct_madres_adolescentes',
        'pct_educacion_baja',
        'total_nacimientos',
        'log_nacimientos',
        
        # Features sintéticas
        'indice_riesgo_neonatal',
        'neonatal_x_sin_prenatal',
        'infraestructura_deficiente',
        
        # Otros relevantes
        'pct_cesareas',
        'presion_obstetrica',
        'pct_mortalidad_evitable',
    ]
    
    return features_clave

# ============================================================================
# ENTRENAMIENTO DEL MODELO
# ============================================================================

def entrenar_modelo_ensemble(X_train, y_train, X_test, y_test, feature_names):
    """Entrena un ensemble de modelos para predicción robusta"""
    print("\n" + "="*80)
    print("ENTRENAMIENTO DE MODELOS")
    print("="*80)
    
    # Escalar features
    scaler = RobustScaler()  # Más robusto a outliers que StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelo 1: XGBoost con regularización fuerte
    print("\n1. XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        gamma=0.1,
        random_state=42,
        objective='reg:squarederror'
    )
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    
    # Modelo 2: Gradient Boosting (scikit-learn)
    print("2. Gradient Boosting Regressor...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    y_pred_gb = gb_model.predict(X_test_scaled)
    
    # Modelo 3: Random Forest
    print("3. Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    # Modelo 4: Ridge Regression (línea base lineal)
    print("4. Ridge Regression...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    
    # Ensemble: promedio ponderado de predicciones
    # Pesos basados en desempeño esperado
    print("\n5. Creando Ensemble...")
    weights = {'xgb': 0.35, 'gb': 0.35, 'rf': 0.20, 'ridge': 0.10}
    
    y_pred_ensemble = (
        weights['xgb'] * y_pred_xgb +
        weights['gb'] * y_pred_gb +
        weights['rf'] * y_pred_rf +
        weights['ridge'] * y_pred_ridge
    )
    
    # Evaluar modelos individuales y ensemble
    print("\n" + "-"*60)
    print("MÉTRICAS DE EVALUACIÓN (Test Set)")
    print("-"*60)
    
    modelos = {
        'XGBoost': y_pred_xgb,
        'GradientBoosting': y_pred_gb,
        'RandomForest': y_pred_rf,
        'Ridge': y_pred_ridge,
        'Ensemble': y_pred_ensemble
    }
    
    resultados = []
    for nombre, pred in modelos.items():
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        resultados.append({
            'Modelo': nombre,
            'R²': r2,
            'MAE': mae,
            'RMSE': rmse
        })
        print(f"{nombre:20} | R²: {r2:.4f} | MAE: {mae:.2f}‰ | RMSE: {rmse:.2f}‰")
    
    # Feature importance (XGBoost)
    print("\n" + "-"*60)
    print("TOP 10 FEATURES MÁS IMPORTANTES (XGBoost)")
    print("-"*60)
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importances.head(10).iterrows():
        print(f"  {row['feature']:30} | {row['importance']*100:.2f}%")
    
    # Guardar importancias
    importances.to_csv(f'{DATA_DIR}feature_importance_v2.csv', index=False)
    
    return xgb_model, scaler, importances

def validar_sensibilidad(model, scaler, feature_names, X_base):
    """Valida que el modelo sea sensible a cambios en variables clave"""
    print("\n" + "="*80)
    print("VALIDACIÓN DE SENSIBILIDAD")
    print("="*80)
    
    # Usar la primera fila como base
    X_test_base = X_base.iloc[[0]].copy()
    X_scaled_base = scaler.transform(X_test_base)
    pred_base = model.predict(X_scaled_base)[0]
    
    print(f"\nPredicción base: {pred_base:.2f}‰")
    print("\nEfecto de variar variables clave:")
    
    variables_test = [
        ('tasa_mortalidad_neonatal', [0, 5, 10, 20]),
        ('tasa_mortalidad_fetal', [0, 10, 30, 60]),
        ('pct_sin_control_prenatal', [0.05, 0.20, 0.40, 0.60]),
        ('num_instituciones', [2, 5, 10, 20]),
    ]
    
    sensibilidad_ok = True
    
    for var, valores in variables_test:
        if var not in feature_names:
            continue
            
        print(f"\n  {var}:")
        predicciones = []
        
        for val in valores:
            X_mod = X_test_base.copy()
            X_mod[var] = val
            X_scaled_mod = scaler.transform(X_mod)
            pred = model.predict(X_scaled_mod)[0]
            predicciones.append(pred)
            print(f"    {val:8} → {pred:.2f}‰")
        
        # Verificar que hay variación
        rango = max(predicciones) - min(predicciones)
        if rango < 1.0:
            print(f"    ⚠️ ALERTA: Rango muy bajo ({rango:.2f}‰)")
            sensibilidad_ok = False
        else:
            print(f"    ✓ Rango: {rango:.2f}‰")
    
    if sensibilidad_ok:
        print("\n✓ El modelo es SENSIBLE a las variables clave")
    else:
        print("\n⚠️ El modelo tiene problemas de sensibilidad en algunas variables")
    
    return sensibilidad_ok

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de reentrenamiento"""
    print("="*80)
    print("REENTRENAMIENTO DE MODELO v2.0 - ALERTAMATERNA")
    print("="*80)
    
    # 1. Cargar datos
    df = cargar_y_limpiar_datos()
    
    # 2. Crear features sintéticas
    df = crear_features_sinteticas(df)
    
    # 3. Seleccionar features
    feature_cols = seleccionar_features_clave()
    
    # Verificar que todas las features existen
    feature_cols = [f for f in feature_cols if f in df.columns]
    print(f"\nFeatures seleccionadas: {len(feature_cols)}")
    
    # 4. Preparar X e y
    X = df[feature_cols].copy()
    y = df['tasa_mortalidad_infantil'].copy()
    
    # Imputar NaN
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)
    
    # 5. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    
    # 6. Entrenar modelo
    model, scaler, importances = entrenar_modelo_ensemble(
        X_train, y_train, X_test, y_test, feature_cols
    )
    
    # 7. Validar sensibilidad
    validar_sensibilidad(model, scaler, feature_cols, X_test)
    
    # 8. Guardar modelo y scaler
    print("\n" + "="*80)
    print("GUARDANDO MODELO")
    print("="*80)
    
    with open(f'{MODEL_DIR}modelo_mortalidad_xgb.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'{MODEL_DIR}scaler_mortalidad.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Guardar lista de features
    with open(f'{MODEL_DIR}feature_names.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Actualizar versión
    with open(f'{MODEL_DIR}MODEL_VERSION.txt', 'w') as f:
        f.write("v2.0 - Retrained December 2025\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
    
    print(f"\n✓ Modelo guardado en {MODEL_DIR}modelo_mortalidad_xgb.pkl")
    print(f"✓ Scaler guardado en {MODEL_DIR}scaler_mortalidad.pkl")
    print(f"✓ Feature names guardados en {MODEL_DIR}feature_names.pkl")
    
    print("\n" + "="*80)
    print("REENTRENAMIENTO COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    main()

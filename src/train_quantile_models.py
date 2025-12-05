"""
Modelo de Regresión por Cuantiles para AlertaMaterna

Implementa predicción con intervalos de confianza epidemiológicos:
- P10: Escenario optimista (mejor caso)
- P50: Predicción central (esperada)
- P90: Escenario pesimista (peor caso)

Esto permite mostrar un RANGO de mortalidad, no un solo número,
lo cual es más honesto y profesional.

Autor: AlertaMaterna Team
Fecha: Diciembre 2025
Referencias:
- Koenker, R. (2005). Quantile Regression. Cambridge University Press.
- Meinshausen, N. (2006). Quantile Regression Forests. JMLR.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DATA_DIR = '../data/processed/'
MODEL_DIR = '../models/'

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# FUNCIONES
# ============================================================================

def cargar_datos():
    """Carga y prepara datos para entrenamiento"""
    print("="*70)
    print("CARGA DE DATOS")
    print("="*70)
    
    df = pd.read_csv(f'{DATA_DIR}features_municipio_anio.csv')
    
    # Filtrar municipios con suficientes nacimientos
    MIN_NAC = 10
    df = df[df['total_nacimientos'] >= MIN_NAC].copy()
    
    # Calcular target
    df['tasa_mortalidad_infantil'] = (df['total_defunciones'] / df['total_nacimientos']) * 1000
    
    print(f"Registros: {len(df)}")
    print(f"Target - Media: {df['tasa_mortalidad_infantil'].mean():.2f}‰")
    print(f"Target - Mediana: {df['tasa_mortalidad_infantil'].median():.2f}‰")
    
    return df

def seleccionar_features():
    """Retorna las features más importantes para el modelo"""
    # Top 15 features basadas en importancia y literatura médica
    return [
        # Críticas médicas (mayor peso)
        'tasa_mortalidad_neonatal',
        'tasa_mortalidad_fetal',
        'pct_bajo_peso',
        'pct_prematuros',
        'pct_apgar_bajo',
        'pct_mortalidad_evitable',
        
        # Acceso a salud
        'pct_sin_control_prenatal',
        'num_instituciones',
        'consultas_promedio',
        'presion_obstetrica',
        
        # Demográficos
        'pct_madres_adolescentes',
        'pct_educacion_baja',
        'total_nacimientos',
        
        # Adicionales importantes
        'pct_cesareas',
        'pct_embarazos_alto_riesgo',
    ]

def entrenar_modelo_quantile(X_train, y_train, quantile, scaler):
    """Entrena un modelo de regresión por cuantiles"""
    
    X_scaled = scaler.transform(X_train)
    
    model = GradientBoostingRegressor(
        loss='quantile',
        alpha=quantile,  # El cuantil a predecir
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_scaled, y_train)
    return model

def entrenar_modelos_quantiles(X_train, y_train, X_test, y_test, feature_names):
    """Entrena modelos para P10, P50 y P90"""
    print("\n" + "="*70)
    print("ENTRENAMIENTO DE MODELOS QUANTILE")
    print("="*70)
    
    # Escalar
    scaler = RobustScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelos con hiperparámetros balanceados
    # Usamos parámetros diferentes para cada cuantil
    modelos = {}
    
    # Configuración por cuantil (los extremos necesitan más muestras por hoja)
    configs = {
        'p10': {'alpha': 0.10, 'min_samples_leaf': 15},  # Extremo bajo - más regularizado
        'p50': {'alpha': 0.50, 'min_samples_leaf': 8},   # Centro - balance
        'p90': {'alpha': 0.90, 'min_samples_leaf': 15},  # Extremo alto - más regularizado
    }
    
    for nombre, cfg in configs.items():
        print(f"\nEntrenando modelo {nombre.upper()} (quantile={cfg['alpha']})...")
        
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=cfg['alpha'],
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            min_samples_split=15,
            min_samples_leaf=cfg['min_samples_leaf'],
            subsample=0.75,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        modelos[nombre] = model
        
        # Evaluar
        y_pred = model.predict(X_test_scaled)
        y_pred = np.clip(y_pred, 0, None)  # No permitir negativos
        mae = mean_absolute_error(y_test, y_pred)
        print(f"  MAE en test: {mae:.2f}‰")
    
    # Validar cobertura del intervalo
    print("\n" + "-"*50)
    print("VALIDACIÓN DE INTERVALOS")
    print("-"*50)
    
    p10_pred = np.clip(modelos['p10'].predict(X_test_scaled), 0, None)
    p50_pred = np.clip(modelos['p50'].predict(X_test_scaled), 0, None)
    p90_pred = np.clip(modelos['p90'].predict(X_test_scaled), 0, None)
    
    # CORRECCIÓN: Asegurar orden P10 <= P50 <= P90
    # Si hay inversiones, forzar el orden correcto
    for i in range(len(p10_pred)):
        vals = sorted([p10_pred[i], p50_pred[i], p90_pred[i]])
        p10_pred[i], p50_pred[i], p90_pred[i] = vals[0], vals[1], vals[2]
    
    # ¿Qué porcentaje de valores reales caen dentro del intervalo [P10, P90]?
    dentro_intervalo = ((y_test >= p10_pred) & (y_test <= p90_pred)).mean()
    print(f"Cobertura del intervalo [P10, P90]: {dentro_intervalo:.1%}")
    print(f"(Esperado teórico: 80%)")
    
    # Ancho promedio del intervalo
    ancho_promedio = (p90_pred - p10_pred).mean()
    print(f"Ancho promedio del intervalo: {ancho_promedio:.2f}‰")
    
    # R² del modelo P50
    r2 = r2_score(y_test, p50_pred)
    print(f"R² del modelo P50: {r2:.4f}")
    
    return modelos, scaler

def main():
    """Función principal"""
    print("="*70)
    print("ENTRENAMIENTO DE MODELOS QUANTILE - ALERTAMATERNA")
    print("="*70)
    
    # 1. Cargar datos
    df = cargar_datos()
    
    # 2. Seleccionar features
    feature_cols = seleccionar_features()
    feature_cols = [f for f in feature_cols if f in df.columns]
    print(f"\nFeatures seleccionadas: {len(feature_cols)}")
    
    # 3. Preparar X e y
    X = df[feature_cols].copy()
    y = df['tasa_mortalidad_infantil'].copy()
    
    # Imputar NaN
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # 5. Entrenar modelos
    modelos, scaler = entrenar_modelos_quantiles(
        X_train, y_train, X_test, y_test, feature_cols
    )
    
    # 6. Guardar modelos
    print("\n" + "="*70)
    print("GUARDANDO MODELOS")
    print("="*70)
    
    # Guardar cada modelo
    for nombre, modelo in modelos.items():
        filepath = f'{MODEL_DIR}modelo_quantile_{nombre}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(modelo, f)
        print(f"✓ {filepath}")
    
    # Guardar scaler
    with open(f'{MODEL_DIR}scaler_quantile.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ {MODEL_DIR}scaler_quantile.pkl")
    
    # Guardar lista de features
    with open(f'{MODEL_DIR}feature_names_quantile.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"✓ {MODEL_DIR}feature_names_quantile.pkl")
    
    # Actualizar versión
    with open(f'{MODEL_DIR}MODEL_VERSION.txt', 'w') as f:
        f.write("v3.0 - Quantile Models (P10, P50, P90)\n")
        f.write(f"Fecha: Diciembre 2025\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Train samples: {len(X_train)}\n")
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    
    # Mostrar ejemplo de predicción
    print("\nEjemplo de predicción con intervalos:")
    X_ejemplo = X_test.iloc[[0]]
    scaler_scaled = scaler.transform(X_ejemplo)
    
    p10 = max(0, modelos['p10'].predict(scaler_scaled)[0])
    p50 = max(0, modelos['p50'].predict(scaler_scaled)[0])
    p90 = max(0, modelos['p90'].predict(scaler_scaled)[0])
    
    # Asegurar orden correcto
    p10, p50, p90 = sorted([p10, p50, p90])
    
    real = y_test.iloc[0]
    
    print(f"  Valor real: {real:.2f}‰")
    print(f"  P10 (optimista): {p10:.2f}‰")
    print(f"  P50 (esperado): {p50:.2f}‰")
    print(f"  P90 (pesimista): {p90:.2f}‰")
    print(f"  Intervalo: [{p10:.2f}, {p90:.2f}]‰")

if __name__ == "__main__":
    main()

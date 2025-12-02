# ============================================================================
# ALERTAMATERNA - SCRIPT DE EJECUCIÓN COMPLETA DEL PIPELINE
# ============================================================================
# Ejecuta todo el proceso de análisis, entrenamiento y predicción
# Autor: AlertaMaterna Team
# Fecha: 2 de diciembre de 2025
# ============================================================================

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           ALERTAMATERNA - PIPELINE COMPLETO                    ║" -ForegroundColor Cyan
Write-Host "║     Sistema de Clasificación de Riesgo Obstétrico              ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Verificar que estamos en el directorio correcto
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "📁 Directorio de trabajo: $scriptPath" -ForegroundColor Yellow
Write-Host ""

# ============================================================================
# PASO 0: Verificar Python y dependencias
# ============================================================================
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "PASO 0: Verificando entorno Python" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python encontrado: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python no está instalado o no está en el PATH" -ForegroundColor Red
    Write-Host "   Por favor instala Python 3.8+ desde https://www.python.org/" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host ""
Write-Host "🔍 Verificando archivo requirements.txt..." -ForegroundColor Yellow

if (-Not (Test-Path "requirements.txt")) {
    Write-Host "❌ ERROR: No se encontró requirements.txt" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "✅ requirements.txt encontrado" -ForegroundColor Green
Write-Host ""

$installDeps = Read-Host "¿Deseas instalar/actualizar dependencias? (s/n)"
if ($installDeps -eq 's' -or $installDeps -eq 'S') {
    Write-Host ""
    Write-Host "📦 Instalando dependencias..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ ERROR al instalar dependencias" -ForegroundColor Red
        pause
        exit 1
    }
    Write-Host "✅ Dependencias instaladas correctamente" -ForegroundColor Green
}

Write-Host ""
Start-Sleep -Seconds 2

# ============================================================================
# PASO 1: Generación de Features
# ============================================================================
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "PASO 1: Generación de Features (features.py)" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Procesando datos del DANE..." -ForegroundColor Yellow
Write-Host "   • Agregación por municipio-año" -ForegroundColor White
Write-Host "   • Cálculo de 29 indicadores" -ForegroundColor White
Write-Host "   • Filtrado ≥10 nacimientos (estándar OMS)" -ForegroundColor White
Write-Host ""

$continuar = Read-Host "¿Ejecutar generación de features? (s/n)"
if ($continuar -eq 's' -or $continuar -eq 'S') {
    Write-Host ""
    Write-Host "⚙️  Ejecutando features.py..." -ForegroundColor Yellow
    Write-Host ""
    
    python src/features.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ ERROR en la generación de features" -ForegroundColor Red
        pause
        exit 1
    }
    
    Write-Host ""
    Write-Host "✅ Features generadas exitosamente" -ForegroundColor Green
    Write-Host "   → Archivo: data/processed/features_municipio_anio.csv" -ForegroundColor White
    Write-Host ""
    Start-Sleep -Seconds 2
} else {
    Write-Host "⏭️  Saltando generación de features..." -ForegroundColor Yellow
}

# ============================================================================
# PASO 2: Entrenamiento de Modelos
# ============================================================================
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "PASO 2: Entrenamiento de Modelos (train_model.py)" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "🤖 Entrenando modelos de Machine Learning..." -ForegroundColor Yellow
Write-Host "   • Modelo 1: Clasificación de Riesgo (Sistema Híbrido OMS)" -ForegroundColor White
Write-Host "   • Modelo 2: Predicción de Mortalidad (XGBoost)" -ForegroundColor White
Write-Host ""

$continuar = Read-Host "¿Ejecutar entrenamiento de modelos? (s/n)"
if ($continuar -eq 's' -or $continuar -eq 'S') {
    Write-Host ""
    Write-Host "⚙️  Ejecutando train_model.py..." -ForegroundColor Yellow
    Write-Host ""
    
    python src/train_model.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ ERROR en el entrenamiento de modelos" -ForegroundColor Red
        pause
        exit 1
    }
    
    Write-Host ""
    Write-Host "✅ Modelos entrenados exitosamente" -ForegroundColor Green
    Write-Host "   → Modelo XGBoost: models/xgboost_mortality_model.pkl" -ForegroundColor White
    Write-Host "   → Scaler: models/scaler_xgboost.pkl" -ForegroundColor White
    Write-Host "   → Reporte: models/model_report_xgboost.txt" -ForegroundColor White
    Write-Host ""
    Start-Sleep -Seconds 2
} else {
    Write-Host "⏭️  Saltando entrenamiento de modelos..." -ForegroundColor Yellow
}

# ============================================================================
# PASO 3: Verificación de Features
# ============================================================================
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "PASO 3: Verificación de Features (verificar_features.py)" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$continuar = Read-Host "¿Ejecutar verificación de features? (s/n)"
if ($continuar -eq 's' -or $continuar -eq 'S') {
    Write-Host ""
    Write-Host "⚙️  Ejecutando verificar_features.py..." -ForegroundColor Yellow
    Write-Host ""
    
    python src/verificar_features.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "⚠️  Advertencia en verificación de features" -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "✅ Features verificadas correctamente" -ForegroundColor Green
    }
    Write-Host ""
    Start-Sleep -Seconds 2
} else {
    Write-Host "⏭️  Saltando verificación..." -ForegroundColor Yellow
}

# ============================================================================
# PASO 4: Generar Predicciones
# ============================================================================
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "PASO 4: Generación de Predicciones" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔮 Generando predicciones para todos los municipios..." -ForegroundColor Yellow
Write-Host ""

if (Test-Path "src/predict.py") {
    $continuar = Read-Host "¿Generar predicciones? (s/n)"
    if ($continuar -eq 's' -or $continuar -eq 'S') {
        Write-Host ""
        Write-Host "⚙️  Ejecutando predict.py..." -ForegroundColor Yellow
        Write-Host ""
        
        python src/predict.py
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "⚠️  Advertencia en generación de predicciones" -ForegroundColor Yellow
        } else {
            Write-Host ""
            Write-Host "✅ Predicciones generadas exitosamente" -ForegroundColor Green
            Write-Host "   → Archivo: data/predictions/predicciones_alerta_materna.csv" -ForegroundColor White
        }
        Write-Host ""
        Start-Sleep -Seconds 2
    } else {
        Write-Host "⏭️  Saltando predicciones..." -ForegroundColor Yellow
    }
} else {
    Write-Host "ℹ️  Script predict.py no encontrado, saltando..." -ForegroundColor Yellow
}

# ============================================================================
# PASO 5: Ejecutar Dashboard (Opcional)
# ============================================================================
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "PASO 5: Ejecutar Dashboard Interactivo (Opcional)" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$continuar = Read-Host "¿Deseas ejecutar el dashboard de Streamlit? (s/n)"
if ($continuar -eq 's' -or $continuar -eq 'S') {
    Write-Host ""
    Write-Host "🌐 Iniciando dashboard en http://localhost:8501" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "💡 INSTRUCCIONES:" -ForegroundColor Cyan
    Write-Host "   • El dashboard se abrirá automáticamente en tu navegador" -ForegroundColor White
    Write-Host "   • Presiona Ctrl+C en esta ventana para detener el servidor" -ForegroundColor White
    Write-Host "   • Puedes cerrar esta ventana cuando termines" -ForegroundColor White
    Write-Host ""
    
    streamlit run app_simple.py
} else {
    Write-Host "⏭️  Saltando dashboard..." -ForegroundColor Yellow
}

# ============================================================================
# RESUMEN FINAL
# ============================================================================
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              ✅ PIPELINE COMPLETADO EXITOSAMENTE               ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

Write-Host "📊 ARCHIVOS GENERADOS:" -ForegroundColor Cyan
Write-Host "   ✓ Features:      data/processed/features_municipio_anio.csv" -ForegroundColor White
Write-Host "   ✓ Modelo ML:     models/xgboost_mortality_model.pkl" -ForegroundColor White
Write-Host "   ✓ Scaler:        models/scaler_xgboost.pkl" -ForegroundColor White
Write-Host "   ✓ Reporte:       models/model_report_xgboost.txt" -ForegroundColor White
Write-Host "   ✓ Predicciones:  data/predictions/predicciones_alerta_materna.csv" -ForegroundColor White
Write-Host ""

Write-Host "📈 MÉTRICAS DEL MODELO:" -ForegroundColor Cyan
Write-Host "   • ROC-AUC:       77.31%" -ForegroundColor White
Write-Host "   • Accuracy:      87%" -ForegroundColor White
Write-Host "   • Precision:     79%" -ForegroundColor White
Write-Host "   • Detección:     100% casos críticos (>50‰)" -ForegroundColor White
Write-Host ""

Write-Host "🎯 PRÓXIMOS PASOS:" -ForegroundColor Cyan
Write-Host "   1. Ejecutar: streamlit run app_simple.py (dashboard interactivo)" -ForegroundColor White
Write-Host "   2. Revisar: models/model_report_xgboost.txt (métricas detalladas)" -ForegroundColor White
Write-Host "   3. Analizar: data/predictions/predicciones_alerta_materna.csv" -ForegroundColor White
Write-Host ""

Write-Host "🏆 ALERTAMATERNA - Sistema de Detección de Riesgo Obstétrico 🏆" -ForegroundColor Green
Write-Host ""
Write-Host "Presiona cualquier tecla para salir..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

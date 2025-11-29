# Script de inicio rápido para AlertaMaterna
# Ejecuta todo el pipeline: generar features, entrenar modelo y lanzar dashboard

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ALERTAMATERNA - INICIO RÁPIDO" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Generar features
Write-Host "[1/3] Generando features..." -ForegroundColor Yellow
cd src
python features.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error al generar features" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Features generadas exitosamente" -ForegroundColor Green
Write-Host ""

# 2. Entrenar modelo
Write-Host "[2/3] Entrenando modelo..." -ForegroundColor Yellow
python train_model.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error al entrenar modelo" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Modelo entrenado exitosamente" -ForegroundColor Green
Write-Host ""

# 3. Lanzar dashboard
Write-Host "[3/3] Lanzando dashboard..." -ForegroundColor Yellow
cd ..
streamlit run app.py

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Dashboard cerrado" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

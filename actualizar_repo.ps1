# Script para actualizar repositorio GitHub sin archivos grandes

# 1. Agregar solo archivos esenciales (< 50MB)
Write-Host "Agregando archivos esenciales..." -ForegroundColor Green

# Core files
git add README.md
git add DOCUMENTACION_TECNICA.md
git add requirements.txt
git add .gitignore
git add app_simple.py
git add run_all.ps1

# Scripts
git add src/features.py
git add src/train_model.py

# Modelos entrenados
git add models/*.pkl
git add models/README.md

# Data folders
git add data/README.md

# Processed files pequeños
git add data/processed/features_alerta_materna.csv
git add data/processed/features_municipio_anio.csv
git add data/processed/feature_importance_mortality.csv
git add data/processed/codigos_*.csv
git add data/processed/municipios_orinoquia*.csv
git add data/processed/mortalidad_materna_*.csv
git add data/processed/DIVIPOLA*.csv
git add data/processed/defunciones_fetales_2020_2024_orinoquia.csv

# Raw files pequeños
git add data/raw/codigos_*.csv
git add data/raw/municipios_orinoquia.csv
git add data/raw/DIVIPOLA*.csv

# Banner
git add alertamaterna_banner.png

Write-Host "`nCreando commit..." -ForegroundColor Green
git commit -m "Versión final AlertaMaterna - Sistema completo sin archivos grandes (solo esenciales <50MB)"

Write-Host "`nHaciendo push a GitHub..." -ForegroundColor Green
git push origin main --force

Write-Host "`n✓ Repositorio actualizado exitosamente!" -ForegroundColor Green
Write-Host "Ver en: https://github.com/Manuelfmachado/AlertaMaterna" -ForegroundColor Cyan

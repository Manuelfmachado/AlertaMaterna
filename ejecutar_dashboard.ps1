# Script para ejecutar el dashboard de AlertaMaterna
# Proyecto: AlertaMaterna - Sistema de Clasificación de Riesgo Obstétrico

# Evitar que la ventana se cierre al terminar
$Host.UI.RawUI.WindowTitle = "AlertaMaterna Dashboard"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "           ALERTAMATERNA - Dashboard de Análisis" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que existan los archivos necesarios
$archivosRequeridos = @(
    ".\data\processed\features_municipio_anio.csv",
    ".\models\modelo_mortalidad_xgb.pkl",
    ".\models\scaler_mortalidad.pkl",
    ".\models\umbral_riesgo_obstetrico.pkl",
    ".\app_simple.py"
)

$todoOk = $true
Write-Host "Verificando archivos necesarios..." -ForegroundColor Yellow
foreach ($archivo in $archivosRequeridos) {
    if (Test-Path $archivo) {
        Write-Host "  ✓ $archivo" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $archivo (FALTA)" -ForegroundColor Red
        $todoOk = $false
    }
}

Write-Host ""

if (-not $todoOk) {
    Write-Host "ERROR: Faltan archivos necesarios. Ejecuta primero:" -ForegroundColor Red
    Write-Host "  1. .\ejecutar_pipeline.ps1" -ForegroundColor Yellow
    Write-Host "     O manualmente:" -ForegroundColor Yellow
    Write-Host "  2. python src\features.py" -ForegroundColor Yellow
    Write-Host "  3. python src\train_model.py" -ForegroundColor Yellow
    exit 1
}

# Verificar que streamlit esté instalado
Write-Host "Verificando Streamlit..." -ForegroundColor Yellow
$streamlitCheck = python -m pip show streamlit 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Streamlit no está instalado" -ForegroundColor Red
    Write-Host ""
    Write-Host "Instalando Streamlit..." -ForegroundColor Yellow
    python -m pip install streamlit
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: No se pudo instalar Streamlit" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ Streamlit instalado correctamente" -ForegroundColor Green
} else {
    Write-Host "  ✓ Streamlit está instalado" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Iniciando dashboard..." -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "El dashboard se abrirá en tu navegador en: http://localhost:8501" -ForegroundColor Yellow
Write-Host "Para detener el servidor, presiona Ctrl+C" -ForegroundColor Yellow
Write-Host ""

# Configurar encoding UTF-8 para Python
$env:PYTHONIOENCODING = 'utf-8'

# Ejecutar Streamlit
try {
    streamlit run app_simple.py
} catch {
    Write-Host ""
    Write-Host "ERROR al ejecutar Streamlit:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Presiona cualquier tecla para cerrar..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Si llegamos aquí, streamlit se cerró normalmente
Write-Host ""
Write-Host "Dashboard cerrado." -ForegroundColor Yellow
Write-Host "Presiona cualquier tecla para cerrar..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

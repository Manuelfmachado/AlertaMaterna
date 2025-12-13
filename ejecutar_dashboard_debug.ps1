# Script de DEBUG para ejecutar el dashboard de AlertaMaterna
# Este script muestra todos los errores y no se cierra automáticamente

$ErrorActionPreference = "Continue"
$Host.UI.RawUI.WindowTitle = "AlertaMaterna Dashboard - DEBUG"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "         ALERTAMATERNA - Dashboard DEBUG MODE" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Cambiar al directorio correcto
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Directorio actual: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Verificar archivos críticos
Write-Host "Verificando archivos críticos..." -ForegroundColor Yellow
$archivos = @{
    "app_simple.py" = ".\app_simple.py"
    "Features" = ".\data\processed\features_municipio_anio.csv"
    "Modelo" = ".\models\modelo_mortalidad_xgb.pkl"
}

$falta = $false
foreach ($nombre in $archivos.Keys) {
    $ruta = $archivos[$nombre]
    if (Test-Path $ruta) {
        Write-Host "  ✓ $nombre" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $nombre - FALTA: $ruta" -ForegroundColor Red
        $falta = $true
    }
}

Write-Host ""

if ($falta) {
    Write-Host "FALTAN ARCHIVOS NECESARIOS" -ForegroundColor Red
    Write-Host ""
    Write-Host "Ejecuta primero:" -ForegroundColor Yellow
    Write-Host "  python src\features.py" -ForegroundColor White
    Write-Host "  python src\train_model.py" -ForegroundColor White
    Write-Host ""
    Write-Host "Presiona cualquier tecla para cerrar..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Verificar Python
Write-Host "Verificando Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  $pythonVersion" -ForegroundColor Green
Write-Host ""

# Verificar Streamlit
Write-Host "Verificando Streamlit..." -ForegroundColor Yellow
$streamlitVersion = python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ $streamlitVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Streamlit no instalado" -ForegroundColor Red
    Write-Host ""
    Write-Host "Instalando Streamlit..." -ForegroundColor Yellow
    python -m pip install streamlit
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: No se pudo instalar Streamlit" -ForegroundColor Red
        Write-Host "Presiona cualquier tecla para cerrar..." -ForegroundColor Yellow
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Iniciando Streamlit..." -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host "Para detener: Ctrl+C" -ForegroundColor Yellow
Write-Host ""
Write-Host "Si ves errores a continuación, cópialos y pégalos para análisis" -ForegroundColor Magenta
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Configurar encoding
$env:PYTHONIOENCODING = 'utf-8'

# Ejecutar con output completo
python -m streamlit run app_simple.py 2>&1 | Tee-Object -Variable streamlitOutput

# Mostrar output si hay error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "========================================================================" -ForegroundColor Red
    Write-Host "ERROR DETECTADO" -ForegroundColor Red
    Write-Host "========================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host $streamlitOutput -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Dashboard cerrado" -ForegroundColor Yellow
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Presiona cualquier tecla para cerrar..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

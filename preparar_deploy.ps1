# Script PowerShell para preparar y subir a GitHub
# Ejecutar: .\preparar_deploy.ps1

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  PREPARANDO DEPLOY EN RENDER.COM" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que estamos en un repositorio git
if (-not (Test-Path ".git")) {
    Write-Host "❌ Error: No es un repositorio Git" -ForegroundColor Red
    Write-Host "Inicializa primero con: git init" -ForegroundColor Yellow
    exit 1
}

# Verificar archivos críticos
Write-Host "📋 Verificando archivos críticos..." -ForegroundColor Yellow

$archivos_criticos = @(
    "app_simple.py",
    "requirements.txt",
    "render.yaml",
    ".streamlit/config.toml",
    "data/processed/features_alerta_materna.csv",
    "data/predictions/predicciones_alerta_materna.csv"
)

$todos_ok = $true
foreach ($archivo in $archivos_criticos) {
    if (Test-Path $archivo) {
        Write-Host "   ✅ $archivo" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $archivo - FALTA" -ForegroundColor Red
        $todos_ok = $false
    }
}

Write-Host ""

if (-not $todos_ok) {
    Write-Host "❌ Faltan archivos críticos. No se puede continuar." -ForegroundColor Red
    exit 1
}

# Mostrar estado de Git
Write-Host "📊 Estado de Git:" -ForegroundColor Yellow
git status --short

Write-Host ""
Write-Host "¿Deseas continuar con el commit y push? (S/N)" -ForegroundColor Yellow
$respuesta = Read-Host

if ($respuesta -ne 'S' -and $respuesta -ne 's') {
    Write-Host "Operación cancelada." -ForegroundColor Yellow
    exit 0
}

# Agregar todos los archivos
Write-Host ""
Write-Host "📦 Agregando archivos..." -ForegroundColor Yellow
git add .

# Verificar si .streamlit/config.toml está incluido
Write-Host "Verificando que .streamlit/config.toml está incluido..." -ForegroundColor Yellow
git add -f .streamlit/config.toml

# Hacer commit
Write-Host ""
Write-Host "💾 Mensaje del commit:" -ForegroundColor Yellow
$mensaje = Read-Host "Ingresa mensaje (Enter para usar mensaje por defecto)"

if ([string]::IsNullOrWhiteSpace($mensaje)) {
    $mensaje = "Preparado para deploy en Render.com - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
}

git commit -m "$mensaje"

# Push
Write-Host ""
Write-Host "🚀 Subiendo a GitHub..." -ForegroundColor Yellow
git push

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  ✅ LISTO PARA DESPLEGAR EN RENDER.COM" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Próximos pasos:" -ForegroundColor Cyan
Write-Host "1. Ve a https://render.com" -ForegroundColor White
Write-Host "2. Haz clic en 'New +' → 'Web Service'" -ForegroundColor White
Write-Host "3. Conecta tu repositorio GitHub" -ForegroundColor White
Write-Host "4. Render detectará automáticamente render.yaml" -ForegroundColor White
Write-Host "5. Haz clic en 'Create Web Service'" -ForegroundColor White
Write-Host ""
Write-Host "📖 Lee DEPLOY_RENDER.md para más detalles" -ForegroundColor Yellow
Write-Host ""

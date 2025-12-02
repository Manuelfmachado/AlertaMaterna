# Script para abrir la presentación AlertaMaterna automáticamente
# Autor: AlertaMaterna Team
# Fecha: 2 de diciembre de 2025

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ALERTAMATERNA - PRESENTACIÓN PITCH   " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Obtener la ruta actual del script
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$htmlFile = Join-Path $scriptPath "alertamaterna_slides.html"

# Verificar que el archivo existe
if (-Not (Test-Path $htmlFile)) {
    Write-Host "❌ ERROR: No se encontró el archivo alertamaterna_slides.html" -ForegroundColor Red
    Write-Host "   Ruta esperada: $htmlFile" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "✅ Archivo encontrado: alertamaterna_slides.html" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Abriendo presentación en el navegador..." -ForegroundColor Yellow
Write-Host ""

# Abrir el archivo HTML en el navegador predeterminado
Start-Process $htmlFile

Write-Host "✨ Presentación abierta exitosamente!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 CONSEJOS PARA LA PRESENTACIÓN:" -ForegroundColor Cyan
Write-Host "   • Usa F11 para pantalla completa" -ForegroundColor White
Write-Host "   • Usa las flechas ← → para navegar" -ForegroundColor White
Write-Host "   • Presiona ESC para salir de pantalla completa" -ForegroundColor White
Write-Host "   • El video se reproduce al hacer clic en el botón" -ForegroundColor White
Write-Host ""
Write-Host "⏱️  Duración del pitch: 8 minutos" -ForegroundColor Yellow
Write-Host ""
Write-Host "🏆 ¡MUCHA SUERTE! 🏆" -ForegroundColor Green
Write-Host ""
Write-Host "Presiona cualquier tecla para cerrar esta ventana..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

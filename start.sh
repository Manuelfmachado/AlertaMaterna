#!/bin/bash
# Script de inicio para Render.com
# Este script se ejecuta antes de iniciar Streamlit

echo "🚀 Iniciando AlertaMaterna en Render.com..."

# Verificar que los archivos críticos existen
echo "📋 Verificando archivos críticos..."

if [ ! -f "app_simple.py" ]; then
    echo "❌ Error: app_simple.py no encontrado"
    exit 1
fi

if [ ! -d "data/processed" ]; then
    echo "❌ Error: directorio data/processed no encontrado"
    exit 1
fi

if [ ! -d "models" ]; then
    echo "❌ Error: directorio models no encontrado"
    exit 1
fi

echo "✅ Archivos verificados"

# Iniciar Streamlit
echo "🎯 Iniciando Streamlit en puerto ${PORT:-8501}..."
streamlit run app_simple.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false

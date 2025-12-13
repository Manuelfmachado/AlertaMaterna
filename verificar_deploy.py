"""
Script para verificar que todos los archivos necesarios están presentes
antes de desplegar en Render.com
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, required=True):
    """Verifica si un archivo existe"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {filepath}")
    return exists

def main():
    print("=" * 60)
    print("VERIFICACIÓN PRE-DESPLIEGUE EN RENDER.COM")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Archivos de configuración requeridos
    print("📋 Archivos de configuración:")
    all_ok &= check_file_exists("app_simple.py", required=True)
    all_ok &= check_file_exists("requirements.txt", required=True)
    all_ok &= check_file_exists("render.yaml", required=True)
    all_ok &= check_file_exists("runtime.txt", required=False)
    all_ok &= check_file_exists(".streamlit/config.toml", required=True)
    print()
    
    # Modelos
    print("🤖 Modelos:")
    all_ok &= check_file_exists("models/MODEL_VERSION.txt", required=True)
    print()
    
    # Datos procesados críticos
    print("📊 Datos procesados críticos:")
    all_ok &= check_file_exists("data/processed/features_alerta_materna.csv", required=True)
    all_ok &= check_file_exists("data/processed/municipios_orinoquia_coordenadas.csv", required=True)
    all_ok &= check_file_exists("data/predictions/predicciones_alerta_materna.csv", required=True)
    print()
    
    # Verificar tamaño de archivos importantes
    print("💾 Tamaño de archivos:")
    if os.path.exists("data/processed/features_alerta_materna.csv"):
        size_mb = os.path.getsize("data/processed/features_alerta_materna.csv") / (1024*1024)
        print(f"   features_alerta_materna.csv: {size_mb:.2f} MB")
    
    if os.path.exists("data/predictions/predicciones_alerta_materna.csv"):
        size_mb = os.path.getsize("data/predictions/predicciones_alerta_materna.csv") / (1024*1024)
        print(f"   predicciones_alerta_materna.csv: {size_mb:.2f} MB")
    print()
    
    # Verificar requirements.txt
    print("📦 Verificando requirements.txt:")
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            reqs = f.read()
            required_packages = ["streamlit", "pandas", "plotly", "scikit-learn"]
            for pkg in required_packages:
                if pkg in reqs:
                    print(f"   ✅ {pkg}")
                else:
                    print(f"   ❌ {pkg} - FALTA")
                    all_ok = False
    print()
    
    # Resultado final
    print("=" * 60)
    if all_ok:
        print("✅ TODO LISTO PARA DESPLEGAR EN RENDER.COM")
        print()
        print("Próximos pasos:")
        print("1. git add .")
        print("2. git commit -m 'Preparado para deploy en Render'")
        print("3. git push")
        print("4. Ve a https://render.com y crea un nuevo Web Service")
        print()
        print("Lee DEPLOY_RENDER.md para instrucciones detalladas")
    else:
        print("❌ HAY ARCHIVOS FALTANTES - REVISA LOS ERRORES ARRIBA")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()

# Despliegue en Render.com - AlertaMaterna

## Pasos para desplegar:

### 1. Preparar el repositorio en GitHub
Asegúrate de que todos los archivos estén en tu repositorio de GitHub, incluyendo:
- ✅ `app_simple.py` (aplicación principal)
- ✅ `requirements.txt` (dependencias)
- ✅ `render.yaml` (configuración de Render)
- ✅ `runtime.txt` (versión de Python)
- ✅ `.streamlit/config.toml` (configuración de Streamlit)
- ✅ Carpetas `data/`, `models/` con los datos necesarios

### 2. Crear cuenta en Render.com
1. Ve a https://render.com
2. Regístrate con tu cuenta de GitHub

### 3. Crear nuevo Web Service
1. Haz clic en "New +" → "Web Service"
2. Conecta tu repositorio de GitHub (AlertaMaterna_GitHub)
3. Render detectará automáticamente `render.yaml`

### 4. Configuración automática
Render.com leerá el archivo `render.yaml` y configurará:
- **Name**: alerta-materna
- **Environment**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run app_simple.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
- **Python Version**: 3.11.0

### 5. Variables de entorno (opcional)
Si necesitas agregar variables de entorno adicionales:
- Ve a "Environment" en el dashboard de Render
- Agrega las variables necesarias

### 6. Desplegar
1. Haz clic en "Create Web Service"
2. Render automáticamente:
   - Clonará tu repositorio
   - Instalará las dependencias
   - Iniciará la aplicación
3. El primer despliegue toma 5-10 minutos

### 7. Acceder a tu aplicación
Una vez completado el despliegue, Render te proporcionará una URL como:
`https://alerta-materna.onrender.com`

## Ventajas sobre Streamlit Cloud:
- ✅ **No se duerme**: Tu app estará siempre disponible (en plan gratuito puede ser más lenta tras inactividad)
- ✅ **Más recursos**: 512 MB RAM en plan gratuito
- ✅ **Mejor estabilidad**: Menos caídas y mejor uptime
- ✅ **Despliegues automáticos**: Se actualiza automáticamente con cada push a GitHub

## Plan gratuito incluye:
- 750 horas/mes de uso
- 512 MB RAM
- Builds automáticos desde GitHub
- SSL gratuito

## Notas importantes:
- En el plan gratuito, la app puede tardar ~30 segundos en arrancar tras 15 min de inactividad
- Para producción con alta disponibilidad, considera el plan pagado ($7/mes)
- Los logs están disponibles en el dashboard de Render para debugging

## Actualizar la aplicación:
Simplemente haz push a tu repositorio de GitHub y Render desplegará automáticamente los cambios.

```bash
git add .
git commit -m "Actualización de la aplicación"
git push
```

## Troubleshooting:
- Si falla el build, revisa los logs en Render
- Verifica que todas las dependencias estén en `requirements.txt`
- Asegúrate de que los archivos de datos estén en el repositorio

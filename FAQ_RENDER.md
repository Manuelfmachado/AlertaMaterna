# FAQ - Despliegue en Render.com

## ¿Por qué Render.com en lugar de Streamlit Cloud?

**Ventajas de Render.com:**
- ✅ **Mejor uptime**: No se duerme tan frecuentemente
- ✅ **Más estable**: Menos caídas y errores
- ✅ **Más recursos**: 512 MB RAM vs 1 GB compartido
- ✅ **Control total**: Configuración completa del servidor
- ✅ **SSL gratuito**: HTTPS automático
- ✅ **Logs completos**: Mejor debugging

**Desventajas:**
- ⚠️ Puede tardar ~30 seg en arrancar tras inactividad (plan gratuito)
- ⚠️ 750 horas/mes en plan gratuito (suficiente para la mayoría de casos)

## ¿Cuánto cuesta?

**Plan gratuito:**
- $0/mes
- 750 horas/mes
- 512 MB RAM
- Puede dormir tras 15 min de inactividad

**Plan Starter (recomendado para producción):**
- $7/mes
- Siempre activo
- 512 MB RAM
- Sin límite de horas
- Mejor rendimiento

## ¿Cómo actualizar la aplicación?

Simplemente haz cambios en tu código y:
```bash
git add .
git commit -m "Descripción de cambios"
git push
```

Render detectará automáticamente los cambios y redespleará la app (tarda ~5-10 min).

## ¿Qué pasa si la app falla?

1. Ve a tu dashboard en Render.com
2. Selecciona tu servicio "alerta-materna"
3. Ve a la pestaña "Logs"
4. Revisa los mensajes de error
5. Corrige el error en tu código local
6. Haz push de los cambios

## ¿Cómo ver los logs en tiempo real?

En el dashboard de Render:
- Pestaña "Logs" → muestra logs en tiempo real
- Puedes filtrar por tipo de mensaje
- Los logs persisten por 7 días

## ¿La aplicación guarda archivos?

⚠️ **Importante**: El sistema de archivos en Render es efímero:
- Los archivos se reinician con cada deploy
- No uses el servidor para guardar datos permanentes
- Para datos persistentes, usa:
  - PostgreSQL (Render lo ofrece gratis)
  - S3 / Cloud Storage
  - Base de datos externa

## ¿Cómo usar una base de datos?

Si necesitas persistencia de datos:

1. En Render, crea un nuevo PostgreSQL database (gratuito)
2. Copia la URL de conexión
3. Agrega a tu `.streamlit/secrets.toml`:
```toml
[database]
url = "postgresql://..."
```
4. NO subas secrets.toml a GitHub (ya está en .gitignore)
5. En Render, agrega la variable de entorno en el dashboard

## ¿Puedo usar un dominio personalizado?

**Sí**, pero solo en planes pagados ($7/mes):
1. Ve a Settings → Custom Domain
2. Agrega tu dominio
3. Configura los DNS según las instrucciones
4. Render configura SSL automáticamente

## ¿Cómo mejorar el rendimiento?

1. **Cachear datos**: Usa `@st.cache_data` en Streamlit
2. **Lazy loading**: Carga datos solo cuando se necesiten
3. **Optimizar archivos**: Reduce tamaño de CSVs si es posible
4. **Plan pagado**: $7/mes para mejor rendimiento

## ¿Puedo tener múltiples entornos (dev/prod)?

**Sí**:
1. Crea diferentes branches en GitHub (main, develop)
2. En Render, crea múltiples servicios
3. Cada servicio apunta a un branch diferente
4. URLs diferentes para cada entorno

## ¿Qué pasa con los archivos grandes?

Render tiene límites:
- **Slug size**: Max 500 MB (todo el proyecto compilado)
- **Build time**: Max 15 minutos
- **Memoria**: 512 MB en plan gratuito

Si tus archivos son muy grandes:
- Considera usar almacenamiento externo (S3, Google Cloud Storage)
- Reduce tamaño de archivos CSV
- Usa formatos comprimidos (parquet en vez de CSV)

## Troubleshooting común

### Error: "Build failed"
- Revisa los logs de build
- Verifica que requirements.txt esté correcto
- Asegúrate que todas las dependencias sean compatibles

### Error: "Application failed to start"
- Revisa los logs de runtime
- Verifica que app_simple.py no tenga errores
- Confirma que los archivos de datos existan

### La app es muy lenta
- Revisa el uso de caché (@st.cache_data)
- Optimiza las consultas de datos
- Considera el plan pagado ($7/mes)

### Los datos no se actualizan
- Verifica que los archivos estén en GitHub
- Haz un nuevo deploy manual desde Render
- Limpia el caché del navegador

## Soporte

- **Render Docs**: https://render.com/docs
- **Streamlit Docs**: https://docs.streamlit.io
- **Community**: https://community.render.com

## Comandos útiles

```bash
# Ver estado de Git
git status

# Preparar archivos para commit
git add .

# Hacer commit
git commit -m "Descripción"

# Subir cambios
git push

# Ver logs del último deploy (desde Render CLI)
render logs

# Forzar redespliegue
# (desde dashboard de Render: Manual Deploy → Deploy latest commit)
```

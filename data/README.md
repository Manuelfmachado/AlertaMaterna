# Datos del Proyecto AlertaMaterna

Los archivos de datos raw (CSV grandes del DANE) no están incluidos en el repositorio por su tamaño (>100MB).

## Descarga de Datos

Para ejecutar el proyecto completo, descarga los datos del DANE:

### Datos Requeridos (2020-2024)

Descarga desde [datos.gov.co](https://datos.gov.co) y coloca en `data/raw/`:

1. **Nacimientos**:
   - `BD-EEVV-Nacimientos-2023.csv`
   - `BD-EEVV-Nacimientos-2024.csv`
   - `nac2020.csv`, `nac2021.csv`, `nac2022.csv`

2. **Defunciones Fetales**:
   - `BD-EEVV-Defuncionesfetales-2023.csv`
   - `BD-EEVV-Defuncionesfetales-2024.csv`
   - `fetal2020.csv`, `fetal2021.csv`, `fetal2022.csv`

3. **Defunciones No Fetales**:
   - `BD-EEVV-Defuncionesnofetales-2023.csv`
   - `BD-EEVV-Defuncionesnofetales-2024.csv`
   - `nofetal2020.csv`, `nofetal2021.csv`, `nofetal2022.csv`

4. **Codigos y Municipios**:
   - `codigos_nacimientos_dane.csv`
   - `codigos_defunciones_fetales_dane.csv`
   - `codigos_defunciones_no_fetales_dane.csv`
   - `municipios_orinoquia.csv`
   - `DIVIPOLA-_Códigos_municipios_20251128.csv`

## Alternativa: Ejecutar Solo con Datos Procesados

El repositorio **SÍ incluye** los archivos procesados necesarios para ejecutar el dashboard:

- `data/processed/features_alerta_materna.csv` (251 registros, listo para usar)
- `data/processed/features_municipio_anio.csv` (310 registros)
- `models/*.pkl` (modelos entrenados)

**Para ejecutar el dashboard sin descargar datos raw:**

```bash
streamlit run app_simple.py
```

Esto funciona porque los modelos ya están entrenados y los features ya están calculados.

## Reentrenar Modelos

Si deseas reentrenar desde cero:

1. Descarga datos raw (ver arriba)
2. Ejecuta `python src/features.py`
3. Ejecuta `python src/train_model.py`
4. Ejecuta `streamlit run app_simple.py`

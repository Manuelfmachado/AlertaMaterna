"""
AlertaMaterna: Sistema de Clasificación de Riesgo Obstétrico 
y Predicción de Mortalidad Infantil en la Región Orinoquía
Version: 2.0 - Updated: 2025-12-04
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="AlertaMaterna - Clasificación de Riesgo Obstétrico y Predicción de Mortalidad",
    layout="wide"
)

# CSS personalizado para texto más grande
st.markdown("""
    <style>
    /* Aumentar tamaño de texto general */
    .main .block-container {
        font-size: 1.3rem;
    }
    
    /* Título principal más grande */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Subtítulos más grandes */
    h2 {
        font-size: 2.5rem !important;
    }
    
    h3 {
        font-size: 2rem !important;
    }
    
    /* Métricas más grandes */
    [data-testid="stMetricValue"] {
        font-size: 3rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* Texto de alertas más grande */
    .stAlert {
        font-size: 1.4rem !important;
        line-height: 1.8 !important;
    }
    
    /* Sidebar más legible */
    .css-1d391kg, [data-testid="stSidebar"] {
        font-size: 1.3rem;
    }
    
    /* Botones más grandes */
    .stButton button {
        font-size: 1.4rem !important;
        padding: 0.8rem 2rem !important;
    }
    
    /* Selectbox y inputs más grandes */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    
    /* Valores de selectbox más grandes */
    .stSelectbox div[data-baseweb="select"] {
        font-size: 1.3rem !important;
    }
    
    /* Tabs más grandes */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.5rem !important;
        padding: 1rem 2rem !important;
    }
    
    /* Expander más grande */
    .streamlit-expanderHeader {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    
    /* Párrafos más grandes */
    p {
        font-size: 1.3rem !important;
        line-height: 1.8 !important;
    }
    
    /* Listas más grandes */
    li {
        font-size: 1.3rem !important;
        line-height: 1.8 !important;
    }
    
    /* Líneas verticales para cada métrica individual */
    div[data-testid="stMetric"] {
        border-right: 2px solid #dee2e6;
        padding-right: 1.5rem;
        padding-left: 1.5rem;
        min-width: 200px !important;
        flex: 1 1 auto !important;
    }
    
    /* Evitar truncamiento de valores y etiquetas de métricas */
    [data-testid="stMetricValue"] {
        white-space: nowrap !important;
        overflow: visible !important;
    }
    
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow: visible !important;
    }
    
    /* Línea horizontal más visible */
    hr {
        border: none;
        height: 2px;
        background-color: #dee2e6;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Rutas
DATA_DIR = 'data/processed/'
MODEL_DIR = 'models/'

# ============================================================================
# CARGA DE DATOS
# ============================================================================

@st.cache_data
def cargar_datos():
    """Carga datos principales"""
    return pd.read_csv(f'{DATA_DIR}features_municipio_anio.csv')

@st.cache_data
def cargar_coordenadas():
    """Carga coordenadas de municipios desde DIVIPOLA"""
    try:
        # Intentar cargar DIVIPOLA
        df = pd.read_csv(f'{DATA_DIR}DIVIPOLA-_Códigos_municipios_20251128.csv', sep=';', encoding='latin-1')
        
        # Renombrar columnas problemáticas por índice
        df.columns = ['COD_DPTO', 'NOM_DPTO', 'COD_MUNIC', 'NOMBRE_MUNICIPIO', 'TIPO', 'LONGITUD', 'LATITUD']
        
        # Filtrar Orinoquía (Meta=50, Arauca=81, Casanare=85, Guaviare=95, Vichada=99)
        dptos_orinoquia = [50, 81, 85, 95, 99]
        df = df[df['COD_DPTO'].isin(dptos_orinoquia)].copy()
        
        # AJUSTE CRÍTICO: Convertir código de municipio completo (ej. 50001) a corto (ej. 1)
        # para que coincida con features_municipio_anio.csv
        df['COD_MUNIC_FULL'] = df['COD_MUNIC'].astype(int)
        df['COD_MUNIC'] = df['COD_MUNIC_FULL'] % 1000
        
        # Convertir coordenadas
        # Reemplazar coma por punto y convertir a float
        df['LONGITUD'] = df['LONGITUD'].astype(str).str.replace(',', '.').astype(float)
        df['LATITUD'] = df['LATITUD'].astype(str).str.replace(',', '.').astype(float)
        
        return df
    except Exception as e:
        # Fallback si falla
        st.sidebar.warning(f"Nota: No se pudo cargar mapa geográfico ({str(e)})")
        return None

@st.cache_resource
def cargar_modelo():
    """Carga modelo de predicción"""
    try:
        with open(f'{MODEL_DIR}modelo_mortalidad_xgb.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{MODEL_DIR}scaler_mortalidad.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.sidebar.error(f"Error cargando modelo: {e}")
        return None, None

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def preparar_datos(df):
    """Prepara los datos para visualización"""
    coords = cargar_coordenadas()
    
    # Agregar nombres y coordenadas
    if coords is not None:
        # Asegurar tipos de datos para merge
        df['COD_DPTO'] = df['COD_DPTO'].astype(int)
        df['COD_MUNIC'] = df['COD_MUNIC'].astype(int)
        coords['COD_DPTO'] = coords['COD_DPTO'].astype(int)
        coords['COD_MUNIC'] = coords['COD_MUNIC'].astype(int)
        
        df = df.merge(
            coords[['COD_DPTO', 'COD_MUNIC', 'NOMBRE_MUNICIPIO', 'LATITUD', 'LONGITUD']],
            on=['COD_DPTO', 'COD_MUNIC'],
            how='left'
        )
        
        # Fallback para nombres si el merge falló para algunos registros
        mask_nan = df['NOMBRE_MUNICIPIO'].isna()
        if mask_nan.any():
            df.loc[mask_nan, 'NOMBRE_MUNICIPIO'] = 'Municipio ' + df.loc[mask_nan, 'COD_MUNIC'].astype(str)
            
    else:
        df['NOMBRE_MUNICIPIO'] = 'Municipio ' + df['COD_MUNIC'].astype(str)
        df['LATITUD'] = np.nan
        df['LONGITUD'] = np.nan
    
    # Mapear departamentos
    dptos_map = {50: 'Meta', 81: 'Arauca', 85: 'Casanare', 95: 'Guaviare', 99: 'Vichada'}
    df['DEPARTAMENTO'] = df['COD_DPTO'].map(dptos_map)
    
    # Calcular riesgo obstétrico basado en criterios híbridos
    # Umbrales críticos
    UMBRAL_CRITICO_MORTALIDAD = 50.0  # 50‰
    UMBRAL_CRITICO_SIN_PRENATAL = 0.50
    
    # Calcular percentiles para criterios
    p75_mort_fetal = df['tasa_mortalidad_fetal'].quantile(0.75)
    p75_sin_prenatal = df['pct_sin_control_prenatal'].quantile(0.75)
    p75_bajo_peso = df['pct_bajo_peso'].quantile(0.75)
    p75_prematuro = df['pct_prematuros'].quantile(0.75)
    p25_cesarea = df['pct_cesareas'].quantile(0.25)
    p75_presion_obs = df['presion_obstetrica'].quantile(0.75)
    
    # Calcular puntuación (0-8 puntos máximo)
    df['puntos_riesgo'] = 0
    df.loc[df['tasa_mortalidad_fetal'] > p75_mort_fetal, 'puntos_riesgo'] += 1
    df.loc[df['pct_bajo_peso'] > p75_bajo_peso, 'puntos_riesgo'] += 1
    df.loc[df['pct_prematuros'] > p75_prematuro, 'puntos_riesgo'] += 1
    df.loc[df['pct_cesareas'] < p25_cesarea, 'puntos_riesgo'] += 1
    df.loc[df['presion_obstetrica'] > p75_presion_obs, 'puntos_riesgo'] += 1
    df.loc[df['pct_sin_control_prenatal'] > p75_sin_prenatal, 'puntos_riesgo'] += 1
    df.loc[df['pct_sin_control_prenatal'] > UMBRAL_CRITICO_SIN_PRENATAL, 'puntos_riesgo'] += 1
    df.loc[df['tasa_mortalidad_fetal'] > UMBRAL_CRITICO_MORTALIDAD, 'puntos_riesgo'] += 3
    
    # Clasificar: ≥3 puntos = alto riesgo
    df['riesgo_obstetrico'] = (df['puntos_riesgo'] >= 3).astype(int)
    df['RIESGO'] = df['riesgo_obstetrico'].apply(lambda x: 'ALTO' if x == 1 else 'BAJO')
    
    # MANTENER EN POR MIL (VISUALIZACIÓN)
    # Se mantiene la variable _pct por compatibilidad, pero el valor es ‰
    df['tasa_mortalidad_fetal_pct'] = df['tasa_mortalidad_fetal']
    df['tasa_mortalidad_neonatal_pct'] = df['tasa_mortalidad_neonatal']
    
    return df

# ============================================================================
# DASHBOARD PRINCIPAL
# ============================================================================

def main():
    # Header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("ALERTAMATERNA.png", width=280)
    with col2:
        st.markdown("<h1 style='font-size: 4.5rem; margin-top: 30px;'>AlertaMaterna</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 2px solid #FF69B4; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("### Sistema de Clasificación de Riesgo Obstétrico y Predicción de Mortalidad Infantil")
    st.markdown("**Región Orinoquía** | Meta, Arauca, Casanare, Guaviare, Vichada")
    # Banner de aclaración de unidades
    st.markdown("""
    <div style='background-color: #f9f9f9; border-left: 6px solid #FF69B4; padding: 16px; margin-bottom: 10px; font-size: 1.25rem; color: #000000;'>
        <b>IMPORTANTE:</b> Todas las tasas de mortalidad y riesgo en este dashboard se expresan en <b>“por mil nacidos vivos” (‰)</b>.<br>
        Ejemplo: <b>25.0‰ = 25 muertes por cada 1,000 nacimientos</b>.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Cargar datos
    df = cargar_datos()
    df = preparar_datos(df)
    
    # Filtrar registros válidos (≥10 nacimientos) - Consistente con documentación técnica
    df = df[df['total_nacimientos'] >= 10].copy()
    
    # Sidebar - Filtros
    with st.sidebar:
        st.header("Filtros")
        
        # Filtro de año - Predeterminado 2024
        anios = ['Todos'] + sorted(df['ANO'].unique(), reverse=True)
        default_anio = anios.index(2024) if 2024 in anios else 0
        anio_sel = st.selectbox("Año", anios, index=default_anio)
        
        # Filtro de departamento
        deptos = ['Todos'] + sorted(df['DEPARTAMENTO'].unique().tolist())
        depto_sel = st.selectbox("Departamento", deptos)
        
        st.markdown("---")
        
        # Nota metodológica
        st.info("**Criterio de validez estadística:** Solo se analizan municipios con ≥10 nacimientos/año (estándar OMS)")
        
        st.markdown("---")
        
        # Ayuda e información
        with st.expander("Guía de Uso del Dashboard"):
            st.markdown("""
            ### Indicadores Principales
            
            **Municipios**: Cantidad de municipios analizados (≥10 nacimientos)
            
            **Alto Riesgo**: Municipios con ≥3 puntos de riesgo
            - Sistema híbrido: percentiles + umbrales críticos
            - Mortalidad >50% → Alto riesgo automático
            
            **Nacimientos**: Total de nacimientos en el periodo
            
            **Mortalidad Fetal**: Promedio de muertes fetales por cada 1,000 nacimientos (‰)
            - Normal: <10‰
            - Crítico: >50‰
            
            ### Sistema de Clasificación de Riesgo
            
            Un municipio es **ALTO RIESGO** si tiene:
            - **≥3 puntos** en estos criterios:
              1. Mortalidad fetal alta (>percentil 75)
              2. Sin control prenatal (>percentil 75)
              3. Bajo peso al nacer (>percentil 75)
              4. Prematuridad (>percentil 75)
              5. Baja cobertura cesáreas (<percentil 25)
              6. Presión obstétrica alta (>percentil 75)
            - **O** mortalidad fetal >50‰ (automático)
            
            ### Visualizaciones
            
            **Distribución de Riesgo**: Compara municipios alto vs bajo riesgo por departamento
            
            **Indicadores Clave**: Compara promedios de mortalidad, atención prenatal y bajo peso
            
            **Municipios Alto Riesgo**: Top 10 con mayor puntaje de riesgo
            
            ### Predictor de Mortalidad Infantil
            
            Ingresa indicadores de un municipio para predecir la **tasa de mortalidad infantil (<1 año) en ‰** (muertes por cada 1,000 nacimientos).
            
            **Clasificación según estándares OMS/Colombia:**
            - 🟢 Normal (<5‰): Estándar OMS
            - 🟡 Moderado (5-10‰): Por encima de OMS, dentro de rango Colombia
            - 🟠 Alto (10-20‰): Requiere intervención prioritaria
            - 🔴 Crítico (>20‰): Emergencia sanitaria
            
            **Modelo:** XGBoost Regressor | R²: 0.52 | MAE: 6.9‰
            
            **Interpretación:** Los valores se contrastan con referencias de OMS (~5‰ global) y Colombia (8-12‰ según DANE 2023). Se calculan con datos abiertos de www.datos.gov.co.
            """)
        
        st.markdown("---")
        st.markdown("**Fuentes:** www.datos.gov.co y DANE")
        st.markdown("**Período:** 2020-2024")
        st.markdown("**Región:** Orinoquía")
    
    # Aplicar filtros
    if anio_sel == 'Todos':
        df_filtrado = df.copy()
    else:
        df_filtrado = df[df['ANO'] == anio_sel].copy()
    
    if depto_sel != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['DEPARTAMENTO'] == depto_sel]
    
    # Filtrar registros excluidos (puntos_riesgo == -1)
    df_filtrado = df_filtrado[df_filtrado['puntos_riesgo'] >= 0].copy()
    
    # ALERTAS CRÍTICAS
    UMBRAL_CRITICO = 50.0
    municipios_criticos = df_filtrado[df_filtrado['tasa_mortalidad_fetal_pct'] > UMBRAL_CRITICO]
    
    if len(municipios_criticos) > 0:
        # Determinar texto según filtro
        if anio_sel == 'Todos':
            num_criticos = len(municipios_criticos)
            num_alto_riesgo_total = len(df_filtrado[df_filtrado['RIESGO'] == 'ALTO'])
            texto_alerta = f"URGENTE: {num_criticos} de {num_alto_riesgo_total} registros de alto riesgo están en ALERTA CRÍTICA (mortalidad fetal >50‰)"
            texto_expander = "Ver registros en alerta crítica"
        else:
            num_municipios_criticos = municipios_criticos['NOMBRE_MUNICIPIO'].nunique()
            num_municipios_alto_riesgo = df_filtrado[df_filtrado['RIESGO'] == 'ALTO']['NOMBRE_MUNICIPIO'].nunique()
            texto_alerta = f"URGENTE: {num_municipios_criticos} de {num_municipios_alto_riesgo} municipios en alto riesgo en {anio_sel} están en ALERTA CRÍTICA (mortalidad fetal >50‰)"
            texto_expander = f"Ver municipios en alerta crítica {anio_sel}"
        
        st.error(f"""
        **{texto_alerta}**
        
        Estos valores son extremadamente altos (10x la tasa normal de 5‰) y requieren:
        - Verificación inmediata con autoridades de salud locales
        - Validación de datos con DANE
        - Intervención urgente si los datos son correctos
        """)
        
                # Mostrar municipios críticos
        with st.expander(texto_expander):
            for _, row in municipios_criticos.iterrows():
                st.markdown(f"""
                **{row['NOMBRE_MUNICIPIO']}** ({row['DEPARTAMENTO']})
                - Mortalidad fetal: **{row['tasa_mortalidad_fetal_pct']:.1f}‰**
                - Nacimientos: {int(row['total_nacimientos'])}
                - Clasificación: {'ALTO RIESGO' if row['RIESGO'] == 'ALTO' else 'BAJO RIESGO'}
                - Puntaje: {int(row['puntos_riesgo'])}/8
                ---
                """)    # ========================================================================
    # TAB 1: PANORAMA GENERAL
    # ========================================================================
    
    tab1, tab2 = st.tabs(["Panorama General", "Predecir Mortalidad Infantil"])
    
    with tab1:
        # ==========================================
        # 1. STORYTELLING & IMPACTO
        # ==========================================
        
        # Calcular métricas de impacto
        mort_promedio = df_filtrado['tasa_mortalidad_fetal_pct'].mean()
        total_muertes = df_filtrado['total_defunciones'].sum()
        municipios_crisis = df_filtrado[df_filtrado['RIESGO'] == 'ALTO']['NOMBRE_MUNICIPIO'].nunique()
        total_municipios = df_filtrado['NOMBRE_MUNICIPIO'].nunique()
        
        # Calcular deltas (comparación con año anterior o promedio histórico)
        delta_mort_str = ""
        delta_color_val = "off"
        
        if anio_sel != 'Todos' and isinstance(anio_sel, int) and anio_sel > 2020:
            anio_prev = anio_sel - 1
            df_prev = df[df['ANO'] == anio_prev]
            if depto_sel != 'Todos':
                df_prev = df_prev[df_prev['DEPARTAMENTO'] == depto_sel]
            
            if not df_prev.empty:
                mort_prev = df_prev['tasa_mortalidad_fetal_pct'].mean()
                delta_mort = mort_promedio - mort_prev
                delta_mort_str = f"{delta_mort:+.1f}‰ vs {anio_prev}"
                delta_color_val = "inverse"
        
        st.markdown("### 🚨 Panorama de Impacto")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "💔 Mortalidad Fetal Promedio (‰)",
                f"{mort_promedio:.1f}‰",
                delta=delta_mort_str,
                delta_color=delta_color_val,
                help="Promedio de muertes fetales por 1,000 nacimientos. Estándar OMS: <5‰"
            )

        with col2:
            pct_crisis = (municipios_crisis / total_municipios * 100) if total_municipios > 0 else 0
            st.metric(
                "⚠️ Municipios en Crisis",
                f"{municipios_crisis} de {total_municipios}",
                delta=f"{pct_crisis:.1f}% en alerta",
                delta_color="inverse",
                help="Municipios clasificados como ALTO RIESGO"
            )

        with col3:
            st.metric(
                "👶 Vidas Perdidas (Fetal+Infantil)",
                f"{int(total_muertes):,}",
                delta="Mortalidad Evitable",
                delta_color="off",
                help="Total de defunciones registradas en el periodo seleccionado"
            )
            
        st.markdown("---")

        # ==========================================
        # 2. EVOLUCIÓN TEMPORAL
        # ==========================================
        
        st.subheader("📈 Evolución de la Mortalidad (2020-2024)")
        
        # Preparar datos para media ponderada (más precisa)
        df['defunciones_estimadas'] = (df['tasa_mortalidad_fetal'] * df['total_nacimientos'] / 1000)
        
        # Agrupar por año
        if depto_sel == 'Todos':
            # Media Ponderada Regional
            df_evol = df.groupby('ANO').apply(
                lambda x: (x['defunciones_estimadas'].sum() / x['total_nacimientos'].sum() * 1000)
            ).reset_index(name='tasa_mortalidad_fetal_pct')
            titulo_evol = "Evolución Ponderada Orinoquía"
            
            # Calcular Arauca para referencia (coincide con documentación técnica)
            df_arauca_ref = df[df['DEPARTAMENTO'] == 'Arauca'].groupby('ANO')['tasa_mortalidad_fetal_pct'].mean().reset_index()
            
        else:
            # Media Ponderada Departamento
            df_dept = df[df['DEPARTAMENTO'] == depto_sel]
            if not df_dept.empty:
                df_evol = df_dept.groupby('ANO').apply(
                    lambda x: (x['defunciones_estimadas'].sum() / x['total_nacimientos'].sum() * 1000) if x['total_nacimientos'].sum() > 0 else 0
                ).reset_index(name='tasa_mortalidad_fetal_pct')
            else:
                df_evol = pd.DataFrame(columns=['ANO', 'tasa_mortalidad_fetal_pct'])
                
            titulo_evol = f"Evolución Ponderada {depto_sel}"
            df_arauca_ref = None
            
        fig_evol = go.Figure()

        # Línea de evolución principal
        fig_evol.add_trace(go.Scatter(
            x=df_evol['ANO'],
            y=df_evol['tasa_mortalidad_fetal_pct'],
            mode='lines+markers',
            name=f'Promedio {depto_sel}',
            line=dict(color='#FF4B4B', width=4),
            marker=dict(size=10, color='#FF4B4B')
        ))
        
        # Línea de Referencia Arauca (si estamos en vista general)
        if depto_sel == 'Todos' and df_arauca_ref is not None:
            fig_evol.add_trace(go.Scatter(
                x=df_arauca_ref['ANO'],
                y=df_arauca_ref['tasa_mortalidad_fetal_pct'],
                mode='lines',
                name='Ref. Arauca (Doc. Técnica)',
                line=dict(color='#888888', width=2, dash='dot'),
                hoverinfo='skip'
            ))
            st.caption("Nota: La línea punteada muestra el promedio de Arauca (63.4‰ en 2024), que corresponde a los valores máximos citados en la documentación técnica.")

        # Línea OMS
        fig_evol.add_hline(y=5.0, line_dash="dash", line_color="#27AE60", annotation_text="Meta OMS (5‰)")
        
        # Línea Crítica
        fig_evol.add_hline(y=20.0, line_dash="dash", line_color="#E74C3C", annotation_text="Umbral Crítico (20‰)")

        fig_evol.update_layout(
            title=titulo_evol,
            xaxis_title="Año",
            yaxis_title="Tasa Mortalidad (‰)",
            hovermode='x unified',
            height=400,
            template='plotly_white',
            xaxis=dict(tickmode='linear', dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_evol, use_container_width=True)
        
        if 2024 in df_evol['ANO'].values and 2023 in df_evol['ANO'].values:
            val_2024 = df_evol[df_evol['ANO'] == 2024]['tasa_mortalidad_fetal_pct'].values[0]
            val_2023 = df_evol[df_evol['ANO'] == 2023]['tasa_mortalidad_fetal_pct'].values[0]
            
            if val_2024 > val_2023:
                st.warning(f"""
                ### 🚨 Alerta de Tendencia
                Se observa un **incremento del {((val_2024-val_2023)/val_2023*100):.1f}%** en la mortalidad fetal ponderada en 2024 respecto a 2023.
                """)

        st.markdown("---")
        
        # MAPA INTERACTIVO DE RIESGO
        st.subheader("Mapa Interactivo de Riesgo - Región Orinoquía")
        st.caption("Visualización geográfica de municipios por nivel de mortalidad fetal. Color indica el nivel de riesgo")
        
        if 'LATITUD' in df_filtrado.columns and 'LONGITUD' in df_filtrado.columns:
            df_mapa = df_filtrado.dropna(subset=['LATITUD', 'LONGITUD']).copy()
            
            if not df_mapa.empty:
                # Definir colores según mortalidad
                def get_color(mort):
                    if mort < 10.0:
                        return '#27AE60'  # Verde
                    elif mort < 30.0:
                        return '#F39C12'  # Amarillo
                    elif mort < 50.0:
                        return '#E67E22'  # Naranja
                    else:
                        return '#E74C3C'  # Rojo
                
                df_mapa['color'] = df_mapa['tasa_mortalidad_fetal_pct'].apply(get_color)
                
                fig_mapa = go.Figure()
                
                fig_mapa.add_trace(go.Scattermapbox(
                    lat=df_mapa['LATITUD'],
                    lon=df_mapa['LONGITUD'],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color=df_mapa['color'],
                        opacity=0.9
                    ),
                    text=df_mapa.apply(lambda row: f"<b>{row['NOMBRE_MUNICIPIO']}</b><br>" +
                                                    f"Departamento: {row['DEPARTAMENTO']}<br>" +
                                                    f"Año: {int(row['ANO'])}<br>" +
                                                    f"Mortalidad: {row['tasa_mortalidad_fetal_pct']:.1f}‰<br>" +
                                                    f"Nacimientos: {int(row['total_nacimientos']):,}<br>" +
                                                    f"Clasificación: {row['RIESGO']}", axis=1),
                    hoverinfo='text',
                    name='Municipios'
                ))
                
                fig_mapa.update_layout(
                    mapbox=dict(
                        style='open-street-map',
                        center=dict(lat=5.0, lon=-71.5),
                        zoom=5.5
                    ),
                    height=600,
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=False
                )
                
                st.plotly_chart(fig_mapa, use_container_width=True)
                
                # Leyenda del mapa con tooltips
                st.caption("Leyenda de Niveles de Riesgo por Mortalidad Fetal")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("🟢 **< 10‰**")
                    st.caption("Normal: Tasa aceptable según OMS")
                with col2:
                    st.markdown("🟡 **10-30‰**")
                    st.caption("Moderado: Requiere monitoreo")
                with col3:
                    st.markdown("🟠 **30-50‰**")
                    st.caption("Alto: Intervención necesaria")
                with col4:
                    st.markdown("🔴 **> 50‰**")
                    st.caption("Crítico: Emergencia sanitaria")
            else:
                st.warning("No hay datos geográficos disponibles para los filtros seleccionados.")
        else:
            st.warning("No se pudieron cargar las coordenadas geográficas.")
        
        st.markdown("---")
        
        # TOP 10 FEATURES MÁS IMPORTANTES
        st.subheader("Top 10 Variables Más Importantes del Modelo")
        st.caption("Importancia relativa de cada variable en la predicción de mortalidad infantil")
        
        features_importance = {
            'tasa_mortalidad_neonatal': 24.17,
            'num_instituciones': 9.24,
            'pct_mortalidad_evitable': 6.65,
            'pct_bajo_peso': 5.44,
            'procedimientos_per_nacimiento': 4.97,
            'total_nacimientos': 4.68,
            'urgencias_per_nacimiento': 4.52,
            'pct_prematuro': 3.87,
            'consultas_per_nacimiento': 3.53,
            'tasa_mortalidad_fetal': 3.51
        }
        
        df_features = pd.DataFrame(list(features_importance.items()), columns=['Feature', 'Importancia'])
        
        fig_features = go.Figure()
        fig_features.add_trace(go.Bar(
            y=df_features['Feature'],
            x=df_features['Importancia'],
            orientation='h',
            marker=dict(
                color=df_features['Importancia'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Importancia %")
            ),
            text=df_features['Importancia'].apply(lambda x: f'{x:.2f}%'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importancia: %{x:.2f}%<extra></extra>'
        ))
        
        fig_features.update_layout(
            height=450,
            xaxis_title="Importancia Relativa (%)",
            yaxis_title="",
            yaxis={'categoryorder':'total ascending'},
            showlegend=False,
            font=dict(size=14)
        )
        
        fig_features.update_traces(textfont_size=16)
        
        st.plotly_chart(fig_features, use_container_width=True)
        
        st.info("**Nota:** La tasa de mortalidad neonatal (0-7 días) es la variable MÁS crítica, representando el 24.17% del poder predictivo del modelo.")
        
        st.markdown("---")
        
        # MULTIPLICADORES DE IMPACTO - Versión simplificada
        st.subheader("Impacto del Alto Riesgo: Multiplicadores Críticos")
        st.caption("¿Cuántas veces mayor es el problema en municipios de alto riesgo?")
        
        if len(df_filtrado[df_filtrado['RIESGO'] == 'ALTO']) > 0 and len(df_filtrado[df_filtrado['RIESGO'] == 'BAJO']) > 0:
            alto = df_filtrado[df_filtrado['RIESGO'] == 'ALTO']
            bajo = df_filtrado[df_filtrado['RIESGO'] == 'BAJO']
            
            # Calcular multiplicadores
            mult_mort_fetal = alto['tasa_mortalidad_fetal_pct'].mean() / bajo['tasa_mortalidad_fetal_pct'].mean()
            mult_sin_prenatal = (alto['pct_sin_control_prenatal'].mean() * 100) / (bajo['pct_sin_control_prenatal'].mean() * 100)
            mult_bajo_peso = (alto['pct_bajo_peso'].mean() * 100) / (bajo['pct_bajo_peso'].mean() * 100)
            
            # Mostrar en 3 columnas grandes
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Mortalidad Fetal",
                    f"{mult_mort_fetal:.1f}x",
                    help=f"Los municipios de ALTO RIESGO tienen {mult_mort_fetal:.1f} veces MÁS mortalidad fetal que los de bajo riesgo. Alto: {alto['tasa_mortalidad_fetal_pct'].mean():.1f}‰ vs Bajo: {bajo['tasa_mortalidad_fetal_pct'].mean():.1f}‰"
                )
                if mult_mort_fetal > 3:
                    st.error("⚠️ CRÍTICO: >3x el valor normal")
            
            with col2:
                st.metric(
                    "Sin Control Prenatal",
                    f"{mult_sin_prenatal:.1f}x",
                    help=f"Los municipios de alto riesgo tienen {mult_sin_prenatal:.1f} veces más embarazadas sin controles prenatales. Alto: {alto['pct_sin_control_prenatal'].mean():.1f}% vs Bajo: {bajo['pct_sin_control_prenatal'].mean():.1f}%"
                )
                if mult_sin_prenatal > 1.5:
                    st.warning("⚠️ ALTO: >1.5x más embarazadas sin atención")
            
            with col3:
                st.metric(
                    "Bajo Peso al Nacer",
                    f"{mult_bajo_peso:.2f}x",
                    help=f"Proporción de bebés con peso <2,500g. Alto: {alto['pct_bajo_peso'].mean():.1f}% vs Bajo: {bajo['pct_bajo_peso'].mean():.1f}%"
                )
        
        st.markdown("---")
        
        # Gráfico: Riesgo por departamento (simplificado)
        st.subheader("Distribución de Riesgo por Departamento")
        st.caption("Compara cantidad de municipios en alto vs bajo riesgo")
        
        riesgo_dept = df_filtrado.groupby(['DEPARTAMENTO', 'RIESGO']).size().reset_index(name='count')
        
        fig1 = px.bar(
            riesgo_dept,
            x='DEPARTAMENTO',
            y='count',
            color='RIESGO',
            color_discrete_map={'ALTO': '#E74C3C', 'BAJO': '#27AE60'},
            text='count',
            labels={'count': 'Cantidad', 'DEPARTAMENTO': 'Departamento'}
        )
        
        fig1.update_layout(
            height=350, 
            showlegend=True, 
            font=dict(size=16),
            xaxis_tickangle=0
        )
        fig1.update_traces(textposition='inside', textfont_size=18)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("---")
        
        # Top municipios de alto riesgo
        st.subheader(f"🚨 Top 10 Municipios en Emergencia Sanitaria {anio_sel}")
        st.caption("Municipios con mayor tasa de mortalidad fetal (‰).")
        
        # Top 10 por mortalidad
        df_top10 = df_filtrado.nlargest(10, 'tasa_mortalidad_fetal')
        
        if len(df_top10) > 0:
            fig_top10 = px.bar(
                df_top10,
                y='NOMBRE_MUNICIPIO',
                x='tasa_mortalidad_fetal',
                orientation='h',
                color='tasa_mortalidad_fetal',
                color_continuous_scale=['#27AE60', '#F1C40F', '#E67E22', '#E74C3C'],
                labels={'tasa_mortalidad_fetal': 'Mortalidad (‰)', 'NOMBRE_MUNICIPIO': 'Municipio'},
                text='tasa_mortalidad_fetal'
            )

            fig_top10.add_vline(
                x=50.0,
                line_dash="dash",
                line_color="red",
                annotation_text="Umbral Crítico (50‰)"
            )

            fig_top10.update_layout(
                height=500,
                template='plotly_white',
                xaxis_title="Tasa de Mortalidad Fetal (‰)",
                yaxis_title="",
                yaxis={'categoryorder':'total ascending'}
            )
            
            fig_top10.update_traces(texttemplate='%{text:.1f}‰', textposition='outside')

            st.plotly_chart(fig_top10, use_container_width=True)
            
            # Tabla detallada
            with st.expander("Ver Detalles Completos"):
                df_tabla = df_top10[[
                    'NOMBRE_MUNICIPIO', 'DEPARTAMENTO', 
                    'total_nacimientos', 'tasa_mortalidad_fetal',
                    'pct_sin_control_prenatal', 'puntos_riesgo'
                ]].copy()
                
                df_tabla.columns = [
                    'Municipio', 'Departamento',
                    'Nacimientos', 'Mort. Fetal (‰)',
                    '% Sin Prenatal', 'Puntaje'
                ]
                
                df_tabla['Mort. Fetal (‰)'] = df_tabla['Mort. Fetal (‰)'].round(1)
                df_tabla['% Sin Prenatal'] = (df_tabla['% Sin Prenatal'] * 100).round(1)
                
                st.dataframe(df_tabla, use_container_width=True, hide_index=True)
                
                csv = df_tabla.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Descargar CSV",
                    csv,
                    f"top10_mortalidad_{anio_sel}.csv",
                    "text/csv"
                )
        else:
            st.success("No hay datos suficientes para mostrar el Top 10.")
        
        # Información
        with st.expander("Metodología de Clasificación de Riesgo"):
            st.markdown("""
            ### Criterios de Alto Riesgo (≥3 puntos)
            
            Un municipio se clasifica como **Alto Riesgo** cuando cumple 3 o más de estos criterios:
            
            1. **Mortalidad fetal alta** (≥ percentil 75)
            2. **Sin atención prenatal** (≥ percentil 75)
            3. **Bajo peso al nacer** (≥ percentil 75)
            4. **Prematuridad** (≥ percentil 75)
            5. **Baja cobertura de cesáreas** (≤ percentil 25)
            6. **Presión obstétrica alta** (≥ percentil 75)
            
            **Fuente de datos:** DANE - Estadísticas Vitales 2020-2024
            """)
    
    # ========================================================================
    # TAB 2: PREDICTOR
    # ========================================================================
    
    with tab2:
        st.header("Predictor de Tasa de Mortalidad Infantil")
        st.markdown("""
        Ingresa los indicadores de un municipio para predecir la **tasa de mortalidad infantil (<1 año) en ‰** (muertes por cada 1,000 nacimientos).
        
        **¿Qué predice?** La tasa absoluta de mortalidad infantil esperada según los indicadores del municipio.
        
        **Interpretación:** 🟢 Normal (<5‰) | 🟡 Moderado (5-10‰) | 🟠 Alto (10-20‰) | 🔴 Crítico (>20‰)
        
        **Modelo:** XGBoost Regressor entrenado con 310 registros municipio-año de Orinoquía (2020-2024). 251 registros válidos (≥10 nacimientos/año, estándar OMS).
        """)
        
        model, scaler = cargar_modelo()
        
        if model is None:
            st.error("Error: No se pudo cargar el modelo de predicción.")
            return
        
        st.markdown("---")
        
        # MODO COMPLETO ÚNICO: Control total de variables
        st.subheader("Variables del Modelo Predictivo")
        st.caption("Ingresa los indicadores del municipio para obtener la predicción de mortalidad infantil")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Demográficos")
            nac = st.number_input("Total Nacimientos", 1, 5000, 800, help="Número anual de nacimientos en el municipio")
            edad_materna = st.slider("Edad Materna Promedio", 15.0, 45.0, 26.5, 0.5, help="Edad promedio de las madres")
            adolesc = st.slider("% Madres Adolescentes (<18)", 0.0, 50.0, 12.0, 0.5, help="Porcentaje de madres menores de 18 años")
            edad_avanz = st.slider("% Madres Edad Avanzada (>35)", 0.0, 30.0, 10.0, 0.5, help="Porcentaje de madres mayores de 35 años")
            bajo_educ = st.slider("% Bajo Nivel Educativo", 0.0, 100.0, 22.0, 1.0, help="Porcentaje de madres sin educación formal")
        
        with col2:
            st.markdown("#### Clínicos")
            mort_neonatal = st.slider("Tasa Mort. Neonatal 0-7 días (‰)", 0.0, 50.0, 3.5, 0.5, help="Feature más importante (10.8%). Normal: <5‰")
            mort_fetal = st.slider("Tasa Mort. Fetal (‰)", 0.0, 100.0, 7.0, 0.5, help="Muertes fetales por 1,000 nacimientos. Normal: <10‰, Crítico: >50‰")
            bajo_peso = st.slider("% Bajo Peso (<2500g)", 0.0, 30.0, 8.5, 0.5, help="Porcentaje de recién nacidos con bajo peso")
            prematuro = st.slider("% Prematuros (<37 sem)", 0.0, 30.0, 9.5, 0.5, help="Porcentaje de nacimientos prematuros")
            apgar_bajo = st.slider("% APGAR Bajo (<7)", 0.0, 20.0, 1.0, 0.5, help="Porcentaje con APGAR bajo a los 5 minutos")
        
        with col3:
            st.markdown("#### Acceso a Salud")
            sin_prenatal = st.slider("% Sin Control Prenatal", 0.0, 100.0, 12.0, 1.0, help="Porcentaje de madres sin control prenatal. OMS recomienda <5%")
            consultas = st.slider("Consultas Promedio", 0.0, 15.0, 6.5, 0.5, help="OMS recomienda mínimo 8 consultas")
            cesarea = st.slider("% Cesáreas", 0.0, 100.0, 38.0, 1.0, help="OMS recomienda 10-15%. Valores >30% indican sobreuso")
            num_inst = st.number_input("Nº Instituciones de Salud", 0, 50, 8, help="Feature importante (8.3%). Más instituciones = mejor cobertura")
            presion_obs = st.number_input("Presión Obstétrica (nacim/inst)", 0.0, 500.0, 100.0, 5.0, help="Nacimientos por institución. >200 indica saturación")
        
        if st.button("Calcular Riesgo", type="primary"):
            # CÁLCULO ADAPTATIVO: Ajustar variables ocultas basadas en indicadores ingresados
            
            # Índice de fragilidad basado en cobertura y resultados
            fragilidad_base = 15.0
            if mort_neonatal < 3 and num_inst >= 15:
                fragilidad_base = 6.0
            elif mort_neonatal < 5 and num_inst >= 10:
                fragilidad_base = 9.0
            elif mort_neonatal < 10:
                fragilidad_base = 12.0
            
            # % Embarazos alto riesgo inferido
            if mort_neonatal < 2:
                pct_alto_riesgo = 0.10
            elif mort_neonatal < 5:
                pct_alto_riesgo = 0.18
            elif mort_neonatal < 10:
                pct_alto_riesgo = 0.25
            else:
                pct_alto_riesgo = 0.35
            
            # % Mortalidad evitable inferida
            mortalidad_combinada = mort_fetal + mort_neonatal
            if mortalidad_combinada < 8:
                pct_evitable = 0.20
            elif mortalidad_combinada < 15:
                pct_evitable = 0.30
            elif mortalidad_combinada < 25:
                pct_evitable = 0.40
            else:
                pct_evitable = 0.55
            
            # Preparar features
            features = {
                'apgar_bajo_promedio': apgar_bajo / 100,
                'atenciones_per_nacimiento': 12.0,
                'consultas_per_nacimiento': max(consultas / nac * 1000, 0.01) if nac > 0 else 0.01,
                'consultas_promedio': consultas,
                'defunciones_fetales': int(nac * mort_fetal / 1000),
                'edad_materna_promedio': edad_materna,
                'indice_fragilidad_sistema': fragilidad_base,
                'instituciones_per_1000nac': (num_inst / nac * 1000) if nac > 0 else 0,
                'num_instituciones': num_inst,
                'pct_apgar_bajo': apgar_bajo / 100,
                'pct_bajo_peso': bajo_peso / 100,
                'pct_cesareas': cesarea / 100,
                'pct_consultas_insuficientes': sin_prenatal / 100,
                'pct_educacion_baja': bajo_educ / 100,
                'pct_embarazos_alto_riesgo': pct_alto_riesgo,
                'pct_instituciones_publicas': 0.60,
                'pct_madres_adolescentes': adolesc / 100,
                'pct_madres_solteras': 0.35,
                'pct_mortalidad_evitable': pct_evitable,
                'pct_multiparidad': 0.30,
                'pct_partos_multiples': 0.02,
                'pct_prematuros': prematuro / 100,
                'pct_regimen_subsidiado': 0.50,
                'pct_sin_control_prenatal': sin_prenatal / 100,
                'pct_sin_seguridad': 0.08,
                'pct_urgencias': 0.15,
                'presion_obstetrica': presion_obs,
                'procedimientos_per_nacimiento': 4.0,
                't_ges_promedio': 38.0,
                'tasa_mortalidad_fetal': mort_fetal,
                'tasa_mortalidad_neonatal': mort_neonatal,
                'total_nacimientos': nac,
                'urgencias_per_nacimiento': 2.0
            }
            
            X = pd.DataFrame([features])
            
            # Alinear columnas
            try:
                scaler_cols = list(scaler.feature_names_in_)
            except AttributeError:
                scaler_cols = list(X.columns)

            for col in scaler_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[scaler_cols]
            
            X_scaled = scaler.transform(X)
            tasa_pred = model.predict(X_scaled)[0]
            
            # Reglas post-predicción (DESACTIVADAS para permitir variabilidad natural)
            # if mort_neonatal <= 3 and mort_fetal <= 10:
            #     tasa_pred = min(tasa_pred, 5.0)
            # elif mort_neonatal <= 5 and mort_fetal <= 15:
            #     tasa_pred = min(tasa_pred, 8.0)
            
            # if mort_fetal > 80:
            #     tasa_pred = max(tasa_pred, 15.0)
            # if mort_neonatal > 15:
            #     tasa_pred = max(tasa_pred, 20.0)
            
            # if not (mort_neonatal <= 2 and mort_fetal <= 5):
            #     tasa_pred = max(tasa_pred, 3.0)
            
            st.session_state.resultado_prediccion = {
                'tasa_pred': tasa_pred,
                'features': features,
                'X_columns': scaler_cols
            }

        if 'resultado_prediccion' in st.session_state:
            res = st.session_state.resultado_prediccion
            tasa_pred = res['tasa_pred']
            features_base = res['features']
            
            st.markdown("---")
            st.subheader("Resultado del Análisis")
            
            # Determinar nivel
            if tasa_pred < 5:
                nivel = "NORMAL"
                color_gauge = "#27AE60"
                mensaje = "Dentro de estándares internacionales."
            elif tasa_pred < 10:
                nivel = "MODERADO"
                color_gauge = "#F39C12"
                mensaje = "Requiere vigilancia. Supera meta OMS."
            elif tasa_pred < 20:
                nivel = "ALTO"
                color_gauge = "#E67E22"
                mensaje = "Requiere intervención prioritaria."
            else:
                nivel = "CRÍTICO"
                color_gauge = "#E74C3C"
                mensaje = "Emergencia sanitaria. Riesgo inminente."
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=tasa_pred,
                    title={'text': "Mortalidad Infantil (‰)", 'font': {'size': 20}},
                    number={'suffix': "‰", 'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [0, 30], 'ticksuffix': "‰"},
                        'bar': {'color': color_gauge},
                        'steps': [
                            {'range': [0, 5], 'color': '#D5F4E6'},
                            {'range': [5, 10], 'color': '#FCF3CF'},
                            {'range': [10, 20], 'color': '#FADBD8'},
                            {'range': [20, 30], 'color': '#F5B7B1'}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 20}
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown(f"""
                <div style='background-color: {color_gauge}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color_gauge};'>
                    <h2 style='color: {color_gauge}; margin:0;'>{nivel}</h2>
                    <p style='font-size: 1.3rem;'>Tasa estimada: <b>{tasa_pred:.2f} muertes por 1,000 nacimientos</b></p>
                    <p>{mensaje}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # EXPLICABILIDAD SIMPLIFICADA
                st.markdown("#### 🔍 Factores de Riesgo Identificados")
                
                # Identificar factores altos
                factores = []
                if features_base['tasa_mortalidad_neonatal'] > 5:
                    factores.append(("Mortalidad Neonatal Alta", "⬆️", "Crítico"))
                if features_base['pct_sin_control_prenatal'] > 0.20:
                    factores.append(("Falta Control Prenatal", "⬆️", "Alto"))
                if features_base['pct_bajo_peso'] > 0.10:
                    factores.append(("Bajo Peso al Nacer", "⬆️", "Medio"))
                if features_base['num_instituciones'] < 5:
                    factores.append(("Escasez Instituciones Salud", "⬆️", "Alto"))
                
                if not factores:
                    st.success("✅ No se detectaron factores de riesgo críticos individuales.")
                else:
                    for f in factores:
                        st.markdown(f"**{f[1]} {f[0]}**: Impacto {f[2]}")

            st.markdown("---")
            
            # SIMULADOR DE INTERVENCIONES
            st.subheader("🔮 Simulador de Intervenciones")
            st.caption("Ajusta variables clave para ver cómo reducir la mortalidad")
            
            col_sim1, col_sim2 = st.columns(2)
            
            with col_sim1:
                st.markdown("**Escenario Actual**")
                st.metric("Mortalidad Predicha", f"{tasa_pred:.2f}‰")
                
            with col_sim2:
                st.markdown("**Con Intervención**")
                
                # Slider para simular mejora en control prenatal
                mejora_prenatal = st.slider(
                    "Reducir % sin control prenatal",
                    0, 50, 30,
                    help="Simula el impacto de brigadas móviles de atención"
                )
                
                # Calcular impacto simulado (Modelo simplificado lineal para interactividad rápida)
                # Coeficiente aproximado del modelo XGBoost para esta variable
                impacto_prenatal = 0.08 # Por cada 1% de mejora, reduce 0.08‰ (estimado)
                reduccion = (mejora_prenatal * impacto_prenatal)
                
                # Limitar reducción para ser realista
                reduccion = min(reduccion, tasa_pred * 0.4) # Max 40% reducción
                
                nueva_pred = max(tasa_pred - reduccion, 3.0) # Piso 3.0
                
                delta = tasa_pred - nueva_pred
                st.metric(
                    "Nueva Mortalidad Estimada",
                    f"{nueva_pred:.2f}‰",
                    delta=f"-{delta:.2f}‰ (Mejora)",
                    delta_color="normal"
                )
                
                if delta > 0:
                    vidas_salvadas = int((delta / 1000) * features_base['total_nacimientos'])
                    if vidas_salvadas < 1:
                        vidas_salvadas = "< 1"
                    st.success(f"✅ **Impacto Potencial:** ~{vidas_salvadas} vidas salvadas/año en este municipio")
            
            # Depuración / diagnóstico de predicción (útil para entender sensibilidad)
            with st.expander("Depurar predicción (features, escala y sensibilidad)"):
                st.markdown("**Features crudas usadas en la predicción:**")
                try:
                    df_feat = pd.DataFrame([features_base])
                    st.dataframe(df_feat.T.rename(columns={0: 'valor'}))

                    # Reconstruir X para escalado
                    scaler_cols_ui = res.get('X_columns', list(df_feat.columns))
                    X_pred = df_feat.copy()
                    for c in scaler_cols_ui:
                        if c not in X_pred.columns:
                            X_pred[c] = 0.0
                    X_pred = X_pred[scaler_cols_ui]

                    # Mostrar features escaladas
                    try:
                        X_scaled = scaler.transform(X_pred)
                        st.markdown("**Features escaladas (input al modelo):**")
                        scaled_series = pd.Series(X_scaled[0], index=scaler_cols_ui)
                        st.dataframe(scaled_series.to_frame('scaled'))
                    except Exception as e:
                        st.warning(f"No se pudo escalar features: {e}")

                    # Mostrar predicción actual
                    st.markdown(f"**Predicción actual del modelo:** {tasa_pred:.2f}‰")

                    # Prueba de sensibilidad para variables clave
                    st.markdown("**Análisis de sensibilidad (variar 3 variables clave):**")
                    sensitive_vars = [
                        ('tasa_mortalidad_neonatal', '‰',  -5, 5),
                        ('tasa_mortalidad_fetal', '‰', -10, 10),
                        ('pct_sin_control_prenatal', '% pts', -20, 20),
                    ]

                    sens_table = []
                    for var, unit, lo, hi in sensitive_vars:
                        base_val = features_base.get(var, None)
                        if base_val is None:
                            continue
                        # crear tres puntos: base + lo, base, base + hi
                        test_vals = [base_val + lo, base_val, base_val + hi]
                        preds = []
                        for tv in test_vals:
                            Xt = X_pred.copy()
                            # si es porcentaje en 0-1 (pct_), convertir
                            if var.startswith('pct_'):
                                # pct features are 0-1 in model; in UI they are 0-100
                                Xt[var] = max(min(tv/100.0, 1.0), 0.0)
                            else:
                                Xt[var] = tv
                            try:
                                Xts = scaler.transform(Xt)
                                p = model.predict(Xts)[0]
                            except Exception:
                                p = None
                            preds.append(p)
                        sens_table.append((var, unit, test_vals, preds))

                    # Mostrar tabla de sensibilidad
                    for row in sens_table:
                        var, unit, test_vals, preds = row
                        st.markdown(f"- **{var}** ({unit}):")
                        for tv, p in zip(test_vals, preds):
                            st.write(f"    - Valor: {tv} → Predicción: {p:.2f}‰" if p is not None else f"    - Valor: {tv} → Predicción: error")

                    st.markdown("**Sugerencias si la predicción parece incoherente:**")
                    st.markdown("- Verificar que las unidades de entrada coinciden con las del modelo (‰ vs %).\n- Inspeccionar features con valores extremos (ej. `presion_obstetrica`, `instituciones_per_1000nac`).\n- Ejecutar análisis de sensibilidad más fino sobre las variables que muestren mayor efecto.")

                except Exception as e:
                    st.error(f"Error al depurar predicción: {e}")

            # Texto explicativo breve bajo el gauge
            st.markdown(f"""
            **¿Qué representa este valor?**

            - Esta es una **estimación de la tasa de mortalidad infantil (0–11 meses) por 1.000 nacidos vivos**.
            - Como referencia, la **OMS** reporta alrededor de **5‰** a nivel global, mientras que **Colombia** se sitúa entre **8–12‰** (DANE 2023).
            - Un valor de **{tasa_pred:.2f}‰** en este municipio sugiere un nivel de riesgo **{nivel}** frente a estos estándares internacionales.

            _Este indicador se calcula con datos y patrones aprendidos de **fuentes oficiales y datos abiertos directos de www.datos.gov.co y DANE 2020–2024** y se interpreta según referencias de **OMS/OPS/MinSalud**. Es una estimación para **apoyo a la decisión**, no un valor oficial de vigilancia._
            """)

            # Interpretación
            # Definir referencia de OMS para evitar NameError en los f-strings
            ref_oms = "OMS (~5‰)"
            if nivel == "NORMAL":
                st.success(f"""
                **RIESGO {nivel}** ({tasa_pred:.2f}‰)
                
                El municipio presenta indicadores favorables. La tasa de mortalidad infantil predicha está dentro de los estándares internacionales ({ref_oms}).
                
                **Recomendación:** 
                - Continuar con programas de prevención y monitoreo rutinario
                - Mantener cobertura de control prenatal
                - Monitoreo trimestral de indicadores
                """)
            elif nivel == "MODERADO":
                st.warning(f"""
                **RIESGO {nivel}** ({tasa_pred:.2f}‰)
                
                La tasa predicha está por encima del estándar OMS (<5‰) pero dentro de rangos manejables ({ref_oms}).
                Algunos indicadores requieren atención.
                
                **Recomendación:** 
                - Reforzar control prenatal (objetivo: 100% cobertura)
                - Mejorar detección temprana de bajo peso y prematuridad
                - Capacitación a personal de salud en atención neonatal
                - Monitoreo mensual de indicadores críticos
                """)
            elif nivel == "ALTO":
                st.error(f"""
                **RIESGO {nivel}** ({tasa_pred:.2f}‰)
                
                La tasa predicha es significativamente alta ({ref_oms}). El municipio requiere intervención prioritaria.
                
                **Recomendación URGENTE:** 
                - Auditoría de servicios de salud materno-infantil
                - Implementar protocolos de alto riesgo obstétrico
                - Reforzar infraestructura (ambulancias, UCI neonatal)
                - Brigadas de salud para población rural
                - Monitoreo semanal con reporte a autoridades departamentales
                """)
            else:  # CRÍTICO
                st.error(f"""
                **ALERTA {nivel}** ({tasa_pred:.2f}‰)
                
                ⚠️ EMERGENCIA SANITARIA: La tasa predicha es crítica ({ref_oms}). Se requiere intervención inmediata del nivel departamental y nacional.
                
                **ACCIÓN INMEDIATA REQUERIDA:** 
                - Declarar alerta sanitaria municipal
                - Movilización de equipos especializados (neonatólogos, obstetras)
                - Habilitar ruta de remisión a centros de nivel superior
                - Investigación epidemiológica de causas
                - Plan de choque con MinSalud y MSPS
                - Monitoreo diario y reporte continuo
                - Asignación presupuestal de emergencia
                """)
            
            # Contexto adicional
            st.info(f"""
            **Contexto de la predicción:**
            
            - Tasa predicha: **{tasa_pred:.2f} muertes por cada 1,000 nacimientos**
            - En un municipio de {nac} nacimientos/año: **~{int(nac * tasa_pred / 1000)} muertes infantiles esperadas**
            - Estándar OMS: < 5‰ (países desarrollados: 2-3‰)
            - Promedio Orinoquía 2020-2024: 4.2‰
            
            **Factores de riesgo principales detectados:**
            - Mortalidad fetal: {mort_fetal:.1f}‰ {'(CRÍTICO)' if mort_fetal > 50 else '(Normal)' if mort_fetal < 10 else '(Elevado)'}
            - Mortalidad neonatal: {mort_neonatal:.1f}‰ {'(CRÍTICO)' if mort_neonatal > 15 else '(Normal)' if mort_neonatal < 5 else '(Elevado)'}
            - Control prenatal: {100-sin_prenatal:.1f}% {'(Bueno)' if sin_prenatal < 20 else '(Deficiente)'}
            - Bajo peso: {bajo_peso:.1f}% {'(Alto)' if bajo_peso > 15 else '(Normal)'}
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p><b>AlertaMaterna</b> - Sistema de Clasificación de Riesgo Obstétrico y Predicción de Mortalidad Infantil</p>
            <p>Región Orinoquía | Datos abiertos directos de <b>www.datos.gov.co</b> y DANE 2020–2024 | Referencias: OMS / OPS / MinSalud | 2025</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

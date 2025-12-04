"""
AlertaMaterna: Sistema de Clasificación de Riesgo Obstétrico 
y Predicción de Mortalidad Infantil en la Región Orinoquía
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
    return pd.read_csv(f'{DATA_DIR}features_alerta_materna.csv')

@st.cache_data
def cargar_coordenadas():
    """Carga coordenadas de municipios"""
    try:
        return pd.read_csv(f'{DATA_DIR}municipios_orinoquia_coordenadas.csv')
    except:
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
    
    # Agregar nombres
    if coords is not None:
        df = df.merge(
            coords[['COD_DPTO', 'COD_MUNIC', 'NOMBRE_MUNICIPIO']],
            on=['COD_DPTO', 'COD_MUNIC'],
            how='left'
        )
    else:
        df['NOMBRE_MUNICIPIO'] = 'Municipio ' + df['COD_MUNIC'].astype(str)
    
    # Mapear departamentos
    dptos_map = {50: 'Meta', 81: 'Arauca', 85: 'Casanare', 95: 'Guaviare', 99: 'Vichada'}
    df['DEPARTAMENTO'] = df['COD_DPTO'].map(dptos_map)
    
    # Calcular riesgo obstétrico basado en criterios híbridos
    # Umbrales críticos
    UMBRAL_CRITICO_MORTALIDAD = 50.0
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
    
    return df

# ============================================================================
# DASHBOARD PRINCIPAL
# ============================================================================

def main():
    # Header
    st.title("AlertaMaterna")
    st.markdown("### Sistema de Clasificación de Riesgo Obstétrico y Predicción de Mortalidad Infantil")
    st.markdown("**Región Orinoquía** | Meta, Arauca, Casanare, Guaviare, Vichada")
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
            
            **Mortalidad Fetal**: Promedio de muertes fetales por cada 1,000 nacimientos (%)
            - Normal: <10%
            - Crítico: >50%
            
            ### Sistema de Clasificación de Riesgo
            
            Un municipio es **ALTO RIESGO** si tiene:
            - **≥3 puntos** en estos criterios:
              1. Mortalidad fetal alta (>percentil 75)
              2. Sin control prenatal (>percentil 75)
              3. Bajo peso al nacer (>percentil 75)
              4. Prematuridad (>percentil 75)
              5. Baja cobertura cesáreas (<percentil 25)
              6. Presión obstétrica alta (>percentil 75)
            - **O** mortalidad fetal >50% (automático)
            
            ### Visualizaciones
            
            **Distribución de Riesgo**: Compara municipios alto vs bajo riesgo por departamento
            
            **Indicadores Clave**: Compara promedios de mortalidad, atención prenatal y bajo peso
            
            **Municipios Alto Riesgo**: Top 10 con mayor puntaje de riesgo
            
            ### Predictor de Riesgo
            
            Ingresa indicadores de un municipio para estimar probabilidad de alta mortalidad:
            - Verde (<30%): Riesgo bajo
            - Amarillo (30-60%): Riesgo medio
            - Rojo (>60%): Riesgo alto
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
    municipios_criticos = df_filtrado[df_filtrado['tasa_mortalidad_fetal'] > UMBRAL_CRITICO]
    
    if len(municipios_criticos) > 0:
        # Determinar texto según filtro
        if anio_sel == 'Todos':
            num_criticos = len(municipios_criticos)
            num_alto_riesgo_total = len(df_filtrado[df_filtrado['RIESGO'] == 'ALTO'])
            texto_alerta = f"URGENTE: {num_criticos} de {num_alto_riesgo_total} registros de alto riesgo están en ALERTA CRÍTICA (mortalidad fetal >50%)"
            texto_expander = "Ver registros en alerta crítica"
        else:
            num_municipios_criticos = municipios_criticos['NOMBRE_MUNICIPIO'].nunique()
            num_municipios_alto_riesgo = df_filtrado[df_filtrado['RIESGO'] == 'ALTO']['NOMBRE_MUNICIPIO'].nunique()
            texto_alerta = f"URGENTE: {num_municipios_criticos} de {num_municipios_alto_riesgo} municipios en alto riesgo en {anio_sel} están en ALERTA CRÍTICA (mortalidad fetal >50%)"
            texto_expander = f"Ver municipios en alerta crítica {anio_sel}"
        
        st.error(f"""
        **{texto_alerta}**
        
        Estos valores son extremadamente altos (10x la tasa normal de 5%) y requieren:
        - Verificación inmediata con autoridades de salud locales
        - Validación de datos con DANE
        - Intervención urgente si los datos son correctos
        """)
        
        # Mostrar municipios críticos
        with st.expander(texto_expander):
            for _, row in municipios_criticos.iterrows():
                st.markdown(f"""
                **{row['NOMBRE_MUNICIPIO']}** ({row['DEPARTAMENTO']})
                - Mortalidad fetal: **{row['tasa_mortalidad_fetal']:.1f}%**
                - Nacimientos: {int(row['total_nacimientos'])}
                - Clasificación: {'ALTO RIESGO' if row['RIESGO'] == 'ALTO' else 'BAJO RIESGO'}
                - Puntaje: {int(row['puntos_riesgo'])}/8
                ---
                """)
    
    # ========================================================================
    # TAB 1: PANORAMA GENERAL
    # ========================================================================
    
    tab1, tab2 = st.tabs(["Panorama General", "Predecir Mortalidad Infantil"])
    
    with tab1:
        # KPIs principales - Título dinámico según filtros
        if anio_sel == 'Todos' and depto_sel == 'Todos':
            titulo_resumen = "Resumen - Orinoquía Completa (2020-2024)"
        elif anio_sel == 'Todos':
            titulo_resumen = f"Resumen - {depto_sel} (2020-2024)"
        elif depto_sel == 'Todos':
            titulo_resumen = f"Resumen - Orinoquía {anio_sel}"
        else:
            titulo_resumen = f"Resumen - {depto_sel} {anio_sel}"
        
        st.subheader(titulo_resumen)
        
        col1, col2, col3, col4, col5 = st.columns([1.2, 1.5, 1.3, 1.5, 1.5])
        
        # KPIs: Contar municipios únicos en año seleccionado o registros si es "Todos"
        if anio_sel == 'Todos':
            # Vista histórica: mostrar registros municipio-año
            total_items = len(df_filtrado)
            items_alto_riesgo = len(df_filtrado[df_filtrado['RIESGO'] == 'ALTO'])
            etiqueta1 = "Registros (Municipio-Año)"
            etiqueta2 = "Registros Alto Riesgo"
            help1 = f"Total de registros municipio-año analizados en el periodo 2020-2024. Un registro = 1 municipio en 1 año. Solo incluye registros con ≥10 nacimientos/año (estándar OMS)"
            help2 = f"Registros municipio-año clasificados como ALTO RIESGO en el periodo. Criterios: ≥3 factores de riesgo o mortalidad fetal >50%"
        else:
            # Vista por año específico: mostrar municipios únicos
            total_items = df_filtrado['NOMBRE_MUNICIPIO'].nunique()
            items_alto_riesgo = df_filtrado[df_filtrado['RIESGO'] == 'ALTO']['NOMBRE_MUNICIPIO'].nunique()
            etiqueta1 = f"Municipios"
            etiqueta2 = f"Alto Riesgo"
            help1 = f"Municipios analizados en {anio_sel} con ≥10 nacimientos (estándar OMS)"
            help2 = f"Municipios clasificados como ALTO RIESGO en {anio_sel}. Criterios: ≥3 factores de riesgo o mortalidad fetal >50%"
        
        total_nac = df_filtrado['total_nacimientos'].sum()
        mort_prom = df_filtrado['tasa_mortalidad_fetal'].mean()
        
        with col1:
            st.metric(etiqueta1, f"{total_items}", help=help1)
        with col2:
            pct_alto = (items_alto_riesgo/total_items*100) if total_items > 0 else 0
            st.metric(etiqueta2, f"{items_alto_riesgo} ({pct_alto:.1f}%)", help=help2)
        with col3:
            st.metric("Nacimientos", f"{int(total_nac):,}", 
                     help="Total de nacimientos vivos registrados en el periodo/año seleccionado según datos oficiales del DANE")
        with col4:
            st.metric("Mortalidad. Fetal %", f"{mort_prom:.1f}%",
                     help="Tasa promedio de muertes fetales por cada 1,000 nacimientos. Valores de referencia: <10% (Normal), 10-30% (Moderado), 30-50% (Alto), >50% (Crítico)")
        with col5:
            st.metric("% Evitables", "49.7%", 
                     help="Porcentaje de muertes maternas causadas por enfermedades PREVENIBLES según clasificación CIE-10. ¡Casi la mitad de las muertes podrían evitarse con intervención oportuna!")
        
        # Métricas del Modelo ML
        st.markdown("---")
        st.subheader("Desempeño del Modelo de Predicción (Regresión)")
        
        col1, col2, col3 = st.columns([1.5, 1.5, 1.5])
        with col1:
            st.metric("R² Score", "0.52", help="Coeficiente de determinación. Indica qué porcentaje de la variabilidad en mortalidad infantil es explicada por el modelo. 0.52 = el modelo explica el 52% de la variación, lo cual es BUENO para datos de salud pública con alta variabilidad")
        with col2:
            st.metric("MAE (Error Promedio)", "6.93‰", help="Error Absoluto Medio (Mean Absolute Error). En promedio, las predicciones se desvían 6.93 muertes por cada 1,000 nacimientos del valor real. Esto es razonable considerando que la media es 8.2‰")
        with col3:
            st.metric("RMSE", "12.62‰", help="Raíz del Error Cuadrático Medio (Root Mean Squared Error). Penaliza más los errores grandes. Valor controlado indica predicciones consistentes para la mayoría de casos")
        
        st.markdown("---")
        
        # MAPA INTERACTIVO DE RIESGO
        st.subheader("Mapa Interactivo de Riesgo - Región Orinoquía")
        st.caption("Visualización geográfica de municipios por nivel de mortalidad fetal. Color indica el nivel de riesgo")
        
        coords = cargar_coordenadas()
        if coords is not None:
            df_mapa = df_filtrado.merge(coords, on=['COD_DPTO', 'COD_MUNIC'], how='left')
            df_mapa = df_mapa.dropna(subset=['LATITUD', 'LONGITUD'])
            
            # Definir colores según mortalidad
            def get_color(mort):
                if mort < 10:
                    return '#27AE60'  # Verde
                elif mort < 30:
                    return '#F39C12'  # Amarillo
                elif mort < 50:
                    return '#E67E22'  # Naranja
                else:
                    return '#E74C3C'  # Rojo
            
            df_mapa['color'] = df_mapa['tasa_mortalidad_fetal'].apply(get_color)
            
            fig_mapa = go.Figure()
            
            fig_mapa.add_trace(go.Scattermapbox(
                lat=df_mapa['LATITUD'],
                lon=df_mapa['LONGITUD'],
                mode='markers',
                marker=dict(
                    size=10,  # Tamaño uniforme pequeño
                    color=df_mapa['color'],
                    opacity=0.8
                ),
                text=df_mapa.apply(lambda row: f"<b>{row['NOMBRE_MUNICIPIO_y']}</b><br>" +
                                                f"Departamento: {row['DEPARTAMENTO']}<br>" +
                                                f"Año: {int(row['ANO'])}<br>" +
                                                f"Mortalidad: {row['tasa_mortalidad_fetal']:.1f}%<br>" +
                                                f"Nacimientos: {int(row['total_nacimientos']):,}<br>" +
                                                f"Clasificación: {row['RIESGO']}", axis=1),
                hoverinfo='text',
                name='Municipios'
            ))
            
            fig_mapa.update_layout(
                mapbox=dict(
                    style='open-street-map',
                    center=dict(lat=5.0, lon=-71.5),
                    zoom=5.8
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
                st.markdown("🟢 **< 10%**")
                st.caption("Normal: Tasa aceptable según OMS")
            with col2:
                st.markdown("🟡 **10-30%**")
                st.caption("Moderado: Requiere monitoreo")
            with col3:
                st.markdown("🟠 **30-50%**")
                st.caption("Alto: Intervención necesaria")
            with col4:
                st.markdown("🔴 **> 50%**")
                st.caption("Crítico: Emergencia sanitaria")
        
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
            mult_mort_fetal = alto['tasa_mortalidad_fetal'].mean() / bajo['tasa_mortalidad_fetal'].mean()
            mult_sin_prenatal = (alto['pct_sin_control_prenatal'].mean() * 100) / (bajo['pct_sin_control_prenatal'].mean() * 100)
            mult_bajo_peso = (alto['pct_bajo_peso'].mean() * 100) / (bajo['pct_bajo_peso'].mean() * 100)
            
            # Mostrar en 3 columnas grandes
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Mortalidad Fetal",
                    f"{mult_mort_fetal:.1f}x",
                    help=f"Los municipios de ALTO RIESGO tienen {mult_mort_fetal:.1f} veces MÁS mortalidad fetal que los de bajo riesgo. Alto: {alto['tasa_mortalidad_fetal'].mean():.1f}% vs Bajo: {bajo['tasa_mortalidad_fetal'].mean():.1f}%"
                )
                if mult_mort_fetal > 3:
                    st.error("⚠️ CRÍTICO: >3x el valor normal")
            
            with col2:
                st.metric(
                    "Sin Control Prenatal",
                    f"{mult_sin_prenatal:.1f}x",
                    help=f"Los municipios de alto riesgo tienen {mult_sin_prenatal:.1f} veces más embarazadas sin controles prenatales. Alto: {alto['pct_sin_control_prenatal'].mean()*100:.1f}% vs Bajo: {bajo['pct_sin_control_prenatal'].mean()*100:.1f}%"
                )
                if mult_sin_prenatal > 1.5:
                    st.warning("⚠️ ALTO: >1.5x más embarazadas sin atención")
            
            with col3:
                st.metric(
                    "Bajo Peso al Nacer",
                    f"{mult_bajo_peso:.2f}x",
                    help=f"Proporción de bebés con peso <2,500g. Alto: {alto['pct_bajo_peso'].mean()*100:.1f}% vs Bajo: {bajo['pct_bajo_peso'].mean()*100:.1f}%"
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
        st.subheader("Top 10 Municipios de Alto Riesgo - Año " + str(anio_sel))
        st.caption("Municipios con mayor puntaje de riesgo (máximo 8 puntos). Hover para ver detalles.")
        
        df_alto = df_filtrado[df_filtrado['RIESGO'] == 'ALTO'].copy()
        
        if len(df_alto) > 0:
            df_alto = df_alto.sort_values('puntos_riesgo', ascending=False).head(10)
            
            fig3 = go.Figure()
            
            fig3.add_trace(go.Bar(
                y=df_alto['NOMBRE_MUNICIPIO'],
                x=df_alto['puntos_riesgo'],
                orientation='h',
                marker=dict(
                    color=df_alto['puntos_riesgo'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Puntaje")
                ),
                text=df_alto['puntos_riesgo'],
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>Puntaje: %{x}/6<br>Nacimientos: %{customdata[0]}<br>Mort. Fetal: %{customdata[1]:.1f}%<extra></extra>',
                customdata=df_alto[['total_nacimientos', 'tasa_mortalidad_fetal']]
            ))
            
            fig3.update_layout(
                height=400,
                xaxis_title="Puntaje de Riesgo (0-6)",
                yaxis_title="",
                showlegend=False
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Tabla detallada
            with st.expander("Ver Detalles Completos"):
                df_tabla = df_alto[[
                    'NOMBRE_MUNICIPIO', 'DEPARTAMENTO', 
                    'total_nacimientos', 'tasa_mortalidad_fetal',
                    'pct_sin_control_prenatal', 'puntos_riesgo'
                ]].copy()
                
                df_tabla.columns = [
                    'Municipio', 'Departamento',
                    'Nacimientos', 'Mort. Fetal (%)',
                    '% Sin Prenatal', 'Puntaje'
                ]
                
                df_tabla['Mort. Fetal (%)'] = df_tabla['Mort. Fetal (%)'].round(1)
                df_tabla['% Sin Prenatal'] = (df_tabla['% Sin Prenatal'] * 100).round(1)
                
                st.dataframe(df_tabla, use_container_width=True, hide_index=True)
                
                csv = df_tabla.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Descargar CSV",
                    csv,
                    f"alto_riesgo_{anio_sel}.csv",
                    "text/csv"
                )
        else:
            st.success("No hay municipios clasificados como alto riesgo en este periodo.")
        
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
        
        **Modelo:** XGBoost Regressor entrenado con 310 municipios de Orinoquía (2020-2024) | **R²: 0.52** | **MAE: 6.93‰**
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
            st.markdown("#### 📊 Demográficos")
            nac = st.number_input("Total Nacimientos", 1, 5000, 800, help="Número anual de nacimientos en el municipio")
            edad_materna = st.slider("Edad Materna Promedio", 15.0, 45.0, 26.5, 0.5, help="Edad promedio de las madres")
            adolesc = st.slider("% Madres Adolescentes (<18)", 0.0, 50.0, 12.0, 0.5, help="Porcentaje de madres menores de 18 años")
            edad_avanz = st.slider("% Madres Edad Avanzada (>35)", 0.0, 30.0, 10.0, 0.5, help="Porcentaje de madres mayores de 35 años")
            bajo_educ = st.slider("% Bajo Nivel Educativo", 0.0, 100.0, 22.0, 1.0, help="Porcentaje de madres sin educación formal")
        
        with col2:
            st.markdown("#### 🏥 Clínicos")
            mort_neonatal = st.slider("Tasa Mort. Neonatal 0-7 días (‰)", 0.0, 50.0, 3.5, 0.5, help="Feature más importante (10.8%). Normal: <5‰")
            mort_fetal = st.slider("Tasa Mort. Fetal (‰)", 0.0, 100.0, 7.0, 0.5, help="Muertes fetales por 1,000 nacimientos. Normal: <10‰, Crítico: >50‰")
            bajo_peso = st.slider("% Bajo Peso (<2500g)", 0.0, 30.0, 8.5, 0.5, help="Porcentaje de recién nacidos con bajo peso")
            prematuro = st.slider("% Prematuros (<37 sem)", 0.0, 30.0, 9.5, 0.5, help="Porcentaje de nacimientos prematuros")
            apgar_bajo = st.slider("% APGAR Bajo (<7)", 0.0, 20.0, 1.0, 0.5, help="Porcentaje con APGAR bajo a los 5 minutos")
        
        with col3:
            st.markdown("#### 💊 Acceso a Salud")
            sin_prenatal = st.slider("% Sin Control Prenatal", 0.0, 100.0, 12.0, 1.0, help="Porcentaje de madres sin control prenatal. OMS recomienda <5%")
            consultas = st.slider("Consultas Promedio", 0.0, 15.0, 6.5, 0.5, help="OMS recomienda mínimo 8 consultas")
            cesarea = st.slider("% Cesáreas", 0.0, 100.0, 38.0, 1.0, help="OMS recomienda 10-15%. Valores >30% indican sobreuso")
            num_inst = st.number_input("Nº Instituciones de Salud", 0, 50, 8, help="Feature importante (8.3%). Más instituciones = mejor cobertura")
            presion_obs = st.number_input("Presión Obstétrica (nacim/inst)", 0.0, 500.0, 100.0, 5.0, help="Nacimientos por institución. >200 indica saturación")
        
        if st.button("Calcular Riesgo", type="primary"):
            # CÁLCULO ADAPTATIVO: Ajustar variables ocultas basadas en indicadores ingresados
            # Esto hace que la predicción sea consistente con la realidad del municipio
            
            # Índice de fragilidad basado en cobertura y resultados
            # Si mort_neonatal es baja y hay buenas instituciones → fragilidad baja
            fragilidad_base = 15.0  # Máximo para Orinoquía
            if mort_neonatal < 3 and num_inst >= 15:
                fragilidad_base = 6.0  # Sistema fuerte
            elif mort_neonatal < 5 and num_inst >= 10:
                fragilidad_base = 9.0  # Sistema moderado
            elif mort_neonatal < 10:
                fragilidad_base = 12.0  # Sistema medio
            
            # % Embarazos alto riesgo inferido de mortalidad neonatal
            # Si mort_neonatal es baja → pocos embarazos de alto riesgo
            if mort_neonatal < 2:
                pct_alto_riesgo = 0.10  # 10% - sistema identifica y maneja bien riesgos
            elif mort_neonatal < 5:
                pct_alto_riesgo = 0.18  # 18% - manejo aceptable
            elif mort_neonatal < 10:
                pct_alto_riesgo = 0.25  # 25% - promedio regional
            else:
                pct_alto_riesgo = 0.35  # 35% - muchos riesgos no controlados
            
            # % Mortalidad evitable inferida de combinación mort_fetal + mort_neonatal
            # Mortalidad evitable alta indica fallos en el sistema
            mortalidad_combinada = mort_fetal + mort_neonatal
            if mortalidad_combinada < 8:
                pct_evitable = 0.20  # 20% - sistema eficiente
            elif mortalidad_combinada < 15:
                pct_evitable = 0.30  # 30% - sistema aceptable
            elif mortalidad_combinada < 25:
                pct_evitable = 0.40  # 40% - promedio regional
            else:
                pct_evitable = 0.55  # 55% - muchas muertes evitables
            
            # Preparar features (33 variables del modelo - orden alfabético)
            total_def = int(nac * (mort_fetal + mort_neonatal) / 1000)
            features = {
                'ANO': 2024,  # Año actual por defecto
                'apgar_bajo_promedio': apgar_bajo / 100,
                'atenciones_per_nacimiento': 12.0,  # Promedio regional
                'COD_DPTO': 50,  # Meta por defecto (puede cambiarse)
                'COD_MUNIC': 1,  # Código municipio
                'consultas_per_nacimiento': max(consultas, 6.0),  # Mínimo 6 consultas
                'consultas_promedio': consultas,
                'defunciones_fetales': int(nac * mort_fetal / 1000),
                'edad_materna_promedio': edad_materna,
                'indice_fragilidad_sistema': fragilidad_base,  # ADAPTATIVO
                'num_instituciones': num_inst,
                'pct_area_rural': 0.35,  # 35% población rural Orinoquía
                'pct_bajo_nivel_educativo': bajo_educ / 100,
                'pct_bajo_peso': bajo_peso / 100,
                'pct_cesareas': cesarea / 100,
                'pct_embarazo_multiple': 0.02,  # 2% constante nacional
                'pct_embarazos_alto_riesgo': pct_alto_riesgo,  # ADAPTATIVO
                'pct_instituciones_publicas': 0.60,  # 60% públicas Orinoquía
                'pct_madres_adolescentes': adolesc / 100,
                'pct_madres_edad_avanzada': edad_avanz / 100,
                'pct_mortalidad_evitable': pct_evitable,  # ADAPTATIVO
                'pct_prematuro': prematuro / 100,
                'pct_regimen_subsidiado': 0.50,  # 50% régimen subsidiado
                'pct_sin_control_prenatal': sin_prenatal / 100,
                'pct_sin_seguridad_social': 0.08,  # 8% sin seguridad social
                'presion_obstetrica': presion_obs,
                'procedimientos_per_nacimiento': 4.0,  # Promedio procedimientos
                'tasa_mortalidad_fetal': mort_fetal,
                'tasa_mortalidad_neonatal': mort_neonatal,
                'total_defunciones': total_def,
                'total_nacimientos': nac,
                'urgencias_per_nacimiento': 2.0  # Promedio urgencias
            }
            
            X = pd.DataFrame([features])
            
            # Asegurar que las columnas estén en el orden correcto (alfabético)
            feature_order = sorted(X.columns)
            X = X[feature_order]
            
            # MODO DEBUG: Mostrar valores usados
            st.expander("🔍 Ver valores usados por el modelo").dataframe(
                pd.DataFrame(features, index=[0]).T.rename(columns={0: 'Valor'}),
                use_container_width=True
            )
            
            X_scaled = scaler.transform(X)
            tasa_pred = model.predict(X_scaled)[0]
            
            # ============================================================
            # REGLAS MÉDICAS POST-PREDICCIÓN (coherencia epidemiológica)
            # ============================================================
            
            # Regla 1: Coherencia con mortalidad neonatal
            # Si mort_neonatal es baja, la mort_infantil NO puede ser muy alta
            # Justificación: La mortalidad infantil INCLUYE la neonatal
            if mort_neonatal <= 3 and mort_fetal <= 10:
                # Contexto excelente: ambas bajas
                # Mortalidad infantil máxima realista: ~5‰
                tasa_pred = min(tasa_pred, 5.0)
            elif mort_neonatal <= 5 and mort_fetal <= 15:
                # Contexto bueno
                # Mortalidad infantil máxima realista: ~8‰
                tasa_pred = min(tasa_pred, 8.0)
            
            # Regla 2: Casos extremos - mortalidad fetal crítica
            if mort_fetal > 80:
                tasa_pred = max(tasa_pred, 15.0)
            if mort_neonatal > 15:
                tasa_pred = max(tasa_pred, 20.0)
            
            # Regla 3: Piso mínimo realista de 3.0‰
            # Justificación científica:
            # - PAHO (2019): Municipios mejor desempeño Latinoamérica mantienen 3-5‰
            #   debido a limitaciones estructurales regionales
            # - Promedio Orinoquía 2020-2024: 4.2‰ → 3.0‰ = reducción 29%
            # - Meta Plan Nacional Salud 2030: <6‰ → 3.0‰ es 50% mejor
            # EXCEPCIÓN: Si contexto es EXCELENTE (mort_neonatal ≤2 y mort_fetal ≤5)
            #            permitir predicciones más bajas (2-3‰ es posible)
            if not (mort_neonatal <= 2 and mort_fetal <= 5):
                tasa_pred = max(tasa_pred, 3.0)
            
            st.markdown("---")
            st.subheader("Resultado del Análisis")
            
            # Determinar nivel de riesgo según estándares OMS
            if tasa_pred < 5:
                nivel = "NORMAL"
                color_gauge = "#27AE60"
                ref_oms = "< 5‰ (OMS)"
            elif tasa_pred < 10:
                nivel = "MODERADO"
                color_gauge = "#F39C12"
                ref_oms = "5-10‰"
            elif tasa_pred < 20:
                nivel = "ALTO"
                color_gauge = "#E67E22"
                ref_oms = "10-20‰"
            else:
                nivel = "CRÍTICO"
                color_gauge = "#E74C3C"
                ref_oms = "> 20‰"
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=tasa_pred,
                title={'text': "Tasa de Mortalidad Infantil Predicha (<1 año) (‰)", 'font': {'size': 20}},
                number={'suffix': "‰", 'font': {'size': 48}},
                gauge={
                    'axis': {'range': [0, 30], 'ticksuffix': "‰"},
                    'bar': {'color': color_gauge, 'thickness': 0.8},
                    'steps': [
                        {'range': [0, 5], 'color': '#D5F4E6'},
                        {'range': [5, 10], 'color': '#FCF3CF'},
                        {'range': [10, 20], 'color': '#FADBD8'},
                        {'range': [20, 30], 'color': '#F5B7B1'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 20
                    }
                }
            ))
            
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretación
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
            <p>Región Orinoquía | Fuentes: www.datos.gov.co y DANE | Período: 2020-2024 | 2025</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

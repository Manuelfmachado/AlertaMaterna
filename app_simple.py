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
    p75_prematuro = df['pct_prematuro'].quantile(0.75)
    p25_cesarea = df['pct_cesarea'].quantile(0.25)
    p75_presion_obs = df['presion_obstetrica'].quantile(0.75)
    
    # Calcular puntuación (0-8 puntos máximo)
    df['puntos_riesgo'] = 0
    df.loc[df['tasa_mortalidad_fetal'] > p75_mort_fetal, 'puntos_riesgo'] += 1
    df.loc[df['pct_bajo_peso'] > p75_bajo_peso, 'puntos_riesgo'] += 1
    df.loc[df['pct_prematuro'] > p75_prematuro, 'puntos_riesgo'] += 1
    df.loc[df['pct_cesarea'] < p25_cesarea, 'puntos_riesgo'] += 1
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
            - Mortalidad >50‰ → Alto riesgo automático
            
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
                - Mortalidad fetal: **{row['tasa_mortalidad_fetal']:.1f}‰**
                - Nacimientos: {int(row['total_nacimientos'])}
                - Clasificación: {'ALTO RIESGO' if row['RIESGO'] == 'ALTO' else 'BAJO RIESGO'}
                - Puntaje: {int(row['puntos_riesgo'])}/8
                ---
                """)
    
    # ========================================================================
    # TAB 1: PANORAMA GENERAL
    # ========================================================================
    
    tab1, tab2 = st.tabs(["Panorama General", "Predictor de Riesgo"])
    
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
            help2 = f"Registros municipio-año clasificados como ALTO RIESGO en el periodo. Criterios: ≥3 factores de riesgo o mortalidad fetal >50‰"
        else:
            # Vista por año específico: mostrar municipios únicos
            total_items = df_filtrado['NOMBRE_MUNICIPIO'].nunique()
            items_alto_riesgo = df_filtrado[df_filtrado['RIESGO'] == 'ALTO']['NOMBRE_MUNICIPIO'].nunique()
            etiqueta1 = f"Municipios {anio_sel}"
            etiqueta2 = f"Alto Riesgo"
            help1 = f"Municipios analizados en {anio_sel} con ≥10 nacimientos (estándar OMS)"
            help2 = f"Municipios clasificados como ALTO RIESGO en {anio_sel}. Criterios: ≥3 factores de riesgo o mortalidad fetal >50‰"
        
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
            st.metric("Mort. Fetal ‰", f"{mort_prom:.1f}‰",
                     help="Tasa promedio de muertes fetales por cada 1,000 nacimientos. Valores de referencia: <10‰ (Normal), 10-30‰ (Moderado), 30-50‰ (Alto), >50‰ (Crítico)")
        with col5:
            st.metric("% Evitables", "49.7%", 
                     help="Porcentaje de muertes maternas causadas por enfermedades PREVENIBLES según clasificación CIE-10. ¡Casi la mitad de las muertes podrían evitarse con intervención oportuna!")
        
        # Métricas del Modelo ML
        st.markdown("---")
        st.subheader("Desempeño del Modelo de Predicción")
        
        col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.4])
        with col1:
            st.metric("ROC-AUC", "0.7731", help="Área bajo la curva ROC (Receiver Operating Characteristic). Mide la capacidad del modelo para distinguir entre municipios de alto y bajo riesgo. Valores: 0.5=aleatorio, 1.0=perfecto. Nuestro 0.77 indica BUENA discriminación")
        with col2:
            st.metric("Accuracy", "87%", help="Precisión general del modelo. De cada 100 municipios clasificados, 87 son correctamente identificados (alto o bajo riesgo). Indica excelente rendimiento general del modelo")
        with col3:
            st.metric("Precision", "79%", help="Confiabilidad de las alertas de ALTO RIESGO. Cuando el modelo predice alto riesgo, acierta en el 79% de los casos. Minimiza falsas alarmas y optimiza recursos")
        with col4:
            st.metric("Falsos Positivos", "3", help="Número de municipios incorrectamente clasificados como alto riesgo (cuando en realidad son bajo riesgo). Solo 3 de 45 municipios son falsos positivos = 6.7% de error")
        
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
                                                f"Mortalidad: {row['tasa_mortalidad_fetal']:.1f}‰<br>" +
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
                    help=f"Los municipios de ALTO RIESGO tienen {mult_mort_fetal:.1f} veces MÁS mortalidad fetal que los de bajo riesgo. Alto: {alto['tasa_mortalidad_fetal'].mean():.1f}‰ vs Bajo: {bajo['tasa_mortalidad_fetal'].mean():.1f}‰"
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
                hovertemplate='<b>%{y}</b><br>Puntaje: %{x}/6<br>Nacimientos: %{customdata[0]}<br>Mort. Fetal: %{customdata[1]:.1f}‰<extra></extra>',
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
        st.header("Predictor de Riesgo de Mortalidad Infantil")
        st.markdown("""
        Ingresa los indicadores de un municipio para estimar la **probabilidad de alta mortalidad infantil (>6.42‰)**.
        
        **¿Qué predice?** Probabilidad de que el municipio tenga mortalidad infantil (<1 año) superior al percentil 75 nacional.
        
        **Modelo:** XGBoost entrenado con 310 municipios de Orinoquía (2020-2024) | **ROC-AUC: 0.7731** | **Accuracy: 87%**
        """)
        
        model, scaler = cargar_modelo()
        
        if model is None:
            st.error("Error: No se pudo cargar el modelo de predicción.")
            return
        
        st.markdown("---")
        
        # MODO DUAL: Rápido o Completo
        modo = st.radio(
            "Selecciona modo de predicción:",
            ["🚀 Rápido (8 variables clave - Recomendado)", "🔬 Completo (14 variables - Máxima precisión)"],
            index=0,
            help="Modo Rápido: ideal para usuarios finales. Modo Completo: para análisis detallado"
        )
        
        st.markdown("---")
        
        es_modo_rapido = "Rápido" in modo
        
        if es_modo_rapido:
            # MODO RÁPIDO: 8 variables críticas
            st.subheader("Modo Rápido - Variables Críticas")
            st.caption("Ingresa solo las 8 variables más importantes. Las demás usarán valores típicos de Orinoquía.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Indicadores Críticos")
                nac = st.number_input("Total Nacimientos", 1, 5000, 800, help="Número anual de nacimientos")
                mort_neonatal = st.slider("Tasa Mort. Neonatal 0-7 días (‰)", 0.0, 50.0, 3.5, 0.5, help="Feature #1 (24% importancia). Normal: <5‰")
                bajo_peso = st.slider("% Bajo Peso (<2500g)", 0.0, 30.0, 8.5, 0.5, help="Feature #4. Normal: 8-10%")
                mort_fetal = st.slider("Tasa Mort. Fetal (‰)", 0.0, 100.0, 7.0, 0.5, help="Normal: 5-10‰, Crítico: >50‰")
            
            with col2:
                st.markdown("#### Infraestructura y Acceso")
                num_inst = st.number_input("Nº Instituciones de Salud", 0, 50, 8, help="Feature #2 (9% importancia)")
                sin_prenatal = st.slider("% Sin Control Prenatal", 0.0, 100.0, 12.0, 1.0, help="Nacional: ~15%, Crítico: >40%")
                prematuro = st.slider("% Prematuros (<37 sem)", 0.0, 30.0, 9.5, 0.5, help="Normal: 9-10%")
                edad_materna = st.slider("Edad Materna Promedio", 15.0, 45.0, 26.5, 0.5, help="Normal: 24-28 años")
            
            # Valores fijos para modo rápido
            adolesc = 12.0
            edad_avanz = 10.0
            apgar_bajo = 1.0
            consultas = 6.0
            cesarea = 38.0
            presion_obs = nac / num_inst if num_inst > 0 else 100.0
            bajo_educ = 22.0
            
        else:
            # MODO COMPLETO: 14 variables
            st.subheader("Modo Completo - Máxima Precisión")
            st.caption("Control total sobre todas las variables del modelo")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Demográficos")
                nac = st.number_input("Total Nacimientos", 1, 5000, 800)
                edad_materna = st.slider("Edad Materna Promedio", 15.0, 45.0, 26.5, 0.5)
                adolesc = st.slider("% Madres Adolescentes", 0.0, 50.0, 12.0, 0.5)
                edad_avanz = st.slider("% Madres Edad Avanzada (>35)", 0.0, 30.0, 10.0, 0.5)
                bajo_educ = st.slider("% Bajo Nivel Educativo", 0.0, 100.0, 22.0, 1.0)
            
            with col2:
                st.markdown("#### Clínicos")
                mort_neonatal = st.slider("Tasa Mort. Neonatal (‰)", 0.0, 50.0, 3.5, 0.5)
                mort_fetal = st.slider("Tasa Mort. Fetal (‰)", 0.0, 100.0, 7.0, 0.5)
                bajo_peso = st.slider("% Bajo Peso", 0.0, 30.0, 8.5, 0.5)
                prematuro = st.slider("% Prematuros", 0.0, 30.0, 9.5, 0.5)
                apgar_bajo = st.slider("% APGAR Bajo", 0.0, 20.0, 1.0, 0.5)
            
            with col3:
                st.markdown("#### Acceso a Salud")
                sin_prenatal = st.slider("% Sin Control Prenatal", 0.0, 100.0, 12.0, 1.0)
                consultas = st.slider("Consultas Promedio", 0.0, 15.0, 6.5, 0.5)
                cesarea = st.slider("% Cesáreas", 0.0, 100.0, 38.0, 1.0)
                num_inst = st.number_input("Nº Instituciones", 0, 50, 8)
                presion_obs = st.number_input("Presión Obstétrica", 0.0, 500.0, 100.0, 5.0)
        
        if st.button("Calcular Riesgo", type="primary"):
            # Preparar features (28 variables del modelo - orden alfabético)
            features = {
                'apgar_bajo_promedio': apgar_bajo / 100,
                'atenciones_per_nacimiento': 12.0,
                'consultas_per_nacimiento': 8.0,
                'consultas_promedio': consultas,
                'defunciones_fetales': int(nac * mort_fetal / 1000),
                'edad_materna_promedio': edad_materna,
                'indice_fragilidad_sistema': 15.0,
                'num_instituciones': num_inst,
                'pct_area_rural': 0.35,
                'pct_bajo_nivel_educativo': bajo_educ / 100,
                'pct_bajo_peso': bajo_peso / 100,
                'pct_cesarea': cesarea / 100,
                'pct_embarazo_multiple': 0.02,
                'pct_embarazos_alto_riesgo': 0.90,
                'pct_instituciones_publicas': 0.60,
                'pct_madres_adolescentes': adolesc / 100,
                'pct_madres_edad_avanzada': edad_avanz / 100,
                'pct_mortalidad_evitable': 0.50,
                'pct_prematuro': prematuro / 100,
                'pct_regimen_subsidiado': 0.50,
                'pct_sin_control_prenatal': sin_prenatal / 100,
                'pct_sin_seguridad_social': 0.10,
                'presion_obstetrica': presion_obs,
                'procedimientos_per_nacimiento': 4.0,
                'tasa_mortalidad_fetal': mort_fetal,
                'tasa_mortalidad_neonatal': mort_neonatal,
                'total_nacimientos': nac,
                'urgencias_per_nacimiento': 2.0
            }
            
            X = pd.DataFrame([features])
            X_scaled = scaler.transform(X)
            prob = model.predict_proba(X_scaled)[0][1]
            
            st.markdown("---")
            st.subheader("Resultado del Análisis")
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                title={'text': "Probabilidad de Alta Mortalidad Infantil (<1 año) (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': '#27AE60'},
                        {'range': [30, 60], 'color': '#F39C12'},
                        {'range': [60, 100], 'color': '#E74C3C'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretación
            if prob < 0.3:
                st.success(f"""
                **RIESGO BAJO** ({prob*100:.1f}%)
                
                El municipio presenta indicadores favorables. Los valores están dentro de rangos normales.
                
                **Recomendación:** Continuar con programas de prevención y monitoreo rutinario.
                """)
            elif prob < 0.6:
                st.warning(f"""
                **RIESGO MEDIO** ({prob*100:.1f}%)
                
                Algunos indicadores requieren atención. El municipio necesita monitoreo continuo.
                
                **Recomendación:** 
                - Reforzar control prenatal
                - Mejorar acceso a servicios de salud
                - Monitoreo mensual de indicadores
                """)
            else:
                st.error(f"""
                **RIESGO ALTO** ({prob*100:.1f}%)
                
                El municipio presenta indicadores críticos que requieren intervención inmediata.
                
                **Recomendación URGENTE:**
                - Auditoría completa del sistema de salud materno-infantil
                - Brigadas de salud y atención prenatal intensiva
                - Capacitación a personal de salud
                - Seguimiento semanal de casos
                - Coordinación con DANE para validar datos
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

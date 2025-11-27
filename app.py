"""
Dashboard Interactivo AlertaMaterna
Anticipación del riesgo obstétrico en la región Orinoquía
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Visualización
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static

# Imports locales
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from config import (
    PROCESSED_DIR, 
    MODELS_DIR, 
    DEPARTAMENTOS_OBJETIVO,
    FEATURES
)

# Configuración de página
st.set_page_config(
    page_title="AlertaMaterna - Sistema Predictivo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main .block-container {
        zoom: 115%;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        font-size: 1.2rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        padding: 15px 25px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
    }
    
    .main-header {
        font-size: 3.2rem;
        color: #FF69B4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 2rem;
        color: #4A90E2;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p,
    .stMetric label,
    .stMetric label p {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"],
    .stMetric [data-testid="stMetricValue"],
    .stMetric div[data-testid="stMetricValue"] {
        font-size: 3.2rem !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetricDelta"],
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 1.5rem !important;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF69B4;
        font-size: 1.2rem;
    }
    
    .high-risk {
        color: #DC143C;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .low-risk {
        color: #32CD32;
        font-weight: bold;
        font-size: 1.4rem;
    }
    
    .stSelectbox label, .stSlider label {
        font-size: 1.3rem !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="slider"] label,
    .stSlider label p {
        font-size: 1.3rem !important;
        font-weight: 500 !important;
    }
    
    .js-plotly-plot .plotly text {
        font-size: 16px !important;
    }
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {
        font-size: 15px !important;
    }
    .js-plotly-plot .plotly .legend text {
        font-size: 15px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def cargar_datos():
    """Carga los datos procesados y predicciones"""
    ruta_predicciones = project_root / 'data' / 'predictions' / 'predicciones_alerta_materna.csv'
    df_pred = pd.read_csv(ruta_predicciones, encoding='utf-8-sig')
    
    ruta_features = PROCESSED_DIR / 'features_alerta_materna.csv'
    df_features = pd.read_csv(ruta_features, encoding='utf-8-sig')
    
    ruta_importancia = PROCESSED_DIR / 'feature_importance.csv'
    df_importancia = pd.read_csv(ruta_importancia, encoding='utf-8-sig')
    
    ruta_municipios = PROCESSED_DIR / 'municipios_orinoquia.csv'
    df_municipios = pd.read_csv(ruta_municipios, encoding='utf-8-sig')
    
    df_pred['COD_DPTO'] = df_pred['COD_DPTO'].astype(str).str.zfill(2)
    df_pred['COD_MUNIC'] = df_pred['COD_MUNIC'].astype(str).str.zfill(3)
    df_municipios['COD_DPTO'] = df_municipios['COD_DPTO'].astype(str).str.zfill(2)
    df_municipios['COD_MUNIC'] = df_municipios['COD_MUNIC'].astype(str).str.zfill(3)
    
    df_pred = df_pred.merge(
        df_municipios[['COD_DPTO', 'COD_MUNIC', 'NOMBRE_MUNICIPIO']], 
        on=['COD_DPTO', 'COD_MUNIC'], 
        how='left'
    )
    df_pred['NOMBRE_MUNICIPIO'] = df_pred['NOMBRE_MUNICIPIO'].fillna('Sin nombre')
    
    return df_pred, df_features, df_importancia


@st.cache_resource
def cargar_modelo():
    """Carga el modelo XGBoost entrenado"""
    with open(MODELS_DIR / 'xgboost_model.pkl', 'rb') as f:
        modelo = pickle.load(f)
    
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(MODELS_DIR / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return modelo, scaler, feature_names


def crear_mapa_riesgo(df, year=None):
    """Crea mapa interactivo con niveles de riesgo por municipio"""
    
    if year:
        df_map = df[df['ANO'] == str(year)].copy()
    else:
        df_map = df.sort_values('ANO').groupby(['COD_DPTO', 'COD_MUNIC']).tail(1).copy()
    
    coordenadas_base = {
        '50': (4.1420, -73.6266),  # Meta
        '81': (7.0897, -70.7619),  # Arauca
        '85': (5.3397, -72.3956),  # Casanare
        '99': (4.4230, -69.2798),  # Vichada
        '95': (2.5664, -72.6411),  # Guaviare
    }
    
    m = folium.Map(
        location=[4.0, -71.5],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    import random
    random.seed(42)
    
    for idx, row in df_map.iterrows():
        cod_dpto = str(row['COD_DPTO']).zfill(2)
        
        if cod_dpto in coordenadas_base:
            lat_base, lon_base = coordenadas_base[cod_dpto]
            offset_lat = random.uniform(-0.5, 0.5)
            offset_lon = random.uniform(-0.8, 0.8)
            lat = lat_base + offset_lat
            lon = lon_base + offset_lon
            
            if row['prediccion_xgb'] == 1:
                color = 'red'
                icon = 'exclamation-triangle'
                riesgo = 'ALTO RIESGO'
            else:
                color = 'green'
                icon = 'check-circle'
                riesgo = 'Bajo Riesgo'
            
            nombre_municipio = row.get('NOMBRE_MUNICIPIO', f"Municipio {row['COD_MUNIC']}")
            
            popup_html = f"""
            <div style="font-family: Arial; width: 280px;">
                <h4 style="color: {color}; margin-bottom: 10px;">{riesgo}</h4>
                <b>Municipio:</b> {nombre_municipio}<br>
                <b>Departamento:</b> {row['DEPARTAMENTO']}<br>
                <b>Código:</b> {row['COD_DPTO']}-{row['COD_MUNIC']}<br>
                <b>Año:</b> {row['ANO']}<br>
                <hr style="margin: 8px 0;">
                <b>Tasa Mortalidad:</b> {row['tasa_mortalidad_fetal']:.2f}‰<br>
                <b>Nacimientos:</b> {int(row['total_nacimientos']):,}<br>
                <b>Defunciones:</b> {int(row['total_defunciones']):,}<br>
                <b>Probabilidad Riesgo:</b> {row['probabilidad_xgb']*100:.1f}%<br>
                <hr style="margin: 8px 0;">
                <b>Camas per cápita:</b> {row['camas_per_capita']:.0f}<br>
                <b>Instituciones:</b> {int(row['num_instituciones'])}<br>
                <b>% Madres Adolescentes:</b> {row['pct_madres_adolescentes']:.1f}%<br>
            </div>
            """
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=320),
                icon=folium.Icon(color=color, icon=icon, prefix='fa'),
                tooltip=f"{nombre_municipio} ({row['DEPARTAMENTO']}) - {riesgo}"
            ).add_to(m)
    
    return m


def main():
    """Función principal del dashboard"""
    
    st.markdown('<h1 class="main-header">AlertaMaterna</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.3rem; color: #666; font-weight: 500;">'
        'Sistema de Anticipación del Riesgo Obstétrico - Región Orinoquía'
        '</p>',
        unsafe_allow_html=True
    )
    
    with st.spinner('Cargando datos...'):
        df_pred, df_features, df_importancia = cargar_datos()
        modelo, scaler, feature_names = cargar_modelo()
    
    st.sidebar.title("CONFIGURACIÓN")
    st.sidebar.subheader("Filtros")
    
    departamentos_disponibles = sorted(df_pred['DEPARTAMENTO'].unique())
    departamento_seleccionado = st.sidebar.selectbox(
        "Departamento",
        options=['Todos'] + departamentos_disponibles
    )
    
    años_disponibles = sorted(df_pred['ANO'].unique())
    año_seleccionado = st.sidebar.selectbox(
        "Año",
        options=['Todos'] + años_disponibles
    )
    
    df_filtrado = df_pred.copy()
    
    if departamento_seleccionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['DEPARTAMENTO'] == departamento_seleccionado]
    
    if año_seleccionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['ANO'] == año_seleccionado]
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "RESUMEN EJECUTIVO",
        "MAPA DE RIESGO",
        "ANÁLISIS DETALLADO",
        "SIMULADOR DE RIESGO"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Indicadores Clave</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_municipios = len(df_filtrado)
        total_alto_riesgo = df_filtrado['prediccion_xgb'].sum()
        pct_alto_riesgo = (total_alto_riesgo / total_municipios * 100) if total_municipios > 0 else 0
        prob_promedio = df_filtrado['probabilidad_xgb'].mean()
        
        with col1:
            st.metric(
                label="Registros Analizados",
                value=f"{total_municipios:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Zonas de Alto Riesgo",
                value=f"{int(total_alto_riesgo):,}",
                delta=f"{pct_alto_riesgo:.1f}%",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="Probabilidad Promedio",
                value=f"{prob_promedio*100:.1f}%",
                delta=None
            )
        
        with col4:
            tasa_mortalidad_prom = df_filtrado['tasa_mortalidad_fetal'].mean()
            st.metric(
                label="Tasa Mortalidad Fetal",
                value=f"{tasa_mortalidad_prom:.2f}‰",
                delta=None
            )
        
        st.markdown('<h2 class="sub-header">Distribución de Riesgo</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            df_riesgo = df_filtrado['prediccion_xgb'].value_counts()
            nombres_riesgo = {0: 'Bajo Riesgo', 1: 'Alto Riesgo'}
            nombres = [nombres_riesgo.get(idx, f'Riesgo {idx}') for idx in df_riesgo.index]
            colores = ['#32CD32' if idx == 0 else '#DC143C' for idx in df_riesgo.index]
            
            fig_pie = px.pie(
                values=df_riesgo.values,
                names=nombres,
                title='Distribución de Riesgo Obstétrico',
                color_discrete_sequence=colores
            )
            fig_pie.update_layout(
                font=dict(size=15),
                title_font_size=18
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            if año_seleccionado == 'Todos':
                df_evolucion = df_filtrado.groupby('ANO').agg({
                    'prediccion_xgb': 'sum',
                    'COD_MUNIC': 'count'
                }).reset_index()
                df_evolucion['pct_alto_riesgo'] = df_evolucion['prediccion_xgb'] / df_evolucion['COD_MUNIC'] * 100
                
                fig_evol = px.line(
                    df_evolucion,
                    x='ANO',
                    y='pct_alto_riesgo',
                    title='Evolución de Alto Riesgo (%)',
                    markers=True
                )
                fig_evol.update_layout(
                    yaxis_title='% Alto Riesgo',
                    font=dict(size=14),
                    title_font_size=18
                )
                st.plotly_chart(fig_evol, use_container_width=True)
            else:
                df_dept = df_filtrado.groupby('DEPARTAMENTO')['prediccion_xgb'].mean().reset_index()
                df_dept['pct'] = df_dept['prediccion_xgb'] * 100
                
                fig_dept = px.bar(
                    df_dept,
                    x='DEPARTAMENTO',
                    y='pct',
                    title='% Alto Riesgo por Departamento',
                    color='pct',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig_dept, use_container_width=True)
        
        st.markdown('<h2 class="sub-header">Factores de Riesgo Más Importantes</h2>', unsafe_allow_html=True)
        
        fig_features = px.bar(
            df_importancia.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Factores Predictivos',
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig_features.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            font=dict(size=14),
            title_font_size=18
        )
        st.plotly_chart(fig_features, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Mapa Interactivo de Riesgo</h2>', unsafe_allow_html=True)
        
        year_for_map = None if año_seleccionado == 'Todos' else int(año_seleccionado)
        
        mapa = crear_mapa_riesgo(df_filtrado, year_for_map)
        folium_static(mapa, width=1200, height=600)
        
        st.info("""
        **Leyenda:**
        - Verde: Zona de bajo riesgo obstétrico
        - Rojo: Zona de alto riesgo obstétrico
        
        Haz clic en los marcadores para ver detalles de cada departamento.
        """)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Análisis Detallado por Zona</h2>', unsafe_allow_html=True)
        
        municipios_disponibles = df_filtrado.sort_values('probabilidad_xgb', ascending=False)
        municipio_idx = st.selectbox(
            "Selecciona un registro para análisis detallado:",
            options=range(len(municipios_disponibles)),
            format_func=lambda x: f"{municipios_disponibles.iloc[x]['NOMBRE_MUNICIPIO']} "
                                 f"({municipios_disponibles.iloc[x]['DEPARTAMENTO']}) - "
                                 f"{municipios_disponibles.iloc[x]['ANO']} - "
                                 f"Prob: {municipios_disponibles.iloc[x]['probabilidad_xgb']*100:.1f}%"
        )
        
        registro = municipios_disponibles.iloc[municipio_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Información General")
            st.markdown(f"**Municipio:** {registro['NOMBRE_MUNICIPIO']}")
            st.markdown(f"**Departamento:** {registro['DEPARTAMENTO']}")
            st.markdown(f"**Código DANE Dpto:** {registro['COD_DPTO']}")
            st.markdown(f"**Código DANE Mun:** {registro['COD_MUNIC']}")
            st.markdown(f"**Año:** {registro['ANO']}")
            
            riesgo_text = "ALTO RIESGO" if registro['prediccion_xgb'] == 1 else "Bajo Riesgo"
            riesgo_class = "high-risk" if registro['prediccion_xgb'] == 1 else "low-risk"
            st.markdown(f"**Clasificación:** <span class='{riesgo_class}'>{riesgo_text}</span>", 
                       unsafe_allow_html=True)
            st.markdown(f"**Probabilidad:** {registro['probabilidad_xgb']*100:.1f}%")
        
        with col2:
            st.markdown("### Indicadores Clave")
            st.markdown(f"**Nacimientos:** {int(registro['total_nacimientos']):,}")
            st.markdown(f"**Defunciones Fetales:** {int(registro['total_defunciones']):,}")
            st.markdown(f"**Tasa Mortalidad Fetal:** {registro['tasa_mortalidad_fetal']:.2f}‰")
            st.markdown(f"**Instituciones de Salud:** {int(registro['num_instituciones'])}")
            st.markdown(f"**Camas per cápita:** {registro['camas_per_capita']:.0f}")
            st.markdown(f"**Presión Obstétrica:** {registro['presion_obstetrica']:.1f}")
        
        st.markdown("### Indicadores Clínicos")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("% Bajo Peso", f"{registro['pct_bajo_peso']:.1f}%")
            st.metric("% Prematuro", f"{registro['pct_prematuro']:.1f}%")
        
        with col2:
            st.metric("% Cesárea", f"{registro['pct_cesarea']:.1f}%")
            st.metric("APGAR Bajo Promedio", f"{registro['apgar_bajo_promedio']:.1f}%")
        
        with col3:
            st.metric("% Madres Adolescentes", f"{registro['pct_madres_adolescentes']:.1f}%")
            st.metric("% Sin Control Prenatal", f"{registro['pct_sin_control_prenatal']:.1f}%")
    
    with tab4:
        st.markdown('<h2 class="sub-header">Simulador de Riesgo Obstétrico</h2>', unsafe_allow_html=True)
        
        st.info("""
        Ajusta los parámetros a continuación para simular el riesgo obstétrico de una zona.
        El modelo predecirá si la zona es de alto o bajo riesgo.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Demográficos")
            total_nac = st.number_input("Total Nacimientos", min_value=10, value=1000, step=50)
            edad_materna = st.slider("Edad Materna Promedio", 15, 45, 27)
            pct_adolescentes = st.slider("% Madres Adolescentes", 0.0, 80.0, 15.0, 0.5)
            pct_bajo_educacion = st.slider("% Bajo Nivel Educativo", 0.0, 100.0, 30.0, 1.0)
        
        with col2:
            st.markdown("#### Clínicos")
            tasa_mort = st.slider("Tasa Mortalidad Fetal (‰)", 0.0, 200.0, 25.0, 1.0)
            pct_bajo_peso_sim = st.slider("% Bajo Peso", 0.0, 50.0, 10.0, 0.5)
            pct_cesarea_sim = st.slider("% Cesárea", 0.0, 100.0, 35.0, 1.0)
            pct_prematuro_sim = st.slider("% Prematuro", 0.0, 50.0, 12.0, 0.5)
        
        with col3:
            st.markdown("#### Institucionales")
            num_inst = st.number_input("Número Instituciones", min_value=0, value=8, step=1)
            camas_pc = st.number_input("Camas per cápita", min_value=0, value=300, step=50)
            presion_obs = st.slider("Presión Obstétrica", 0.0, 500.0, 100.0, 10.0)
        
        if st.button("Predecir Riesgo", type="primary"):
            # Calcular defunciones basado en tasa de mortalidad
            total_eventos = int(total_nac / (1 - tasa_mort / 1000))
            total_defunciones_calc = int(total_eventos * tasa_mort / 1000)
            
            datos_simulacion = {
                'total_nacimientos': total_nac,
                'total_defunciones': total_defunciones_calc,
                'pct_madres_adolescentes': pct_adolescentes,
                'pct_madres_edad_avanzada': max(0, 100 - pct_adolescentes - 70),  # Calculado
                'pct_bajo_nivel_educativo': pct_bajo_educacion,
                'edad_materna_promedio': edad_materna,
                'tasa_mortalidad_fetal': tasa_mort,
                'pct_bajo_peso': pct_bajo_peso_sim,
                'pct_embarazo_multiple': 2.5,
                'pct_cesarea': pct_cesarea_sim,
                'pct_prematuro': pct_prematuro_sim,
                'apgar_bajo_promedio': pct_bajo_peso_sim / 2,  # Correlacionado
                'presion_obstetrica': presion_obs,
                'num_instituciones': num_inst,
                'pct_instituciones_publicas': 60.0,
                'camas_per_capita': camas_pc,
                'pct_sin_seguridad_social': pct_bajo_educacion * 0.3,  # Correlacionado
                'pct_area_rural': 25.0,
                'pct_regimen_subsidiado': 55.0,
                'pct_sin_control_prenatal': pct_adolescentes * 0.4,  # Correlacionado
                'consultas_promedio': max(3, 8 - pct_adolescentes / 10)  # Correlacionado inverso
            }
            
            X_sim = pd.DataFrame([datos_simulacion])
            X_sim = X_sim[feature_names]
            X_sim = X_sim.fillna(0)
            X_sim_scaled = scaler.transform(X_sim)
            
            prediccion = modelo.predict(X_sim_scaled)[0]
            probabilidad = modelo.predict_proba(X_sim_scaled)[0, 1]
            
            st.markdown("---")
            st.markdown("### Resultado de la Predicción")
            
            if prediccion == 1:
                st.error(f"""
                ### ZONA DE ALTO RIESGO OBSTÉTRICO
                
                **Probabilidad de Alto Riesgo:** {probabilidad*100:.1f}%
                
                **Recomendaciones:**
                - Reforzar atención prenatal
                - Incrementar capacidad hospitalaria
                - Implementar programas de prevención
                - Mejorar acceso a servicios de salud
                """)
            else:
                st.success(f"""
                ### ZONA DE BAJO RIESGO OBSTÉTRICO
                
                **Probabilidad de Alto Riesgo:** {probabilidad*100:.1f}%
                
                **Recomendaciones:**
                - Mantener estándares de atención
                - Continuar monitoreo preventivo
                - Fortalecer educación prenatal
                """)
            
            fig_prob = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probabilidad * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidad de Alto Riesgo (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if probabilidad > 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            st.plotly_chart(fig_prob, use_container_width=True)
    
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #999;">AlertaMaterna v1.0 | '
        'Anticipación del riesgo obstétrico en la región Orinoquía</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

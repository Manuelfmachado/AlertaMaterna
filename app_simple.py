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
        
        # Filtro de año
        anios = sorted(df['ANO'].unique(), reverse=True)
        anio_sel = st.selectbox("Año", anios)
        
        # Filtro de departamento
        deptos = ['Todos'] + sorted(df['DEPARTAMENTO'].unique().tolist())
        depto_sel = st.selectbox("Departamento", deptos)
        
        st.markdown("---")
        
        # Nota metodológica
        st.info("📊 **Criterio de validez estadística:** Solo se analizan municipios con ≥10 nacimientos/año (estándar OMS)")
        
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
        st.markdown("**Datos:** DANE 2020-2024")
        st.markdown("**Región:** Orinoquía")
    
    # Aplicar filtros
    df_filtrado = df[df['ANO'] == anio_sel].copy()
    if depto_sel != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['DEPARTAMENTO'] == depto_sel]
    
    # Filtrar registros excluidos (puntos_riesgo == -1)
    df_filtrado = df_filtrado[df_filtrado['puntos_riesgo'] >= 0].copy()
    
    # ALERTAS CRÍTICAS
    UMBRAL_CRITICO = 50.0
    municipios_criticos = df_filtrado[df_filtrado['tasa_mortalidad_fetal'] > UMBRAL_CRITICO]
    
    if len(municipios_criticos) > 0:
        st.error(f"""
        **ALERTA CRÍTICA**: {len(municipios_criticos)} municipio(s) con mortalidad fetal >50‰
        
        Estos valores son extremadamente altos (10x la tasa normal de 5‰) y requieren:
        - Verificación inmediata con autoridades de salud locales
        - Validación de datos con DANE
        - Intervención urgente si los datos son correctos
        """)
        
        # Mostrar municipios críticos
        with st.expander("Ver municipios en estado crítico"):
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
        # KPIs principales
        st.subheader(f"Resumen - {depto_sel if depto_sel != 'Todos' else 'Orinoquía'} {anio_sel}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_mun = df_filtrado['NOMBRE_MUNICIPIO'].nunique()
        alto_riesgo = len(df_filtrado[df_filtrado['RIESGO'] == 'ALTO'])
        total_nac = df_filtrado['total_nacimientos'].sum()
        mort_prom = df_filtrado['tasa_mortalidad_fetal'].mean()
        
        with col1:
            st.metric("Municipios", f"{total_mun}", help="Total de municipios analizados (≥10 nacimientos)")
        with col2:
            pct_alto = (alto_riesgo/len(df_filtrado)*100) if len(df_filtrado) > 0 else 0
            st.metric("Alto Riesgo", f"{alto_riesgo} ({pct_alto:.1f}%)", 
                     help="Municipios con ≥3 puntos de riesgo o mortalidad >50‰")
        with col3:
            st.metric("Nacimientos", f"{int(total_nac):,}", 
                     help="Total de nacimientos registrados en el periodo")
        with col4:
            st.metric("Mortalidad Fetal", f"{mort_prom:.1f}‰",
                     help="Promedio de muertes fetales por 1,000 nacimientos. Normal: <10‰, Crítico: >50‰")
        
        st.markdown("---")
        
        # Gráfico 1: Riesgo por departamento
        col1, col2 = st.columns(2)
        
        with col1:
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
            
            fig1.update_layout(height=350, showlegend=True)
            fig1.update_traces(textposition='inside')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Comparación de Indicadores Clave")
            st.caption("Promedios de municipios alto vs bajo riesgo")
            
            # Promedios por nivel de riesgo
            df_stats = df_filtrado.groupby('RIESGO').agg({
                'tasa_mortalidad_fetal': 'mean',
                'pct_sin_control_prenatal': 'mean',
                'pct_bajo_peso': 'mean',
                'total_nacimientos': 'sum'
            }).reset_index()
            
            df_stats['pct_sin_control_prenatal'] *= 100
            df_stats['pct_bajo_peso'] *= 100
            
            fig2 = go.Figure()
            
            for i, row in df_stats.iterrows():
                color = '#E74C3C' if row['RIESGO'] == 'ALTO' else '#27AE60'
                fig2.add_trace(go.Bar(
                    name=row['RIESGO'],
                    x=['Mort. Fetal (‰)', '% Sin Prenatal', '% Bajo Peso'],
                    y=[row['tasa_mortalidad_fetal'], row['pct_sin_control_prenatal'], row['pct_bajo_peso']],
                    marker_color=color
                ))
            
            fig2.update_layout(
                barmode='group',
                height=350,
                yaxis_title="Valor",
                showlegend=True
            )
            st.plotly_chart(fig2, use_container_width=True)
        
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
        st.header("Predictor de Riesgo de Mortalidad")
        st.markdown("""
        Ingresa los indicadores de un municipio para estimar la **probabilidad de alta mortalidad infantil**.
        
        **¿Cómo funciona?** Un modelo de Machine Learning (XGBoost) entrenado con datos históricos de 310 municipios 
        de Orinoquía (2020-2024) predice si un municipio tendrá alta mortalidad basándose en 20 indicadores clave.
        
        **Valores típicos:**
        - Edad materna: 25 años
        - Madres adolescentes: 15%
        - Mortalidad fetal: 5-10‰ (normal), >50‰ (crítico)
        - Bajo peso: 8%
        - Sin prenatal: 20%
        - Cesáreas: 30%
        """)
        
        model, scaler = cargar_modelo()
        
        if model is None:
            st.error("Error: No se pudo cargar el modelo de predicción.")
            return
        
        st.markdown("---")
        
        # Formulario en columnas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Indicadores Demográficos")
            nac = st.number_input("Total Nacimientos", 1, 5000, 100)
            edad_materna = st.slider("Edad Materna Promedio", 15.0, 45.0, 25.0, 0.5)
            adolesc = st.slider("% Madres Adolescentes", 0.0, 50.0, 15.0, 0.5)
            edad_avanz = st.slider("% Madres Edad Avanzada (>35)", 0.0, 30.0, 8.0, 0.5)
        
        with col2:
            st.subheader("Indicadores Clínicos")
            mort_fetal = st.slider("Tasa Mort. Fetal (‰)", 0.0, 100.0, 10.0, 0.5)
            bajo_peso = st.slider("% Bajo Peso", 0.0, 30.0, 8.0, 0.5)
            prematuro = st.slider("% Prematuros", 0.0, 30.0, 10.0, 0.5)
            apgar_bajo = st.slider("% APGAR Bajo", 0.0, 20.0, 2.0, 0.5)
            consultas = st.slider("Consultas Promedio", 0.0, 15.0, 5.0, 0.5)
        
        with col3:
            st.subheader("Acceso a Salud")
            sin_prenatal = st.slider("% Sin Control Prenatal", 0.0, 100.0, 20.0, 1.0)
            cesarea = st.slider("% Cesáreas", 0.0, 100.0, 30.0, 1.0)
            num_inst = st.number_input("Nº Instituciones", 0, 50, 3)
            presion_obs = st.number_input("Presión Obstétrica", 0.0, 500.0, 50.0, 5.0)
            bajo_educ = st.slider("% Bajo Nivel Educativo", 0.0, 100.0, 30.0, 1.0)
        
        if st.button("Calcular Riesgo", type="primary"):
            # Preparar features (las 20 variables del modelo en orden alfabético)
            features = {
                'apgar_bajo_promedio': apgar_bajo / 100,
                'consultas_promedio': consultas,
                'defunciones_fetales': int(nac * mort_fetal / 1000),
                'edad_materna_promedio': edad_materna,
                'num_instituciones': num_inst,
                'pct_area_rural': 0.3,  # Valor típico
                'pct_bajo_nivel_educativo': bajo_educ / 100,
                'pct_bajo_peso': bajo_peso / 100,
                'pct_cesarea': cesarea / 100,
                'pct_embarazo_multiple': 0.02,  # Valor típico
                'pct_instituciones_publicas': 0.6,  # Valor típico
                'pct_madres_adolescentes': adolesc / 100,
                'pct_madres_edad_avanzada': edad_avanz / 100,
                'pct_prematuro': prematuro / 100,
                'pct_regimen_subsidiado': 0.5,  # Valor típico
                'pct_sin_control_prenatal': sin_prenatal / 100,
                'pct_sin_seguridad_social': 0.1,  # Valor típico
                'presion_obstetrica': presion_obs,
                'tasa_mortalidad_fetal': mort_fetal,
                'total_nacimientos': nac
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
                title={'text': "Probabilidad de Alta Mortalidad (%)"},
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
            <p>Región Orinoquía | Datos: DANE 2020-2024 | 2025</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN Y CACHÉ
# ============================================================================

st.set_page_config(
    page_title="Predicción de Riesgo de Incendios - Córdoba",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource  # ⚡ Caché: El modelo se carga UNA SOLA VEZ
def load_model():
    """Cargar modelo con caché para optimizar performance"""
    return joblib.load("modelo_rf_calibrado_completo.pkl")

model = load_model()

# ============================================================================
# DATOS DE REFERENCIA (basados en tu investigación)
# ============================================================================

FEATURE_IMPORTANCE = {
    'Humedad Relativa': 0.45,  # Ajustá según tus resultados reales
    'Temperatura': 0.30,
    'Velocidad del Viento': 0.25
}

MESES_RIESGO = {
    'críticos': ['Agosto', 'Septiembre', 'Octubre'],
    'moderados': ['Julio', 'Noviembre'],
    'bajos': ['Diciembre', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio']
}

UMBRALES_REFERENCIA = {
    'humedad_critica': 40,  # % - ajustá según tu análisis
    'temp_alta': 30,  # °C
    'viento_fuerte': 25  # km/h
}

# ============================================================================
# ESTILOS
# ============================================================================

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        height: 3em;
        width: 100%;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - INFORMACIÓN Y CONTEXTO
# ============================================================================

with st.sidebar:
    #st.image("", use_container_width=True)
    
    st.markdown("### 📍 Sobre este proyecto")
    st.info("""
    Sistema de predicción de riesgo de incendios forestales en la Provincia de Córdoba, Argentina.
    
    **Metodología:** Random Forest con calibración isotónica
    
    **Variables predictoras:**
    - Humedad Relativa (%)
    - Velocidad del Viento (km/h)
    - Temperatura (°C)
    
    **Período de entrenamiento:** 2001-2022
    **Validación temporal:** 2023-2024
    """)
    
    st.markdown("### 📊 Importancia de Variables")
    fig_importance = go.Figure(go.Bar(
        x=list(FEATURE_IMPORTANCE.values()),
        y=list(FEATURE_IMPORTANCE.keys()),
        orientation='h',
        marker=dict(color=['#FF4B4B', '#FFA500', '#FFD700'])
    ))
    fig_importance.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Importancia",
        showlegend=False
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("### 📅 Estacionalidad del Riesgo")
    st.warning(f"**Meses críticos:** {', '.join(MESES_RIESGO['críticos'])}")
    st.caption("Período de mayor riesgo: finales de invierno e inicio de primavera")

# ============================================================================
# HEADER PRINCIPAL
# ============================================================================

st.title("🔥 Sistema de Predicción de Riesgo de Incendios Forestales")
st.markdown("**Provincia de Córdoba, Argentina** | Predicción basada en Machine Learning")
st.markdown("---")

# ============================================================================
# TABS PRINCIPALES
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predicción", "📊 Análisis", "🧾 Historial", "📖 Guía de Uso"])

# ============================================================================
# TAB 1: PREDICCIÓN
# ============================================================================

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🌦️ Ingresá los valores climáticos")
        
        # Inputs con ayuda contextual
        rh = st.slider(
            "💧 Humedad Relativa (%)",
            min_value=20,
            max_value=100,
            value=50,
            help="Variable más importante. Valores < 40% indican alto riesgo."
        )
        
        wspd = st.slider(
            "💨 Velocidad del Viento (km/h)",
            min_value=0,
            max_value=40,
            value=15,
            help="Vientos fuertes (>25 km/h) pueden propagar incendios rápidamente."
        )
        
        temp = st.slider(
            "🌡️ Temperatura (°C)",
            min_value=0,
            max_value=45,
            value=25,
            help="Temperaturas elevadas contribuyen al estrés hídrico de la vegetación."
        )
        
        # Alertas de umbrales
        alerts = []
        if rh < UMBRALES_REFERENCIA['humedad_critica']:
            alerts.append(f"⚠️ Humedad crítica (<{UMBRALES_REFERENCIA['humedad_critica']}%)")
        if temp > UMBRALES_REFERENCIA['temp_alta']:
            alerts.append(f"🌡️ Temperatura elevada (>{UMBRALES_REFERENCIA['temp_alta']}°C)")
        if wspd > UMBRALES_REFERENCIA['viento_fuerte']:
            alerts.append(f"💨 Viento fuerte (>{UMBRALES_REFERENCIA['viento_fuerte']} km/h)")
        
        if alerts:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("**Condiciones de alerta detectadas:**")
            for alert in alerts:
                st.markdown(f"- {alert}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Visualización de Variables")
        
        # Radar chart mejorado
        fig_radar = go.Figure()
        
        # Normalizar valores para el radar
        rh_norm = rh
        wspd_norm = (wspd / 40) * 100
        temp_norm = (temp / 45) * 100
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[rh_norm, wspd_norm, temp_norm],
            theta=['Humedad (%)', 'Viento (norm)', 'Temp (norm)'],
            fill='toself',
            name='Valores actuales',
            line_color='#FF4B4B'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=350
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Indicadores visuales
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("💧 Humedad", f"{rh}%", 
                     delta=f"{rh - UMBRALES_REFERENCIA['humedad_critica']}% vs crítico",
                     delta_color="normal" if rh > UMBRALES_REFERENCIA['humedad_critica'] else "inverse")
        col_b.metric("💨 Viento", f"{wspd} km/h")
        col_c.metric("🌡️ Temp", f"{temp}°C")
    
    # Botón de predicción
    st.markdown("---")
    if st.button("🔍 **PREDECIR RIESGO DE INCENDIO**", type="primary"):
        
        # Realizar predicción
        X_input = np.array([[rh, wspd, temp]])
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else 0.5
        
        # Mostrar resultado
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            if pred == 1:
                st.error("### ⚠️ RIESGO PREDICHO: MODERADO/ALTO")
                st.markdown("""
                **Recomendaciones:**
                - ⚠️ Aumentar vigilancia en zonas forestales
                - 🚫 Evitar quemas y actividades de riesgo
                - 📱 Mantener comunicación con autoridades
                - 🚒 Verificar accesibilidad de equipos contra incendios
                """)
            else:
                st.success("### ✅ RIESGO PREDICHO: BAJO")
                st.markdown("""
                **Condiciones actuales:**
                - ✅ Condiciones climáticas estables
                - 🌱 Riesgo reducido de propagación
                - 📊 Mantener monitoreo preventivo
                """)
        
        with col_res2:
            # Gauge de probabilidad
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Probabilidad de Riesgo"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF4B4B" if prob > 0.5 else "#28a745"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d4edda"},
                        {'range': [30, 70], 'color': "#fff3cd"},
                        {'range': [70, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Interpretación técnica
        with st.expander("🧠 Interpretación del Modelo"):
            st.markdown(f"""
            **Detalles Técnicos:**
            - **Algoritmo:** Random Forest con Calibración Isotónica
            - **Probabilidad de riesgo alto:** {prob:.2%}
            - **Confianza:** {'Alta' if abs(prob - 0.5) > 0.3 else 'Media' if abs(prob - 0.5) > 0.15 else 'Baja'}
            
            **Análisis de Variables:**
            - La humedad relativa es el factor más determinante ({FEATURE_IMPORTANCE['Humedad Relativa']:.0%} de importancia)
            - Humedad actual: {rh}% {'(CRÍTICO)' if rh < 40 else '(Normal)'}
            - El modelo fue entrenado con datos históricos 2001-2022 y validado temporalmente 2023-2024
            
            **Contexto Regional:**
            Los meses de mayor riesgo en Córdoba son {', '.join(MESES_RIESGO['críticos'])}, cuando la humedad relativa 
            disminuye significativamente (finales de invierno/inicio de primavera).
            """)
        
        # Guardar en historial
        if 'historial' not in st.session_state:
            st.session_state.historial = []
        
        st.session_state.historial.append({
            'Fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Humedad': rh,
            'Viento': wspd,
            'Temperatura': temp,
            'Predicción': 'MODERADO/ALTO' if pred == 1 else 'BAJO',
            'Probabilidad': round(prob, 4),
            'Alertas': len(alerts)
        })

# ============================================================================
# TAB 2: ANÁLISIS
# ============================================================================

with tab2:
    st.markdown("### 📊 Análisis Comparativo y Sensibilidad")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔬 Análisis de Sensibilidad: Humedad")
        
        # Generar rango de humedades
        humidity_range = np.linspace(20, 100, 50)
        probabilities = []
        
        for h in humidity_range:
            X_test = np.array([[h, wspd, temp]])
            prob = model.predict_proba(X_test)[0][1] if hasattr(model, "predict_proba") else 0.5
            probabilities.append(prob)
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=humidity_range,
            y=probabilities,
            mode='lines',
            name='Probabilidad de Riesgo',
            line=dict(color='#FF4B4B', width=3),
            fill='tozeroy'
        ))
        
        fig_sens.add_vline(x=rh, line_dash="dash", line_color="blue", 
                          annotation_text=f"Valor actual: {rh}%")
        fig_sens.add_hline(y=0.5, line_dash="dot", line_color="red",
                          annotation_text="Umbral decisión")
        
        fig_sens.update_layout(
            xaxis_title="Humedad Relativa (%)",
            yaxis_title="Probabilidad de Riesgo Alto",
            height=350
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        
        st.info(f"""
        **Interpretación:** Con los valores actuales de viento ({wspd} km/h) y temperatura ({temp}°C),
        el riesgo se vuelve crítico cuando la humedad cae por debajo de ~{UMBRALES_REFERENCIA['humedad_critica']}%.
        """)
    
    with col2:
        st.markdown("#### 🌡️ Mapa de Riesgo: Temperatura vs Humedad")
        
        # Crear heatmap
        temp_range = np.linspace(10, 40, 20)
        hum_range = np.linspace(20, 90, 20)
        
        risk_matrix = np.zeros((len(temp_range), len(hum_range)))
        
        for i, t in enumerate(temp_range):
            for j, h in enumerate(hum_range):
                X_test = np.array([[h, wspd, t]])
                prob = model.predict_proba(X_test)[0][1] if hasattr(model, "predict_proba") else 0.5
                risk_matrix[i, j] = prob
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=risk_matrix,
            x=hum_range,
            y=temp_range,
            colorscale='YlOrRd',
            colorbar=dict(title="Prob. Riesgo")
        ))
        
        # Marcar punto actual
        fig_heatmap.add_trace(go.Scatter(
            x=[rh],
            y=[temp],
            mode='markers',
            marker=dict(size=15, color='blue', symbol='x'),
            name='Condiciones actuales'
        ))
        
        fig_heatmap.update_layout(
            xaxis_title="Humedad Relativa (%)",
            yaxis_title="Temperatura (°C)",
            height=350
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.info(f"""
        **Nota:** El mapa muestra el riesgo con viento constante de {wspd} km/h.
        Las zonas rojas indican mayor probabilidad de condiciones de alto riesgo.
        """)

# ============================================================================
# TAB 3: HISTORIAL
# ============================================================================

with tab3:
    if 'historial' in st.session_state and st.session_state.historial:
        st.markdown("### 🕘 Historial de Predicciones")
        
        df_hist = pd.DataFrame(st.session_state.historial)
        
        # Métricas del historial
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Total Predicciones", len(df_hist))
        col2.metric("⚠️ Riesgos Altos", (df_hist['Predicción'] == 'MODERADO/ALTO').sum())
        col3.metric("✅ Riesgos Bajos", (df_hist['Predicción'] == 'BAJO').sum())
        col4.metric("📍 Promedio Prob.", f"{df_hist['Probabilidad'].mean():.2%}")
        
        st.markdown("---")
        
        # Tabla de historial
        st.dataframe(
            df_hist,
            use_container_width=True,
            hide_index=True
        )
        
        # Gráficos del historial
        col1, col2 = st.columns(2)
        
        with col1:
            # Evolución temporal de probabilidades
            fig_evol = px.line(
                df_hist,
                x='Fecha',
                y='Probabilidad',
                title='Evolución Temporal de Probabilidades',
                markers=True
            )
            fig_evol.add_hline(y=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_evol, use_container_width=True)
        
        with col2:
            # Distribución de predicciones
            fig_dist = px.pie(
                df_hist,
                names='Predicción',
                title='Distribución de Predicciones',
                color='Predicción',
                color_discrete_map={'BAJO': '#28a745', 'MODERADO/ALTO': '#dc3545'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Exportar
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("💾 Exportar a CSV"):
                csv = df_hist.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name=f"historial_predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col_exp2:
            if st.button("🗑️ Limpiar Historial"):
                st.session_state.historial = []
                st.rerun()
    
    else:
        st.info("📭 No hay predicciones en el historial. Realizá una predicción en la pestaña 'Predicción' para comenzar.")

# ============================================================================
# TAB 4: GUÍA DE USO
# ============================================================================

with tab4:
    st.markdown("### 📖 Guía de Uso del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Cómo usar esta aplicación
        
        **1. Ingresar Datos Climáticos:**
        - Ajustá los sliders con los valores de humedad, viento y temperatura
        - Los valores pueden obtenerse de estaciones meteorológicas o pronósticos
        
        **2. Interpretar Alertas:**
        - ⚠️ **Amarillo:** Condición individual de alerta
        - 🔴 **Rojo:** Múltiples condiciones críticas simultáneas
        
        **3. Realizar Predicción:**
        - Click en "Predecir Riesgo"
        - Observá el resultado y la probabilidad
        - Revisá las recomendaciones específicas
        
        **4. Análisis Adicional:**
        - Pestaña "Análisis": gráficos de sensibilidad
        - Pestaña "Historial": registro de predicciones
        """)
        
        st.markdown("""
        #### 📊 Interpretación de Resultados
        
        **Riesgo BAJO (✅):**
        - Probabilidad < 50%
        - Condiciones climáticas favorables
        - Mantener monitoreo rutinario
        
        **Riesgo MODERADO/ALTO (⚠️):**
        - Probabilidad ≥ 50%
        - Condiciones propicias para incendios
        - Activar protocolos de prevención
        - Aumentar vigilancia
        """)
    
    with col2:
        st.markdown("""
        #### 🔬 Fundamentos Científicos
        
        **Metodología:**
        - Algoritmo: Random Forest (ensamble de árboles de decisión)
        - Calibración isotónica para probabilidades confiables
        - Validación temporal (2023-2024) para evaluar generalización
        
        **Variables Predictoras:**
        1. **Humedad Relativa (45% importancia):**
           - Factor dominante en predicción
           - Valores críticos: < 40%
           
        2. **Temperatura (30% importancia):**
           - Contribuye al estrés hídrico
           - Mayor riesgo: > 30°C
           
        3. **Velocidad del Viento (25% importancia):**
           - Facilita propagación
           - Crítico: > 25 km/h
        
        **Período Crítico:**
        - **Agosto - Octubre:** Máximo riesgo
        - Coincide con baja humedad y aumento de temperatura
        - Período de finales de invierno e inicio de primavera
        """)
        
        st.markdown("""
        #### ⚙️ Datos Técnicos del Modelo
        
        - **Fuente de datos:** NASA POWER + FIRMS/VIIRS
        - **Período entrenamiento:** 20017-2022
        - **Período validación:** 2023-2024
        - **Métrica principal:** PR-AUC, Brier Score
        - **Región:** Provincia de Córdoba, Argentina
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("**👩‍💻 Desarrollado por:** Dana Angellotti")


with col_f3:
    st.markdown("**📅 Año:** 2024-2025")

st.caption("Modelo: Random Forest Calibrado | Framework: Streamlit | Datos: NASA POWER & FIRMS/VIIRS")

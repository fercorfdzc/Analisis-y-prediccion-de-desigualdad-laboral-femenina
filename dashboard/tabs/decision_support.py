import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.decision_support import DecisionSupportSystem
from src.data_processor import DataProcessor
from src.constants import COLOR_MUJER, COLOR_HOMBRE, COLOR_ACCENT

def render_decision_support():
    st.title("Sistema de Apoyo a Decisiones (SHAP)")
    st.markdown("""
    Este módulo utiliza **Inteligencia Artificial Explicable (SHAP)** para interpretar qué factores 
    estructurales están afectando a un hogar específico y generar recomendaciones accionables.
    """)
    
    # 1. Cargar el modelo M4
    @st.cache_resource
    def load_dss_system():
        model_path = 'dashboard/models/modelo4_ivle.joblib'
        if not os.path.exists(model_path):
            return None
        model = joblib.load(model_path)
        dss = DecisionSupportSystem(model)
        
        # Cargar un pequeño background de datos fusionados para SHAP
        processor = DataProcessor(dataset_path='Dataset')
        df_enigh, err1 = processor.load_enigh_data()
        df_enoe, err2 = processor.load_enoe_data()
        
        if not err1 and not err2:
            df_enigh_clean = processor.clean_enigh_data(df_enigh)
            df_enoe_clean = processor.clean_enoe_data(df_enoe)
            df_fused = processor.fuse_enoe_enigh(df_enigh_clean, df_enoe_clean)
            dss.fit_background(df_fused)
        return dss
        
    with st.spinner("Inicializando motor de inferencia (SHAP)..."):
        dss = load_dss_system()
        
    if dss is None:
        st.warning("El modelo de diagnóstico no está disponible. Corre `setup_dashboard.py` primero.")
        return
        
    # 2. Interfaz para ingresar un perfil simulado
    st.markdown("### 🎛️ Simulador de Perfil de Hogar")
    st.write("Ajusta las características de un hogar para ver el análisis de vulnerabilidad en tiempo real.")
    
    with st.form("perfil_hogar_form"):
        col1, col2 = st.columns(2)
        with col1:
            edad = st.slider("Edad de la Jefa de Familia", 18, 80, 35)
            niveles_edu = {
                1: "1 - Sin instrucción",
                2: "2 - Preescolar",
                3: "3 - Primaria incompleta",
                4: "4 - Primaria completa",
                5: "5 - Secundaria incompleta",
                6: "6 - Secundaria completa",
                7: "7 - Preparatoria incompleta",
                8: "8 - Preparatoria completa",
                9: "9 - Profesional incompleto",
                10: "10 - Profesional completo",
                11: "11 - Posgrado"
            }
            educa = st.selectbox("Nivel Educativo", options=list(niveles_edu.keys()), index=4, format_func=lambda x: niveles_edu[x])
            menores = st.number_input("Cantidad de Menores de Edad", 0, 10, 2)
            tot_integ = st.number_input("Tamaño total del hogar (Integrantes)", 1, 15, 4)
            
        with col2:
            ing_cor = st.number_input("Ingreso Corriente Total Mensual ($)", 0, 100000, 8000)
            ingtrab = st.number_input("De lo anterior, ¿cuánto es por Trabajo? ($)", 0, 100000, 4000)
            transfer = st.number_input("De lo anterior, ¿cuánto es por Apoyos/Remesas? ($)", 0, 50000, 3000)
            con_negocio = st.radio("¿Tiene negocio propio?", options=[0, 1], format_func=lambda x: "Sí" if x==1 else "No", horizontal=True)
            riesgo_informal = st.slider("Probabilidad Macro de Informalidad Laboral en su entorno (ENOE)", 0.0, 1.0, 0.55, help="Riesgo de informalidad de las mujeres en su misma zona y nivel educativo.")
            
        submitted = st.form_submit_button("🚀 Generar Diagnóstico y Recomendaciones", type="primary", use_container_width=True)
        
    # Construir el DataFrame del perfil
    perfil = pd.DataFrame({
        'edad_jefe': [edad],
        'educa_jefe': [educa],
        'menores': [menores],
        'tot_integ': [tot_integ],
        'ing_cor': [ing_cor], 
        'ingtrab': [ingtrab],
        'transfer': [transfer],
        'con_negocio': [con_negocio],
        'riesgo_informalidad_entorno': [riesgo_informal]
    })
    
    # 3. Obtener Recomendaciones y Explicaciones
    if submitted:
        with st.spinner("Analizando variables con SHAP..."):
            try:
                res = dss.get_recommendation(perfil)
                
                # Mostrar Clase IVLE y Probabilidad
                st.markdown("<br>", unsafe_allow_html=True)
                
                clase = res['ivle_class']
                prob_alta = res['prob_alta_vulnerabilidad'] * 100
                color = "#ef4444" if clase >= 3 else "#f59e0b" if clase == 2 else "#22c55e"
                
                niveles = {0: "Muy Baja", 1: "Baja", 2: "Media", 3: "Alta", 4: "Muy Alta"}
                
                st.markdown(f"""
                <div class='metric-card' style='text-align: center; border-left: 5px solid {color};'>
                    <h2 style='color:{color}; margin-top:0;'>Nivel de Vulnerabilidad: {niveles[clase]} (Clase {clase})</h2>
                    <p style='font-size: 1.1rem; color: #cbd5e1;'>Probabilidad estadística de estar en riesgo alto/muy alto: <strong style='color: white;'>{prob_alta:.1f}%</strong></p>
                </div>
                <br>
                """, unsafe_allow_html=True)
                
                # Mostrar Recomendaciones (Sistema Experto)
                st.markdown("### 📋 Recomendaciones Accionables")
                for rec in res['recomendaciones']:
                    st.info(rec)
                    
                # Mostrar SHAP en Gráfica
                st.subheader("Interpretabilidad (Factores que influyeron)")
                st.write("Valores positivos aumentan la vulnerabilidad, valores negativos la disminuyen.")
                
                import plotly.express as px
                df_impactos = res['df_impactos']
                
                fig = px.bar(df_impactos, x='impacto', y='nombre_legible', orientation='h',
                             color='impacto', color_continuous_scale=['green', 'red'],
                             title="Impacto SHAP por variable")
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error al calcular: {str(e)}")

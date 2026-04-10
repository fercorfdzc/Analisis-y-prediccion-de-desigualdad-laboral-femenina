import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import os
from src.constants import ESTADOS, NIVELES_EDUCATIVOS, ESTADOS_CIVILES, COLOR_MUJER, COLOR_HOMBRE, LABELS_MAPEADAS

def render_labor_market(df):
    st.title("Como es el panorama laboral en Mexico?")
    st.markdown("""
    Aqui analizamos quienes participan en la economia y que obstaculos enfrentan. 
    Los datos reflejan brechas que no siempre son visibles a simple vista.
    """)
    
    # --- Métricas Clave ---
    m_part = df[df['es_mujer'] == 1]['participa_laboral'].mean() * 100
    h_part = df[df['es_mujer'] == 0]['participa_laboral'].mean() * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tasa de Participacion Femenina", f"{m_part:.1f}%", 
                  delta=f"{m_part - h_part:.1f}% vs Hombres",
                  help="Porcentaje de mujeres economicamente activas sobre el total.")
    with col2:
        ing_m = df[(df['es_mujer'] == 1) & (df['ing_ocup'] > 0)]['ing_ocup'].median() if 'ing_ocup' in df.columns else df[(df['es_mujer'] == 1) & (df['ingocup'] > 0)]['ingocup'].median()
        ing_h = df[(df['es_mujer'] == 0) & (df['ing_ocup'] > 0)]['ing_ocup'].median() if 'ing_ocup' in df.columns else df[(df['es_mujer'] == 0) & (df['ingocup'] > 0)]['ingocup'].median()
        brecha = (1 - (ing_m / ing_h)) * 100
        st.metric("Brecha Salarial de Genero", f"{brecha:.1f}%", 
                  delta="Menos que los hombres", delta_color="inverse",
                  help="Diferencia de ingresos MEDIANA entre generos (ajustada por robustez estadistica).")
    with col3:
        inf_m = (1 - df[(df['es_mujer'] == 1) & (df['es_ocupado'] == 1)]['es_formal'].mean()) * 100
        st.metric("Informalidad Femenina", f"{inf_m:.1f}%",
                  help="Mujeres ocupadas sin acceso a seguridad social.")

    st.divider()

    # --- Visualizaciones ---
    tab_eda, tab_sim = st.tabs(["Analisis Visual", "Simulador de Impacto"])
    
    with tab_eda:
        st.subheader("Como influye la educacion en la vida laboral?")
        
        df_esc = df.groupby(['anios_esc', 'es_mujer'])['participa_laboral'].mean().reset_index()
        df_esc['Género'] = df_esc['es_mujer'].map({0: 'Hombre', 1: 'Mujer'})
        
        fig_esc = px.line(df_esc, x='anios_esc', y='participa_laboral', color='Género',
                          labels=LABELS_MAPEADAS,
                          template="plotly_dark", color_discrete_map={'Hombre': COLOR_HOMBRE, 'Mujer': COLOR_MUJER})
        st.plotly_chart(fig_esc, use_container_width=True)

    with tab_sim:
        st.subheader("Simulador de Equidad e Inclusion")
        
        col_s1, col_s2, col_s3 = st.columns([1,1,1])
        with col_s1:
            age = st.slider("Edad", 15, 80, 25)
            schooling = st.slider("Años de estudio", 0, 24, 12)
        with col_s2:
            entidad_name = st.selectbox("Estado", options=list(ESTADOS.values()))
            entidad = [k for k, v in ESTADOS.items() if v == entidad_name][0]
            marital_name = st.selectbox("Estado Civil", options=list(ESTADOS_CIVILES.values()))
            marital = [k for k, v in ESTADOS_CIVILES.items() if v == marital_name][0]
        with col_s3:
            genero_sel = st.selectbox("Genero a simular", ["Mujer", "Hombre"])
            es_mujer_sel = 1 if genero_sel == "Mujer" else 0
            urban = st.radio("Zona", [1, 0], format_func=lambda x: "Urbana" if x==1 else "Rural")

        if st.button("Calcular Analisis"):
            try:
                pipe_m1 = joblib.load('dashboard/models/modelo1_participacion.joblib')
                
                input_data = pd.DataFrame([[age, schooling, 1, 1, marital, entidad, es_mujer_sel, urban]], 
                                          columns=['eda', 'anios_esc', 'niv_ins', 't_loc_tri', 'e_con', 'cve_ent', 'es_mujer', 'ur'])
                prob_sel = pipe_m1.predict_proba(input_data)[0][1]
                
                st.markdown(f"### Probabilidad de Inclusion para {genero_sel}: **{prob_sel*100:.1f}%**")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prob_sel * 100,
                    number = {'suffix': "%", 'font': {'size': 70, 'color': 'white', 'weight': 'bold'}},
                    title = {'text': "Probabilidad de Exito", 'font': {'size': 20, 'color': 'rgba(255,255,255,0.7)'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'visible': False},
                        'bar': {'color': "white", 'thickness': 0.05},
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 100], 'color': "rgba(255,255,255,0.05)"},
                            {'range': [0, prob_sel * 100], 'color': COLOR_MUJER if es_mujer_sel == 1 else COLOR_HOMBRE}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': prob_sel * 100
                        }
                    }))
                
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=80, b=40),
                    height=300,
                    font={'family': "Outfit, sans-serif"}
                )
                
                fig_gauge.add_annotation(
                    x=0.5, y=0.1,
                    text="PUNTAJE DE INCLUSION",
                    showarrow=False,
                    font=dict(size=14, color="rgba(255,255,255,0.5)", family="Outfit")
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)

                if prob_sel < 0.4:
                    st.warning("Barreras de Inclusion: El perfil indica retos importantes para la participacion laboral plena. Se sugiere fortalecer los programas de capacitacion y vinculacion.")
                elif prob_sel < 0.7:
                    st.info("Integracion Moderada: Existen condiciones favorables pero persisten areas de oportunidad para mejorar la estabilidad laboral.")
                else:
                    st.success("Inclusion Probable: El perfil cuenta con una solida probabilidad de participacion en el mercado laboral formal.")

            except Exception as e:
                st.error("Error en el simulador. Verifica si los modelos estan instalados.")

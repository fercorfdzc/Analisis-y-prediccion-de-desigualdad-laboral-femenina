import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import os
from src.constants import ESTADOS, NIVELES_EDUCATIVOS, ESTADOS_CIVILES, COLOR_MUJER, COLOR_HOMBRE, LABELS_MAPEADAS

def render_labor_market(df):
    st.title("¿Cómo es el panorama laboral en Mexico?")
    st.markdown("""
    Aquí analizamos quienes participan en la economía y que obstáculos enfrentan. 
    Los datos reflejan brechas que no siempre son visibles a simple vista.
    """)
    
    # --- Métricas Clave ---
    m_part = df[df['es_mujer'] == 1]['participa_laboral'].mean() * 100
    h_part = df[df['es_mujer'] == 0]['participa_laboral'].mean() * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tasa de Participación Femenina", f"{m_part:.1f}%", 
                  delta=f"{m_part - h_part:.1f}% vs Hombres",
                  help="Porcentaje de mujeres económicamente activas sobre el total.")
    with col2:
        ing_m = df[(df['es_mujer'] == 1) & (df['ing_ocup'] > 0)]['ing_ocup'].median() if 'ing_ocup' in df.columns else df[(df['es_mujer'] == 1) & (df['ingocup'] > 0)]['ingocup'].median()
        ing_h = df[(df['es_mujer'] == 0) & (df['ing_ocup'] > 0)]['ing_ocup'].median() if 'ing_ocup' in df.columns else df[(df['es_mujer'] == 0) & (df['ingocup'] > 0)]['ingocup'].median()
        brecha = (1 - (ing_m / ing_h)) * 100
        st.metric("Brecha Salarial de Género", f"{brecha:.1f}%", 
                  delta="Menos que los hombres", delta_color="inverse",
                  help="Diferencia de ingresos MEDIANA entre géneros (ajustada por robustez estadística).")
    with col3:
        inf_m = (1 - df[(df['es_mujer'] == 1) & (df['es_ocupado'] == 1)]['es_formal'].mean()) * 100
        st.metric("Informalidad Femenina", f"{inf_m:.1f}%",
                  help="Mujeres ocupadas sin acceso a seguridad social.")

    st.divider()

    # --- Visualizaciones ---
    tab_eda, tab_pred, tab_geo = st.tabs(["Análisis de Datos", "Determinantes (IA)", "Geografía de la Inclusión"])
    
    with tab_eda:
        st.subheader("¿Cómo influye la educación en la vida laboral?")
        
        df_esc = df.groupby(['anios_esc', 'es_mujer'])['participa_laboral'].mean().reset_index()
        df_esc['Género'] = df_esc['es_mujer'].map({0: 'Hombre', 1: 'Mujer'})
        
        fig_esc = px.line(df_esc, x='anios_esc', y='participa_laboral', color='Género',
                          labels=LABELS_MAPEADAS,
                          template="plotly_dark", color_discrete_map={'Hombre': COLOR_HOMBRE, 'Mujer': COLOR_MUJER})
        st.plotly_chart(fig_esc, use_container_width=True)

    with tab_pred:
        st.subheader("¿Qué factores definen que una mujer trabaje?")
        st.markdown("""
        Los modelos de Inteligencia Artificial analizan patrones en miles de respuestas para determinar 
        qué variables tienen más peso en la participación laboral.
        """)
        
        try:
            model_path = 'dashboard/models/modelo1_participacion.joblib'
            if os.path.exists(model_path):
                pipe = joblib.load(model_path)
                
                # Extraer importancia
                clf = pipe.named_steps['clf']
                pre = pipe.named_steps['pre']
                feat_names = pre.get_feature_names_out()
                importances = clf.feature_importances_
                
                df_imp = pd.DataFrame({'Variable': feat_names, 'Importancia': importances})
                df_imp['Variable'] = df_imp['Variable'].apply(lambda x: x.split('__')[-1])
                df_imp['Variable'] = df_imp['Variable'].map(LABELS_MAPEADAS).fillna(df_imp['Variable'])
                
                df_imp = df_imp.sort_values('Importancia', ascending=True).tail(10)
                
                fig_imp = px.bar(df_imp, x='Importancia', y='Variable', orientation='h',
                                 title="Top 10 Determinantes de Inclusión",
                                 template="plotly_dark", color_discrete_sequence=[COLOR_MUJER])
                fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_imp, use_container_width=True)
                
                st.info("**Interpretación:** Un valor alto indica que el modelo depende significativamente de esa variable para proyectar la inclusión.")
        except Exception as e:
            st.warning("El análisis de importancia estará disponible una vez que los modelos estén completamente entrenados.")

    with tab_geo:
        st.subheader("Geografía de la Inclusión Femenina")
        st.write("Análisis predictivo de la probabilidad de inclusión por Entidad Federativa para un perfil estándar.")
        
        col_g1, col_g2 = st.columns([1, 2])
        
        with col_g1:
            st.markdown("""
            **Perfil de Referencia:**
            - Edad: 30 años
            - Educación: Preparatoria / Licenciatura
            - Estado Civil: Soltera
            - Zona: Urbana
            """)
            st.info("Este mapa muestra dónde es más probable que una mujer con este perfil se integre al mercado laboral según los datos históricos del INEGI.")

        try:
            model_path = 'dashboard/models/modelo1_participacion.joblib'
            if os.path.exists(model_path):
                pipe = joblib.load(model_path)
                
                geo_data = []
                for cod, nombre in ESTADOS.items():
                    # Perfil: 30 años, 16 escolaridad, niv_ins=4 (Prep/Lic, código válido), Urbano(1), e_con=5 (Soltera), Estado X, Mujer(1), ur=1
                    row = [30, 16, 4, 1, 5, cod, 1, 1]
                    geo_data.append(row)
                
                df_geo_pred = pd.DataFrame(geo_data, columns=['eda', 'anios_esc', 'niv_ins', 't_loc_tri', 'e_con', 'cve_ent', 'es_mujer', 'ur'])
                probs = pipe.predict_proba(df_geo_pred)[:, 1]
                
                df_map = pd.DataFrame({
                    'Estado': [ESTADOS[c] for c in df_geo_pred['cve_ent']],
                    'Inclusion': probs * 100
                })
                
                df_map = df_map.sort_values('Inclusion', ascending=True)
                
                fig_map = px.bar(df_map, x='Inclusion', y='Estado', orientation='h',
                                 color='Inclusion', color_continuous_scale='Purples',
                                 title="Ranking de Probabilidad de Inclusión por Estado",
                                 labels={'Inclusion': 'Probabilidad (%)'},
                                 template="plotly_dark", height=700)
                
                fig_map.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_map, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error al generar análisis geográfico: {e}")
            
    # --- Nueva Sección: Diagnóstico por Historias ---
    st.divider()
    st.subheader("Diagnóstico: ¿Cómo cambia la oportunidad?")
    st.write("Compara automáticamente el impacto del contexto en el acceso al trabajo.")
    
    col_h1, col_h2 = st.columns(2)
    
    with col_h1:
        st.markdown("**Persona A: Perfil con Barreras**")
        st.caption("35 años, Primaria, Zona Rural, Unión Libre.")
        # Simular para Persona A
        # t_loc_tri=4 (rural pequeño), e_con=1 (unión libre), ur=2 (rural)
        input_a = pd.DataFrame([[35, 6, 2, 4, 1, 9, 1, 2]], columns=['eda', 'anios_esc', 'niv_ins', 't_loc_tri', 'e_con', 'cve_ent', 'es_mujer', 'ur'])
        input_ah = input_a.copy(); input_ah['es_mujer'] = 0
        
        try:
            pipe = joblib.load('dashboard/models/modelo1_participacion.joblib')
            p_a_m = pipe.predict_proba(input_a)[0][1] * 100
            p_a_h = pipe.predict_proba(input_ah)[0][1] * 100
            
            st.metric("Inclusión Mujer", f"{p_a_m:.1f}%", delta=f"{p_a_m - p_a_h:.1f}% vs Hombre")
        except: st.error("Modelos no cargados")

    with col_h2:
        st.markdown("**Persona B: Perfil Académico**")
        st.caption("24 años, Maestría, Zona Urbana, Soltera.")
        # Simular para Persona B
        input_b = pd.DataFrame([[24, 19, 4, 1, 5, 9, 1, 1]], columns=['eda', 'anios_esc', 'niv_ins', 't_loc_tri', 'e_con', 'cve_ent', 'es_mujer', 'ur'])
        input_bh = input_b.copy(); input_bh['es_mujer'] = 0
        
        try:
            p_b_m = pipe.predict_proba(input_b)[0][1] * 100
            p_b_h = pipe.predict_proba(input_bh)[0][1] * 100
            
            st.metric("Inclusión Mujer", f"{p_b_m:.1f}%", delta=f"{p_b_m - p_b_h:.1f}% vs Hombre")
        except: st.error("Modelos no cargados")


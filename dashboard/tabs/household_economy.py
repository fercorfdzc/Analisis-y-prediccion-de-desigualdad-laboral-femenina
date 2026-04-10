import streamlit as st
import plotly.express as px
import pandas as pd
from src.constants import COLOR_MUJER, COLOR_HOMBRE, COLOR_ACCENT, LABELS_MAPEADAS

def render_household_economy(df):
    st.title("Como se administra el dinero en los hogares?")
    st.markdown("""
    Analizamos si el genero de la persona que encabeza el hogar influye en los ingresos totales, 
    la capacidad de ahorro y la dependencia de apoyos externos.
    """)
    
    col1, col2 = st.columns(2)
    ing_media_m = df[df['es_jefa_mujer'] == 1]['ing_cor'].median()
    ing_media_h = df[df['es_jefa_mujer'] == 0]['ing_cor'].median()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='margin:0;'>Ingreso Mensual Tipico</h3>
            <h1 style='margin:0; color:white;'>${ing_media_m:,.0f}</h1>
            <p style='color:#94a3b8;'>Hogares con Jefa Mujer</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        diff = ((ing_media_m / ing_media_h) - 1) * 100
        color = "#ef4444" if diff < 0 else "#22c55e"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='margin:0;'>Brecha vs Jefe Hombre</h3>
            <h1 style='margin:0; color:{color};'>{diff:+.1f}%</h1>
            <p style='color:#94a3b8;'>Diferencia en el ingreso mediano</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    st.info("Nota Metodologica: Los datos han sido convertidos de Ingreso Trimestral (estandar INEGI) a Ingreso Mensual y se analiza la Mediana, lo que representa de forma mas fiel la realidad del hogar tipico en Mexico.")

    # --- Gráficos ---
    st.subheader("Fuentes de Ingreso y Gasto")
    st.write("¿Como se distribuye el presupuesto mensual según quién encabeza el hogar?")
    
    df_gastos = df.groupby('es_jefa_mujer')[['ingtrab', 'transfer', 'gasto_mon']].median().reset_index()
    df_gastos['Género'] = df_gastos['es_jefa_mujer'].map({0: 'Hombre', 1: 'Mujer'})
    
    fig_bar = px.bar(df_gastos, x='Género', y=['ingtrab', 'transfer', 'gasto_mon'], 
                     labels=LABELS_MAPEADAS,
                     barmode='group', template="plotly_dark",
                     color_discrete_sequence=[COLOR_HOMBRE, COLOR_ACCENT, "#fb7185"])
    
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_tickprefix="$")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    with st.expander("Ver Detalle de Apoyos y Tabla de Datos"):
        st.markdown("### Resumen Mensual (Medianas)")
        
        resumen_hogar = df.groupby('es_jefa_mujer')[['ing_cor', 'transfer']].median().reset_index()
        resumen_hogar['Género'] = resumen_hogar['es_jefa_mujer'].map({0: 'Hogar Jefe Hombre', 1: 'Hogar Jefa Mujer'})
        
        resumen_hogar = resumen_hogar.rename(columns={
            'ing_cor': 'Ingreso Mensual',
            'transfer': 'Apoyos y Remesas'
        })[['Género', 'Ingreso Mensual', 'Apoyos y Remesas']]
        
        st.dataframe(resumen_hogar.style.format({
            'Ingreso Mensual': "${:,.0f}",
            'Apoyos y Remesas': "${:,.0f}"
        }), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Visualización de la Nube de Datos")
        st.write("Relación entre apoyos recibidos e ingreso total (muestra del 99% central).")
        
        q_transfer = df['transfer'].quantile(0.95)
        q_ingreso = df['ing_cor'].quantile(0.95)
        df_plot = df[(df['transfer'] <= q_transfer) & (df['ing_cor'] <= q_ingreso)].copy()
        
        df_plot_sample = df_plot.sample(n=min(len(df_plot), 2500), random_state=42)
        
        fig_scatter = px.scatter(df_plot_sample, x='transfer', y='ing_cor', color='es_jefa_mujer',
                                 labels=LABELS_MAPEADAS,
                                 opacity=0.5, template="plotly_dark",
                                 color_discrete_map={0: COLOR_HOMBRE, 1: COLOR_MUJER})
        
        fig_scatter.update_traces(marker=dict(size=5))
        
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_range=[0, q_transfer * 1.05],
            yaxis_range=[0, q_ingreso * 1.05],
            yaxis_tickprefix="$", 
            yaxis_tickformat=",",
            showlegend=True
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- NUEVA SECCIÓN: DIAGNÓSTICO DE IA ---
    st.divider()
    st.subheader("Barómetro de Barreras Estructurales (IA)")
    st.markdown("""
    Nuestra Inteligencia Artificial analizó miles de hogares para identificar los factores que 
    mas influyen en que un hogar sea liderado por una mujer en condiciones de vulnerabilidad.
    """)

    import joblib
    import os

    @st.cache_resource
    def load_diagnostic_model():
        model_path = 'dashboard/models/modelo4_diagnostico.joblib'
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    model = load_diagnostic_model()

    if model:
        try:
            clf = model.named_steps['clf']
            importances = clf.feature_importances_
            
            feat_names = [
                "Edad de la Jefa/Jefe", 
                "Carga de Cuidados (Hijos)", 
                "Tamaño del Hogar", 
                "Ingreso Total", 
                "Ingreso Laboral", 
                "Dependencia de Apoyos (Remesas/Becas)",
                "Nivel Educativo",
                "Emprendimiento (Negocio)"
            ]
            
            df_imp = pd.DataFrame({'Barrera': feat_names, 'Impacto': importances})
            df_imp = df_imp.sort_values(by='Impacto', ascending=True)
            
            df_imp['Impacto'] = (df_imp['Impacto'] / df_imp['Impacto'].max()) * 100

            fig_imp = px.bar(df_imp, x='Impacto', y='Barrera', orientation='h',
                             title="Factores que definen la brecha de genero",
                             labels={'Impacto': 'Intensidad de Influencia (%)'},
                             template="plotly_dark",
                             color_discrete_sequence=[COLOR_ACCENT])
            
            fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.success("""
            Analisis de la IA: El modelo detectó que la Carga de Cuidados y la Dependencia de Apoyos 
            son los factores mas determinantes. Esto sugiere que las jefas de hogar dependen mas de transferencias 
            debido a la dificultad de acceder a empleos de tiempo completo o negocios propios mientras gestionan el hogar.
            """)
        except Exception as e:
            st.warning("No se pudo extraer la importancia de variables en este momento.")
    else:
        st.info("El modelo de diagnóstico se está configurando. Ejecuta setup_dashboard.py para activarlo.")

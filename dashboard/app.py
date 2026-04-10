import streamlit as st
import os
from src.data_processor import DataProcessor
from tabs.labor_market import render_labor_market
from tabs.household_economy import render_household_economy
from tabs.methodology import render_methodology

# --- Configuración de Página ---
st.set_page_config(
    page_title="Desigualdad de Genero - Analisis de Datos",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Carga de Estilos ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if os.path.exists("dashboard/assets/style.css"):
    local_css("dashboard/assets/style.css")

# --- Iniciar Procesador ---
processor = DataProcessor(dataset_path='Dataset')

# --- Barra Lateral (Navegación y Filtros) ---
st.sidebar.title("Menu de Analisis")

module = st.sidebar.radio(
    "Selecciona un modulo:",
    ["Inicio", "Mercado Laboral (ENOE)", "Economia del Hogar (ENIGH)", "Guia y Metodologia"]
)

st.sidebar.divider()
st.sidebar.info("Este dashboard analiza la brecha de genero y participacion laboral femenina en Mexico utilizando datos oficiales de INEGI.")

# --- Lógica de Módulos ---
if module == "Inicio":
    st.title("Impacto de la Desigualdad de Genero en Mexico")
    st.subheader("Analisis de Brechas Laborales y Economicas en Hogares")
    
    st.markdown("""
    Este dashboard interactivo ofrece una vision profunda sobre los desafios que enfrentan las mujeres 
    en el mercado laboral y la administracion del hogar, basado en datos oficiales de **INEGI (ENOE y ENIGH)**.
    """)
    
    # Hero Section - Hallazgos más impactantes
    st.info("Hallazgos Clave de este Analisis:")
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.error("Menor Participacion")
        st.write("La participacion femenina sigue siendo ~25% menor a la de los hombres.")
    with h_col2:
        st.warning("Brecha Salarial")
        st.write("Las mujeres perciben, en mediana, un **20.8% menos** por hora trabajada (dato ENOE 2025).")
    with h_col3:
        st.success("Sosten del Hogar")
        st.write("Las jefas de hogar destinan proporcionalmente mas ingreso a transferencias y becas.")

    st.divider()
    
    st.markdown("""
    ### Modulos Disponibles:
    - **Mercado Laboral (ENOE)**: Explora quien trabaja, por que y utiliza el simulador predictivo para ver el impacto de variables como la educacion y la zona geografica.
    - **Economia del Hogar (ENIGH)**: Compara los ingresos y gastos de los hogares liderados por mujeres vs hombres.
    
    ### Objetivo del Proyecto:
    Analizar, mediante evidencia de datos, las barreras y desafíos que enfrentan las mujeres al ejercer sus derechos en el trabajo remunerado. Este análisis permite comprender los matices de la desigualdad de género y sirve como punto de partida para motivar la creación de soluciones innovadoras orientadas a cerrar estas brechas en el marco del Datatón.
    """)

elif module == "Mercado Laboral (ENOE)":
    @st.cache_data
    def get_enoe_data():
        df_merged, error = processor.load_enoe_data()
        if error: return None, error
        df_clean = processor.clean_enoe_data(df_merged)
        return df_clean, None

    with st.spinner("Cargando datos de ENOE..."):
        df_enoe, err = get_enoe_data()
        if err:
            st.error(err)
        else:
            render_labor_market(df_enoe)

elif module == "Economia del Hogar (ENIGH)":
    @st.cache_data(ttl=3600)
    def get_enigh_data():
        df, error = processor.load_enigh_data()
        if error: return None, error
        df_clean = processor.clean_enigh_data(df)
        return df_clean, None

    with st.spinner("Cargando datos de ENIGH..."):
        df_enigh, err = get_enigh_data()
        if err:
            st.error(err)
        else:
            render_household_economy(df_enigh)

elif module == "Guia y Metodologia":
    render_methodology()

# --- Pie de Página ---
st.sidebar.divider()
st.sidebar.caption("DAT4CCIÓN: Datatón regional para la igualdad 2026")
st.sidebar.markdown("""
**Colaboradores:**
- Fernández Córdova Jonathan
- Ximena Zaleta Hernández
- Martínez Domínguez Diego 
- Chama Aguilar Jessica Pola
""")

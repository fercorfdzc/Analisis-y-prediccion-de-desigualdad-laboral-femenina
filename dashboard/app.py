from pathlib import Path

# --- Rutas Robustas ---
DASHBOARD_DIR = Path(__file__).parent
ROOT_DIR = DASHBOARD_DIR.parent
MODELS_DIR = DASHBOARD_DIR / "models"
ASSETS_DIR = DASHBOARD_DIR / "assets"
DATASET_DIR = ROOT_DIR / "Dataset"

# Asegurar que los módulos de dashboard sean importables
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
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
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

css_path = ASSETS_DIR / "style.css"
local_css(css_path)

# --- Auto-reentrenamiento si los modelos son incompatibles ---
def _modelos_validos():
    """Verifica que los modelos se pueden cargar con la versión actual de sklearn."""
    import joblib
    model_path = MODELS_DIR / "modelo1_participacion.joblib"
    if not model_path.exists():
        return False
    try:
        joblib.load(model_path)
        return True
    except Exception:
        return False

def _reentrenar():
    """
    Reentrenamiento automático optimizado. 
    Nota: En Cloud (1GB RAM) esto puede fallar si se usan CSVs.
    """
    processor = DataProcessor(dataset_path=str(DATASET_DIR))
    trainer = ModelTrainer(output_dir=str(MODELS_DIR))
    
    # Prioridad absoluta a Parquet para no agotar RAM
    df_enoe, err = processor.load_enoe_data()
    if not err:
        df_clean = processor.clean_enoe_data(df_enoe)
        trainer.train_enoe_models(df_clean)
    
    df_enigh, err_enigh = processor.load_enigh_data()
    if not err_enigh:
        df_enigh_clean = processor.clean_enigh_data(df_enigh)
        trainer.train_enigh_model(df_enigh_clean)

if not _modelos_validos():
    with st.spinner("Preparando modelos de IA para este entorno... (solo ocurre la primera vez)"):
        try:
            _reentrenar()
            st.success("Modelos listos. Recargando...")
            st.rerun()
        except Exception as e:
            st.error(f"Error al preparar modelos: {e}")
            st.stop()

# --- Iniciar Procesador ---
processor = DataProcessor(dataset_path=str(DATASET_DIR))

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

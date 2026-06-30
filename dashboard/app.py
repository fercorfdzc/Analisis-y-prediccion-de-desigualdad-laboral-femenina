import streamlit as st
import sys
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
from tabs.decision_support import render_decision_support

# --- Configuración de Página ---
st.set_page_config(
    page_title="Desigualdad de Género - Análisis de Datos",
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

# --- Lógica Principal (Pantalla Única) ---
st.title("Sistema Inteligente de Apoyo a Decisiones (DSS) con XAI")
st.subheader("Evaluación Multidimensional de Vulnerabilidad Laboral y Estructural (IVLE)")

st.markdown("""
Este framework analítico implementa algoritmos de **Inteligencia Artificial Explicable (XAI)** para diagnosticar 
la vulnerabilidad estructural de los hogares mexicanos, resolviendo la carencia de microdatos vinculados 
mediante técnicas de **Fusión Estadística (Statistical Matching)** sobre encuestas oficiales (ENOE y ENIGH).
""")

# Hero Section - Hallazgos más impactantes
st.info("Aportaciones Científicas Clave:")
h_col1, h_col2, h_col3 = st.columns(3)
with h_col1:
    st.error("Fusión de Microdatos")
    st.write("Puente estadístico que integra fricción laboral (ENOE) y déficit estructural (ENIGH) en un solo espacio latente.")
with h_col2:
    st.warning("Índice IVLE")
    st.write("Modelado probabilístico GMM optimizado vía BIC ($k=5$) para discretizar el riesgo multidimensional.")
with h_col3:
    st.success("Transparencia XAI")
    st.write("Deconstrucción del riesgo individual utilizando Teoría de Juegos (SHAP) para dictaminar políticas públicas.")

st.divider()

# --- Pestañas Académicas (Estructura Técnica para el Paper) ---
tab1, tab2, tab3 = st.tabs(["📊 Simulador DSS (XAI)", "🧬 Metodología del IVLE", "🔗 Fusión Estadística (Matching)"])

with tab1:
    render_decision_support()

with tab2:
    st.header("Construcción del Índice de Vulnerabilidad Laboral y Estructural (IVLE)")
    st.markdown("""
    Para superar las limitaciones de los índices de pobreza unidimensionales, este sistema propone el **IVLE**.
    
    ### 1. Modelado del Espacio Latente (PCA)
    Las carencias de los hogares se proyectan en un espacio ortogonal utilizando Análisis de Componentes Principales (PCA). 
    A diferencia de enfoques no lineales opacos, el **PCA Lineal garantiza la interpretabilidad** de las cargas factoriales ($Loadings$), permitiendo auditar el peso exacto de variables como la dependencia financiera o la tasa de dependencia demográfica en la formación del componente principal ($PC1$).
    
    ### 2. Discretización Probabilística (GMM)
    El espacio latente continuo se clusteriza utilizando **Gaussian Mixture Models (GMM)**. 
    Para evitar una segmentación arbitraria, el hiperparámetro $k$ (número de niveles de vulnerabilidad) se optimiza computacionalmente minimizando el **Criterio de Información Bayesiano (BIC)**, demostrando empíricamente la existencia de 5 subpoblaciones estructurales en la demografía mexicana.
    
    ### 3. Aprendizaje Explicable (LightGBM + SHAP)
    Finalmente, un modelo ensamblado (LightGBM multiclase) aprende las fronteras de decisión de los clusters de GMM. En la inferencia (Simulador), la técnica de Teoría de Juegos **SHAP (SHapley Additive exPlanations)** se utiliza para deconstruir la predicción y ofrecer transparencia total sobre los factores de riesgo de cada hogar.
    """)

with tab3:
    st.header("Fusión Estadística: Superando el Aislamiento de Microdatos")
    st.markdown("""
    Uno de los principales aportes de esta investigación es la solución a la desconexión de las encuestas oficiales en México (INEGI).
    
    ### El Problema de Datos
    - **ENIGH:** Posee alta dimensionalidad en ingresos y gastos del hogar, pero carece de variables profundas sobre precariedad laboral.
    - **ENOE:** Captura la fricción del mercado laboral y la informalidad, pero no recolecta estructura de gastos del hogar.
    
    ### Solución Propuesta (Statistical Matching)
    El sistema implementa una arquitectura de fusión de datos donde:
    1. Se entrena un modelo subyacente en la **ENOE** para predecir la *Probabilidad Macro de Informalidad Laboral* basada en atributos demográficos (edad, educación, ubicación, género).
    2. Esta probabilidad se inyecta como una **variable latente exógena** en los perfiles de la **ENIGH**.
    
    Este puente estadístico permite que el modelo final (IVLE) entienda la vulnerabilidad de un hogar **no solo por su déficit de ingresos internos, sino también por el riesgo de informalidad laboral inherente a su entorno demográfico**.
    """)

# --- Pie de Página ---


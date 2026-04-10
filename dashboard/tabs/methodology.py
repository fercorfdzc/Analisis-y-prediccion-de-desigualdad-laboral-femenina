import streamlit as st

def render_methodology():
    st.title("Guia Metodologica y Glosario")
    st.markdown("""
    Este dashboard no solo muestra cifras, sino que busca explicar **el porqué** de la desigualdad de genero en Mexico. 
    A continuación, detallamos la metodología técnica para que puedas interpretar los resultados con total claridad.
    """)

    with st.expander("Glosario de Terminos", expanded=True):
        st.markdown("""
        - **Mediana**: El valor que divide a la población en dos partes iguales. A diferencia del promedio, la mediana no se ve distorsionada por salarios extremadamente altos, representando fielmente al "mexicano típico".
        - **PEA (Población Económicamente Activa)**: Personas de 15 años y más que tienen empleo o lo están buscando activamente.
        - **Informalidad Laboral**: Trabajadores que, aunque tienen un ingreso, no cuentan con seguridad social ni contrato laboral formal.
        - **Brecha Salarial**: Diferencia porcentual entre lo que gana un hombre y una mujer ante condiciones similares.
        - **Factor de Expansión**: Peso estadístico que permite que los resultados de la encuesta representen a toda la población de México.
        """)

    with st.expander("Fuentes de Datos"):
        st.markdown("""
        Los datos provienen del **INEGI** (Instituto Nacional de Estadística y Geografía):
        1. **ENOE (Encuesta Nacional de Ocupación y Empleo)**: Fuente principal para salarios, jornadas laborales y perfiles sociodemográficos.
        2. **ENIGH (Encuesta Nacional de Ingresos y Gastos de los Hogares)**: Fuente para entender la dinámica económica familiar y la carga de cuidados.
        """)

    with st.expander("Metodologia de los Modelos (Machine Learning)"):
        st.markdown("""
        Utilizamos algoritmos de **Inteligencia Artificial** (LightGBM, Ridge y XGBoost) para un análisis profundo:
        
        - **M1 (Participación)**: Predice la probabilidad de inserción laboral según el perfil sociodemográfico.
        - **M2 (Ingresos)**: Estima el ingreso justo ajustado por capital humano (educación y experiencia). Si persiste la diferencia, se atribuye a discriminación o barreras estructurales.
        - **M3 (Informalidad)**: Determina el riesgo de empleo sin prestaciones sociales.
        - **M4 (IA de Diagnóstico)**: Identifica las **Barreras Estructurales** en los hogares. Permite ver cómo la carga de cuidados y la educación impactan la vulnerabilidad económica.
        """)

    with st.expander("Proceso de Limpieza y Optimizacion (Data Pipeline)"):
        st.markdown("""
        Para asegurar la calidad técnica de la propuesta:
        1. **Política de Medianas**: Usamos la mediana como estándar de comparación para evitar sesgos por salarios atípicos.
        2. **Filtro de Outliers (Percentil 95)**: En las visualizaciones de nubes de datos, filtramos el 5% de casos extremos para centrar el análisis en la mayoría de la población.
        3. **Compresión Parquet**: Optimizamos el almacenamiento usando archivos `.parquet`, reduciendo el peso de los datos en un 90% para un despliegue ligero en la nube.
        4. **Muestreo (Sampling)**: Las visualizaciones complejas usan muestras representativas para mejorar la claridad visual sin perder rigor estadístico.
        """)

    st.info("Objetivo final: Que los tomadores de decisiones puedan ver exactamente dónde están las barreras y proponer políticas públicas basadas en evidencia científica.")

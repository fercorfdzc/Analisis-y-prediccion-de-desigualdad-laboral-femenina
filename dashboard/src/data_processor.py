import pandas as pd
import numpy as np
import os

class DataProcessor:
    """
    Clase encargada del Ciclo de Vida de los Datos para el dashboard.
    Realiza la extracción, limpieza, unión (merge) y feature engineering
    de las encuestas ENOE y ENIGH siguiendo los criterios de análisis técnico.
    """
    def __init__(self, dataset_path='Dataset'):
        self.dataset_path = dataset_path

    def load_enoe_data(self):
        """
        Carga y une los 3 microdatos principales de la ENOE:
        1. SDEM (Sociodemográfico): Datos base de la población.
        2. COE1 (Cuestionario de Ocupación): Datos técnicos sobre el trabajo.
        3. COE2 (Cuestionario de Ocupación): Complemento de ingresos y horas.
        
        La unión se realiza mediante una 'Llave Compuesta' (9 columnas) que identifica 
        de forma única a cada individuo en la encuesta.
        """
        # v2: Prioridad a Parquet para despliegue ligero
        parquet_path = os.path.join(self.dataset_path, 'enoe_lite.parquet')
        if os.path.exists(parquet_path):
            try:
                df_enoe = pd.read_parquet(parquet_path)
                return df_enoe, None
            except Exception as e:
                return None, f"Error al cargar Parquet: {e}"

        paths = {
            'sdem': os.path.join(self.dataset_path, 'conjunto_de_datos_sdem_enoe_2025_4t.csv'),
            'coe1': os.path.join(self.dataset_path, 'conjunto_de_datos_coe1_enoe_2025_4t.csv'),
            'coe2': os.path.join(self.dataset_path, 'conjunto_de_datos_coe2_enoe_2025_4t.csv')
        }
        
        rk = dict(encoding='latin-1', low_memory=False)
        
        try:
            sdem = pd.read_csv(paths['sdem'], **rk)
            coe1 = pd.read_csv(paths['coe1'], **rk)
            coe2 = pd.read_csv(paths['coe2'], **rk)
        except FileNotFoundError as e:
            return None, f"Error: No se encontró el archivo {e.filename}"

        # Estandarizar nombres de columnas a minúsculas para evitar errores de case-sensitivity
        for df in [sdem, coe1, coe2]:
            df.columns = df.columns.str.strip().str.lower()

        # Llave compuesta definida por INEGI
        llave = ['cd_a', 'cve_ent', 'con', 'upm', 'd_sem', 'n_pro_viv', 'v_sel', 'n_hog', 'n_ren']
        
        for df in [sdem, coe1, coe2]:
            for col in llave:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # Selección de columnas específicas del COE2 necesarias para los modelos de ingresos
        cols_coe2 = llave + ['p6b2', 'p6c', 'p6d', 'p6b1']
        cols_coe2 = [c for c in cols_coe2 if c in coe2.columns]

        # Merge secuencial: Sociodemográfico + Cuestionarios
        df_merged = sdem.merge(coe1, on=llave, how='left', suffixes=('', '_c1'))
        df_merged = df_merged.merge(coe2[cols_coe2], on=llave, how='left')
        df_merged.drop(columns=[c for c in df_merged.columns if c.endswith('_c1')], inplace=True)

        return df_merged, None

    def clean_enoe_data(self, df):
        """
        Limpia los datos unificados y crea variables derivadas:
        - Reemplaza códigos de 'No sabe/No responde' (ej: 99, 999) por valores nulos (NaN).
        - Filtra la muestra a Población Adulta (>= 15 años).
        - Crea flags binarios para género, participación, ocupación y formalidad.
        """
        # Clasificación de variables según su naturaleza técnica
        cols_num = ['eda', 'anios_esc', 'hrsocup', 'ingocup', 'n_hij', 'fac_tri']
        cols_ord = ['niv_ins', 't_loc_tri']
        cols_nom = ['scian', 'pos_ocu', 'e_con', 'cve_ent']
        cols_bin = ['sex', 'ur', 'clase1', 'clase2', 'emp_ppal', 'seg_soc', 'sub_o', 'remune2c', 'pre_asa']
        cols_coe2_vars = ['p6b2', 'p6c', 'p6d', 'p6b1']

        todas = cols_num + cols_ord + cols_nom + cols_bin + cols_coe2_vars
        todas = [c for c in todas if c in df.columns]

        raw = df[todas].copy()
        for col in todas:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')

        # Diccionario de códigos de 'limpieza' según el manual de usuario de la ENOE
        codigos_nulos = {
            'eda'    : [97, 98, 99],    # Edades no especificadas
            'anios_esc': [99],          # Escolaridad desconocida
            'n_hij'  : [99],
            'hrsocup': [999],           # Horas no especificadas
            'ingocup': [999998, 999999],# Ingresos no especificados o 'Prefiere no decir'
            'niv_ins': [5, 9],
            'e_con'  : [9],
            'scian'  : [0, 21],         # Sector no especificado
            'pos_ocu': [0, 5],
            'seg_soc': [3],
            'p6c'    : [9],
            'p6d'    : [9],
            'p6b1'   : [9],
            'pre_asa': [0],
            'remune2c': [0],
        }
        for col, vals in codigos_nulos.items():
            if col in raw.columns:
                raw[col] = raw[col].replace(vals, np.nan)

        # Filtro técnico: Mayores de 15 años (Población en Edad de Trabajar)
        raw = raw[raw['eda'] >= 15].copy()
        
        # Generación de variables clave para el análisis de género
        raw['es_mujer']          = (raw['sex'] == 2).astype(int)       # 1 si es mujer, 0 si es hombre
        raw['participa_laboral'] = (raw['clase1'] == 1).astype(int)    # PEA (Población Económicamente Activa)
        raw['es_ocupado']        = (raw['clase2'] == 1).astype(int)    # Ocupados vs Desocupados
        raw['es_formal']         = (raw['emp_ppal'] == 2).astype(int)   # Empleo con prestaciones/contrato
        
        return raw

    def load_enigh_data(self):
        """Carga el Concentrado de Hogares de la ENIGH."""
        # v2: Prioridad a Parquet para despliegue ligero
        parquet_path = os.path.join(self.dataset_path, 'enigh_lite.parquet')
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                return df, None
            except Exception as e:
                return None, f"Error al cargar Parquet ENIGH: {e}"

        path = os.path.join(self.dataset_path, 'conjunto_de_datos_concentradohogar_enigh2024_ns.csv')
        try:
            df = pd.read_csv(path)
            return df, None
        except FileNotFoundError:
            return None, f"Error: No se encontró el archivo {path}"

    def clean_enigh_data(self, df):
        """
        Limpia los datos de ENIGH y estandariza flujos financieros:
        - Convierte ingresos TRIMESTRALES (estándar INEGI) a MENSUALES (división entre 3).
        - Filtra columnas clave para el análisis de jefatura y vulnerabilidad.
        """
        cols_relevant = [
            'clase_hog', 'sexo_jefe', 'edad_jefe', 'educa_jefe', 'menores', 'tot_integ',
            'ing_cor', 'gasto_mon', 'ingtrab', 'transfer', 'negocio', 'becas'
        ]
        df_clean = df[cols_relevant].copy()
        
        # Estandarización a valores Mensuales
        cols_finanzas = ['ing_cor', 'gasto_mon', 'ingtrab', 'transfer', 'negocio', 'becas']
        for col in cols_finanzas:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col] / 3
        
        df_clean['es_jefa_mujer'] = (df_clean['sexo_jefe'] == 2).astype(int)
        return df_clean

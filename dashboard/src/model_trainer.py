import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb

class ModelTrainer:
    """
    Clase encargada de la arquitectura, entrenamiento y persistencia 
    de los modelos de Machine Learning (M1, M2, M3).
    """
    def __init__(self, output_dir='dashboard/models'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.random_state = 42

    def train_enoe_models(self, df_clean):
        """
        Entrena los 3 modelos base para el análisis de desigualdad:
        - M1 (Clasificación): Predice si una persona participará en el mercado laboral.
        - M2 (Regresión): Estima el ingreso mensual logarítmico (para análisis de brecha).
        - M3 (Clasificación): Predice la probabilidad de tener un empleo formal.
        """
        print("Entrenando modelos ENOE...")
        
        # --- M1: Participación Laboral ---
        # Objetivo: Determinar la probabilidad de que una mujer entre a la PEA
        feats_m1 = ['eda', 'anios_esc', 'niv_ins', 't_loc_tri', 'e_con', 'cve_ent', 'es_mujer', 'ur']
        raw_m1 = df_clean.dropna(subset=['participa_laboral']).copy()
        X_m1 = raw_m1[feats_m1]
        y_m1 = raw_m1['participa_laboral']

        prepro_m1 = self._get_preprocessor(['eda', 'anios_esc'], ['niv_ins', 't_loc_tri'], ['e_con', 'cve_ent'], ['es_mujer', 'ur'])
        pipe_m1 = Pipeline([
            ('pre', prepro_m1), 
            ('clf', lgb.LGBMClassifier(random_state=self.random_state, verbose=-1))
        ])
        pipe_m1.fit(X_m1, y_m1)
        joblib.dump(pipe_m1, os.path.join(self.output_dir, 'modelo1_participacion.joblib'))
        print("M1 guardado.")

        # --- M2: Ingreso Mensual ---
        # Objetivo: Modelar el salario basado en capital humano y género
        df_m2 = df_clean[(df_clean['es_ocupado'] == 1) & (df_clean['ingocup'] > 0)].copy()
        
        # Winsorizing: Eliminamos el 1% de valores extremos para evitar sesgos por outliers
        q99 = df_m2['ingocup'].quantile(0.99)
        df_m2 = df_m2[df_m2['ingocup'] <= q99].copy()
        
        # Transformación Logarítmica: Reduce la asimetría de la distribución de ingresos
        df_m2['log_ingreso'] = np.log1p(df_m2['ingocup'])
        
        feats_m2 = ['eda', 'anios_esc', 'hrsocup', 'niv_ins', 't_loc_tri', 'scian', 'pos_ocu', 'e_con', 'cve_ent', 'es_mujer', 'ur']
        X_m2 = df_m2[feats_m2]
        y_m2 = df_m2['log_ingreso']

        prepro_m2 = self._get_preprocessor(['eda', 'anios_esc', 'hrsocup'], ['niv_ins', 't_loc_tri'], ['scian', 'pos_ocu', 'e_con', 'cve_ent'], ['es_mujer', 'ur'])
        pipe_m2 = Pipeline([
            ('pre', prepro_m2), 
            ('reg', Ridge(alpha=1.0))
        ])
        pipe_m2.fit(X_m2, y_m2)
        joblib.dump(pipe_m2, os.path.join(self.output_dir, 'modelo2_ingreso.joblib'))
        print("M2 guardado.")

        # --- M3: Informalidad ---
        # Objetivo: Predecir el acceso a seguridad social (Formalidad)
        raw_m3 = df_clean[df_clean['es_ocupado'] == 1].dropna(subset=['es_formal']).copy()
        feats_m3 = ['eda', 'anios_esc', 'niv_ins', 't_loc_tri', 'scian', 'e_con', 'cve_ent', 'es_mujer', 'ur']
        X_m3 = raw_m3[feats_m3]
        y_m3 = raw_m3['es_formal']

        prepro_m3 = self._get_preprocessor(['eda', 'anios_esc'], ['niv_ins', 't_loc_tri'], ['scian', 'e_con', 'cve_ent'], ['es_mujer', 'ur'])
        pipe_m3 = Pipeline([
            ('pre', prepro_m3), 
            ('clf', lgb.LGBMClassifier(random_state=self.random_state, verbose=-1))
        ])
        pipe_m3.fit(X_m3, y_m3)
        joblib.dump(pipe_m3, os.path.join(self.output_dir, 'modelo3_informalidad.joblib'))
        print("M3 guardado.")

    def train_enigh_model(self, df_enigh):
        """
        Entrena el Modelo de Diagnóstico ENIGH (M4):
        Objetivo: Clasificar el perfil del hogar por género para extraer 
        las barreras estructurales (importancia de variables).
        """
        print("Entrenando modelo de diagnóstico ENIGH...")
        df = df_enigh.copy()
        
        # Crear 'con_negocio' si no existe
        if 'negocio' in df.columns and 'con_negocio' not in df.columns:
            df['con_negocio'] = (df['negocio'] > 0).astype(int)
            
        # Variables continuas con logaritmo para normalizar impacto
        cols_log = ['ing_cor', 'ingtrab', 'gasto_mon', 'transfer']
        for col in cols_log:
            if col in df.columns:
                df[col] = np.log1p(df[col].fillna(0))
        
        # Selección de características (basado en el notebook)
        feats = ['edad_jefe', 'educa_jefe', 'menores', 'tot_integ', 'ing_cor', 'ingtrab', 'transfer', 'con_negocio']
        X = df[feats]
        y = df['es_jefa_mujer']
        
        # Preprocesador: numéricas (continuas), ordinales (educación), binarias (negocio)
        prepro = self._get_preprocessor(['edad_jefe', 'menores', 'tot_integ', 'ing_cor', 'ingtrab', 'transfer'], 
                                        ['educa_jefe'], [], ['con_negocio'])
        
        # Pipeline con Pesos Balanceados para manejar el desbalance de clases
        pipe_enigh = Pipeline([
            ('pre', prepro),
            ('clf', lgb.LGBMClassifier(
                random_state=self.random_state, 
                verbose=-1,
                class_weight='balanced',
                n_estimators=100
            ))
        ])
        
        pipe_enigh.fit(X, y)
        joblib.dump(pipe_enigh, os.path.join(self.output_dir, 'modelo4_diagnostico.joblib'))
        print("Modelo ENIGH guardado.")

    def _get_preprocessor(self, num, ord, nom, bin):
        """
        Crea el preprocesador de columnas (pipelines de transformación):
        - Escalamiento para variables numéricas continuas.
        - Codificación ordinal para niveles con jerarquía (educación, tamaño de localidad).
        - Target Encoding para categorías nominales (sectores, estados) - evita explosión de columnas.
        - Imputación de valores faltantes por mediana/moda.
        """
        pre_num = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
        pre_ord = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
        pre_nom = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', TargetEncoder(random_state=self.random_state))])
        pre_bin = SimpleImputer(strategy='median')
        
        return ColumnTransformer([
            ('num', pre_num, num),
            ('ord', pre_ord, ord),
            ('nom', pre_nom, nom),
            ('bin', pre_bin, bin),
        ])

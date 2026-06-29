import pandas as pd
from src.interpreter import ModelInterpreter

class DecisionSupportSystem:
    """
    Sistema experto que toma un modelo predictivo, interpreta sus decisiones usando SHAP,
    y genera recomendaciones accionables basadas en reglas de negocio.
    """
    def __init__(self, model_diagnostico):
        self.model = model_diagnostico
        self.interpreter = ModelInterpreter(self.model, model_type='tree')
        self.is_fitted = False
        
    def fit_background(self, df_background):
        """Inicializa el explainer de SHAP con un fondo representativo."""
        df = df_background.copy()
        
        # Replicar las transformaciones requeridas por el modelo M4
        if 'negocio' in df.columns and 'con_negocio' not in df.columns:
            df['con_negocio'] = (df['negocio'] > 0).astype(int)
            
        cols_log = ['ing_cor', 'ingtrab', 'gasto_mon', 'transfer']
        for col in cols_log:
            if col in df.columns:
                import numpy as np
                df[col] = np.log1p(df[col].fillna(0))
                
        # Tomar una pequeña muestra para no saturar la memoria y asegurar rapidez
        sample = df.sample(n=min(len(df), 500), random_state=42)
        
        # Las variables esperadas por M4 (diagnóstico IVLE)
        feats = ['edad_jefe', 'educa_jefe', 'menores', 'tot_integ', 'ing_cor', 'ingtrab', 'transfer', 'con_negocio', 'riesgo_informalidad_entorno']
        # Nos aseguramos que estén en el dataframe
        valid_feats = [f for f in feats if f in sample.columns]
        
        self.interpreter.fit_explainer(sample[valid_feats])
        self.is_fitted = True

    def get_recommendation(self, profile_df):
        """Dado un DataFrame con 1 fila (el perfil), retorna predicción, explicaciones y recomendación."""
        if not self.is_fitted:
            raise ValueError("El DSS no ha sido inicializado con datos de fondo. Llama a fit_background primero.")
            
        # 1. Predicción (Clase del IVLE: 0=Muy Baja, 1=Baja, 2=Media, 3=Alta, 4=Muy Alta)
        ivle_class = self.model.predict(profile_df)[0]
        # También podemos sacar las probabilidades si el modelo lo permite
        probs = self.model.predict_proba(profile_df)[0]
        prob_alta = probs[3] + probs[4] # Probabilidad combinada de Alta y Muy Alta
        
        # 2. Explicación SHAP
        explanation = self.interpreter.explain_instance(profile_df)
        
        # 3. Analizar las variables que más impulsaron la probabilidad hacia arriba
        shap_vals = explanation['shap_values']
        feat_names = explanation['feature_names']
        
        impacts = pd.DataFrame({
            'variable': feat_names,
            'impacto': shap_vals
        })
        
        # Mapeo de variables técnicas a nombres legibles para el humano
        mapeo = {
            'edad_jefe': 'Edad',
            'educa_jefe': 'Educación',
            'menores': 'Carga de Cuidados (Menores)',
            'tot_integ': 'Tamaño del Hogar',
            'ing_cor': 'Ingreso Total',
            'ingtrab': 'Ingreso Laboral',
            'transfer': 'Dependencia de Apoyos (Transferencias)',
            'con_negocio': 'Emprendimiento',
            'riesgo_informalidad_entorno': 'Riesgo de Informalidad (ENOE)'
        }
        
        # Como ColumnTransformer genera múltiples columnas dummy, agrupamos por la variable original
        impactos_agrupados = {}
        for var, imp in zip(impacts['variable'], impacts['impacto']):
            original_var = next((k for k in mapeo.keys() if k in var), var)
            impactos_agrupados[original_var] = impactos_agrupados.get(original_var, 0) + imp
            
        df_agrupado = pd.DataFrame(list(impactos_agrupados.items()), columns=['variable', 'impacto'])
        df_agrupado['nombre_legible'] = df_agrupado['variable'].map(mapeo).fillna(df_agrupado['variable'])
        
        # Filtrar solo impactos positivos (que empujan hacia la clase de vulnerabilidad)
        barreras_positivas = df_agrupado[df_agrupado['impacto'] > 0].copy()
        top_barreras = barreras_positivas.sort_values(by='impacto', ascending=False).head(3)
        
        # 4. Generar recomendaciones basadas en reglas (Motor de Inferencia)
        recomendaciones = []
        if ivle_class >= 3:
            recomendaciones.append(f"🔴 **Vulnerabilidad Laboral y Estructural: ALTA (Clase {ivle_class})**. El perfil presenta barreras significativas. Se requieren intervenciones estructurales.")
        elif ivle_class == 2:
            recomendaciones.append("🟡 **Vulnerabilidad Laboral y Estructural: MEDIA (Clase 2)**. Perfil con vulnerabilidad moderada. Riesgo de precarización si ocurren shocks externos.")
        else:
            recomendaciones.append(f"🟢 **Vulnerabilidad Laboral y Estructural: BAJA (Clase {ivle_class})**. Perfil con fuerte autonomía financiera y estructural. Se sugieren políticas de crecimiento.")
            
        barreras_list = top_barreras['variable'].tolist()
        
        if 'menores' in barreras_list:
            recomendaciones.append("👶 **Carga de Cuidados:** Priorizar acceso a guarderías (Sistema Nacional de Cuidados). El tiempo dedicado al hogar limita la participación laboral formal.")
            
        if 'educa_jefe' in barreras_list:
            recomendaciones.append("📚 **Capital Humano:** El rezago educativo es un cuello de botella. Ofrecer certificación de oficios y educación para adultos flexible.")
            
        if 'transfer' in barreras_list:
            recomendaciones.append("💸 **Autonomía Financiera:** Alta fragilidad ante posibles recortes de transferencias gubernamentales o fluctuación de remesas.")
            
        if 'riesgo_informalidad_entorno' in barreras_list:
             recomendaciones.append("⚠️ **Mercado Laboral (ENOE):** El entorno macroeconómico (perfil en la zona) tiene alta fricción informal. Se sugieren subsidios a la formalización (IMSS).")
             
        if 'ingtrab' in barreras_list or 'ing_cor' in barreras_list:
             recomendaciones.append("📉 **Precarización Económica:** Los ingresos directos son insuficientes para sostener al hogar por encima de las líneas de bienestar urbano.")

        if len(recomendaciones) == 1:
             recomendaciones.append("💡 **Resiliencia:** Fomentar redes de apoyo comunitario y mantener el acceso a servicios de salud universales.")

        return {
            'ivle_class': ivle_class,
            'prob_alta_vulnerabilidad': prob_alta,
            'top_barreras': top_barreras,
            'recomendaciones': recomendaciones,
            'base_value': explanation['base_value'],
            'df_impactos': df_agrupado.sort_values(by='impacto', ascending=False)
        }
            'df_impactos': df_agrupado.sort_values(by='impacto', ascending=False)
        }

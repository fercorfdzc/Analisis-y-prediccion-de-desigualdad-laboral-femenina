import shap
import pandas as pd
import numpy as np

class ModelInterpreter:
    """
    Clase encargada de la interpretabilidad de los modelos usando SHAP.
    Extrae la lógica de preprocesamiento de un pipeline de scikit-learn
    para poder aplicar explainer sobre las variables transformadas.
    """
    def __init__(self, pipeline, model_type='tree'):
        self.pipeline = pipeline
        self.model_type = model_type
        # Extraer componentes del pipeline
        self.preprocessor = self.pipeline.named_steps['pre']
        if 'clf' in self.pipeline.named_steps:
            self.model = self.pipeline.named_steps['clf']
        elif 'reg' in self.pipeline.named_steps:
            self.model = self.pipeline.named_steps['reg']
        else:
            raise ValueError("Modelo no reconocido en el pipeline")
            
        self.explainer = None
        
    def _get_feature_names(self):
        """Intenta extraer los nombres de las variables después del preprocesamiento."""
        try:
            return self.preprocessor.get_feature_names_out()
        except:
            return None

    def fit_explainer(self, X_background):
        """Ajusta el explainer con un dataset de fondo preprocesado."""
        X_trans = self.preprocessor.transform(X_background)
        
        # Convertir a array denso si es una matriz dispersa (sparse matrix)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
            
        if self.model_type == 'tree':
            # Para LightGBM usamos TreeExplainer
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'linear':
            # Para Ridge usamos LinearExplainer
            self.explainer = shap.LinearExplainer(self.model, X_trans)
        else:
            self.explainer = shap.Explainer(self.model, X_trans)

    def explain_instance(self, X_instance):
        """Genera la explicación SHAP para una sola instancia (perfil DataFrame de 1 fila)."""
        if self.explainer is None:
            raise ValueError("Primero debes llamar a fit_explainer con un dataset de fondo.")
            
        X_trans = self.preprocessor.transform(X_instance)
        
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
            
        shap_values = self.explainer.shap_values(X_trans)
        
        # Para modelos de clasificación binaria LGBM o Multiclase
        if isinstance(shap_values, list):
            # Si es multiclase (como IVLE que tiene 5 clases), tomamos la clase predicha o la más alta
            # Pero sin la clase predicha aquí, tomamos la última clase (Alta vulnerabilidad) por defecto,
            # o retornamos toda la lista si queremos que el DSS decida
            shap_values_target = shap_values[-1] # Impacto hacia la máxima vulnerabilidad (Clase 4)
        else:
            # En nuevas versiones de SHAP, shape puede ser (n_samples, n_features, n_classes)
            if len(shap_values.shape) == 3:
                shap_values_target = shap_values[:, :, -1]
            else:
                shap_values_target = shap_values
                
        base_value = self.explainer.expected_value
        if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
            base_value = base_value[-1] # Base value de la máxima vulnerabilidad
            
        feature_names = self._get_feature_names()
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X_trans.shape[1])]
            
        # Limpiar prefijos de ColumnTransformer (ej. 'num__eda' -> 'eda')
        clean_names = [name.split('__')[-1] for name in feature_names]
            
        return {
            'shap_values': shap_values_target[0],
            'base_value': base_value,
            'features_transformed': X_trans[0],
            'feature_names': clean_names
        }

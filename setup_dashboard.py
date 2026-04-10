import os
import sys

# Añadir la carpeta dashboard al path para que las importaciones funcionen
sys.path.append(os.path.join(os.getcwd(), 'dashboard'))

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

def setup():
    print("Iniciando configuracion del Dashboard...")
    
    # 1. Procesamiento de Datos
    processor = DataProcessor(dataset_path='Dataset')
    print("--- Cargando datos de ENOE ---")
    df_enoe, error = processor.load_enoe_data()
    
    if error:
        print(f"Error: {error}")
        return

    print("--- Limpiando datos ---")
    df_clean = processor.clean_enoe_data(df_enoe)
    
    # 2. Entrenamiento de Modelos
    trainer = ModelTrainer(output_dir='dashboard/models')
    trainer.train_enoe_models(df_clean)
    
    # 3. Datos y Modelo ENIGH (Nuevo)
    print("\n--- Cargando datos de ENIGH ---")
    df_enigh, error_enigh = processor.load_enigh_data()
    if not error_enigh:
        print("--- Entrenando modelo de diagnostico ENIGH (M4) ---")
        df_enigh_clean = processor.clean_enigh_data(df_enigh)
        trainer.train_enigh_model(df_enigh_clean)
    else:
        print(f"Aviso: No se pudo cargar ENIGH para el diagnostico: {error_enigh}")

    print("\nConfiguracion completada con exito.")
    print("Ahora puedes ejecutar el dashboard con: streamlit run dashboard/app.py")

if __name__ == "__main__":
    setup()

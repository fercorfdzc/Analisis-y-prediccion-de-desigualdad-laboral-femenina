import pandas as pd
import os

def compress_data(dataset_path='Dataset', output_path='Dataset'):
    print("Iniciando compresion de datos para despliegue...")
    
    # 1. Optimizacion de ENOE
    print("--- Procesando ENOE ---")
    rk = dict(encoding='latin-1', low_memory=False)
    
    llave = ['cd_a', 'cve_ent', 'con', 'upm', 'd_sem', 'n_pro_viv', 'v_sel', 'n_hog', 'n_ren']
    cols_extra = [
        'p6b2', 'p6c', 'p6d', 'p6b1', 'sex', 'ur', 'clase1', 'clase2', 
        'emp_ppal', 'seg_soc', 'sub_o', 'remune2c', 'pre_asa', 
        'eda', 'anios_esc', 'hrsocup', 'ingocup', 'n_hij', 'fac_tri', 
        'niv_ins', 't_loc_tri', 'scian', 'pos_ocu', 'e_con'
    ]
    
    try:
        sdem = pd.read_csv(os.path.join(dataset_path, 'conjunto_de_datos_sdem_enoe_2025_4t.csv'), **rk)
        coe1 = pd.read_csv(os.path.join(dataset_path, 'conjunto_de_datos_coe1_enoe_2025_4t.csv'), **rk)
        coe2 = pd.read_csv(os.path.join(dataset_path, 'conjunto_de_datos_coe2_enoe_2025_4t.csv'), **rk)
        
        for df in [sdem, coe1, coe2]:
            df.columns = df.columns.str.strip().str.lower()
            
        cols_sdem = [c for c in llave + cols_extra if c in sdem.columns]
        cols_coe1 = [c for c in llave + cols_extra if c in coe1.columns]
        cols_coe2 = [c for c in llave + cols_extra if c in coe2.columns]
        
        df_enoe = sdem[cols_sdem].merge(coe1[cols_coe1], on=llave, how='left')
        df_enoe = df_enoe.merge(coe2[cols_coe2], on=llave, how='left')
        
        df_enoe.to_parquet(os.path.join(output_path, 'enoe_lite.parquet'), index=False)
        print("OK: ENOE optimizada: enoe_lite.parquet")
        
    except Exception as e:
        print(f"Error en ENOE: {e}")

    # 2. Optimizacion de ENIGH
    print("\n--- Procesando ENIGH ---")
    try:
        path_enigh = os.path.join(dataset_path, 'conjunto_de_datos_concentradohogar_enigh2024_ns.csv')
        df_enigh = pd.read_csv(path_enigh)
        
        cols_enigh = [
            'clase_hog', 'sexo_jefe', 'edad_jefe', 'educa_jefe', 'menores', 'tot_integ',
            'ing_cor', 'gasto_mon', 'ingtrab', 'transfer', 'negocio', 'becas'
        ]
        df_enigh_lite = df_enigh[cols_enigh].copy()
        
        df_enigh_lite.to_parquet(os.path.join(output_path, 'enigh_lite.parquet'), index=False)
        print("OK: ENIGH optimizada: enigh_lite.parquet")
        
    except Exception as e:
        print(f"Error en ENIGH: {e}")

    print("\nOptimizacion completada. Ya puedes subir estos archivos .parquet a GitHub.")

if __name__ == "__main__":
    compress_data()

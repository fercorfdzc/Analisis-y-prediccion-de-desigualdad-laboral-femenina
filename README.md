# Análisis y Predicción de Desigualdad Laboral Femenina

Este proyecto se enfoca en el análisis de la desigualdad laboral y salarial de género en México. Utiliza datos provenientes de encuestas nacionales para caracterizar la situación actual y desarrollar modelos de aprendizaje automático orientados a predecir y entender diferentes factores de esta desigualdad.

## Estructura del Proyecto

El análisis principal se encuentra alojado en la carpeta `Analisis/`, donde se incluyen los siguientes notebooks:

### 1. Análisis sobre Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH)
* **Archivo:** `Analisis_sobre_Encuesta_Nacional_de_Ingresos_y_Gastos_de_los_Hogares_(ENIGH).ipynb`
* **Descripción:** Este notebook explora los datos de la ENIGH, centrándose en las características del hogar y variables sociodemográficas. Se implementan diversos modelos de clasificación (Random Forest, XGBoost, Regresión Logística, SVM, MLP) para analizar aspectos de los ingresos y gastos familiares, y cómo estos varían o se ven influenciados por diferentes factores.

### 2. Análisis de Desigualdad Laboral de Género — ENOE 2025 4T
* **Archivo:** `enoe_desigualdad_laboral_v2.ipynb`
* **Descripción:** Analiza la Encuesta Nacional de Ocupación y Empleo (ENOE). El objetivo principal es construir modelos de machine learning orientados a alimentar un dashboard interactivo sobre la brecha salarial y participación laboral femenina.
* **Modelos construidos:**
  * **Modelo 1 (Clasificación):** ¿Participa una mujer en el mercado laboral?
  * **Modelo 2 (Regresión):** ¿Cuánto gana una persona y cuánto pierde por ser mujer?
  * **Modelo 3 (Clasificación):** ¿Tiene empleo informal?

## Requisitos Previos

Se recomienda utilizar **Python 3.10 o superior** para asegurar la compatibilidad completa con todas las bibliotecas utilizadas en este proyecto (especialmente con versiones recientes de `pandas`, `scikit-learn` y `lightgbm`).

## Instalación de Dependencias

Es altamente recomendable utilizar un entorno virtual (como `venv` o `conda`) para evitar conflictos con otras bibliotecas en tu sistema.

### Opción 1: Usando `venv` (Python estándar)
1. Abre tu terminal o línea de comandos.
2. Crea el entorno virtual ejecutando:
   ```bash
   python -m venv env
   ```
3. Activa el entorno virtual:
   * En **Windows**:
     ```bash
     .\env\Scripts\activate
     ```
   * En **macOS/Linux**:
     ```bash
     source env/bin/activate
     ```

### Opción 2: Usando `conda`
1. Crea un nuevo entorno:
   ```bash
   conda create -n analisis_genero python=3.10
   ```
2. Activa el entorno:
   ```bash
   conda activate analisis_genero
   ```

### Instalación de requerimientos
Una vez activado tu entorno virtual (por cualquiera de los métodos), instala las bibliotecas requeridas ejecutando:

```bash
pip install -r requirements.txt
```

## Bibliotecas Principales Utilizadas

* **Manejo y Análisis de Datos:** `pandas`, `numpy`
* **Visualización:** `matplotlib`, `seaborn`
* **Modelado y Machine Learning:** `scikit-learn`, `lightgbm`, `xgboost`
* **Explicabilidad (XAI):** `shap`
* **Manejo de Desbalanceo de Datos:** `imbalanced-learn`
* **Serialización de Modelos:** `joblib`

## Colaboradores

* Fernández Córdova Jonathan
* Zaleta Hernández Ximena
* Martínez Domínguez Diego 
* Chama Aguilar Jessica Pola

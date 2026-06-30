# Sistema Inteligente de Apoyo a Decisiones (DSS) con Inteligencia Artificial Explicable (XAI)
**Evaluación Multidimensional de Vulnerabilidad Laboral y Estructural (IVLE)**

Este repositorio contiene el código fuente, modelos y análisis de datos de un proyecto de investigación enfocado en el desarrollo de un **Sistema de Apoyo a Decisiones (DSS)**. El objetivo es diagnosticar la vulnerabilidad estructural y laboral de los hogares mexicanos mediante técnicas avanzadas de Inteligencia Artificial Explicable (XAI) y Fusión de Microdatos. 

Este proyecto fue desarrollado en el contexto del Datatón regional para la igualdad y escalado para su presentación en congresos científicos de ciencias computacionales e inteligencia artificial (**ENC / MICAI**).

---

## 1. Fundamento Metodológico y Problema de Datos

La evaluación de la desigualdad en México enfrenta un reto de **aislamiento de microdatos** oficiales proporcionados por el INEGI:
- **ENIGH (Encuesta Nacional de Ingresos y Gastos de los Hogares):** Alta dimensionalidad en estructura de gastos, transferencias y carencias, pero sin profundidad en precariedad laboral.
- **ENOE (Encuesta Nacional de Ocupación y Empleo):** Captura detallada de la fricción del mercado laboral, acceso a seguridad social y brechas salariales, pero sin rastreo de gastos del hogar.

Para entrenar un modelo de Machine Learning capaz de entender ambos fenómenos simultáneamente, este sistema rechaza el uso de datos agregados o macroeconómicos (los cuales no son aptos para el entrenamiento de IA a nivel de hogar individual) y propone una arquitectura de **Fusión Estadística (Statistical Matching)**.

---

## 2. Arquitectura de Machine Learning

El pipeline de datos y modelado se divide en las siguientes fases técnicas:

### 2.1. Fusión Estadística (Statistical Matching)
Se entrena un modelo subyacente en la ENOE para extraer la probabilidad intrínseca de que una mujer, dada su demografía (edad, nivel educativo, urbanización, región), caiga en el sector informal. 
Esta predicción se inyecta posteriormente a los registros individuales de la ENIGH como una **variable latente exógena** denominada *Riesgo Macro de Informalidad del Entorno*.

### 2.2. Construcción del IVLE (Índice de Vulnerabilidad Laboral y Estructural)
En lugar de predecir variables triviales, el target del modelo principal (M4) es el **IVLE**, un índice sintético creado a partir del espacio de características fusionado:

1. **Extracción del Espacio Latente (PCA Lineal):**
   Las dimensiones de carencia (Dependencia financiera, Tasa de dependencia demográfica, Ingreso per cápita, Educación, Riesgo de informalidad) se proyectan en un espacio ortogonal. 
   **Justificación de Transparencia:** Se rechaza el uso de técnicas de proyección en espacios de Hilbert de dimensión infinita (como *Kernel PCA* o *Autoencoders*) para evitar el problema de la "caja negra". El uso estricto de PCA clásico permite extraer las **Cargas Factoriales (Loadings)** del Componente Principal 1 ($PC1$), garantizando la explicabilidad matemática de la contribución de cada variable a la formación del índice.

2. **Discretización Probabilística (GMM) y Optimización BIC:**
   El espacio latente continuo ($PC1$) se clusteriza utilizando **Gaussian Mixture Models (GMM)**. 
   Para evitar la selección arbitraria del número de clústeres ($k$), el sistema implementa una Búsqueda en Grilla (Grid Search). El hiperparámetro $k$ se optimiza computacionalmente minimizando el **Criterio de Información Bayesiano (BIC)**, donde se demostró algorítmicamente la convergencia óptima en un mínimo local de **$k=5$** subpoblaciones. Estas 5 clases representan la escala ordinal de vulnerabilidad (Muy Baja a Muy Alta).

### 2.3. Aprendizaje Explicable (Clasificador Multiclase)
Una vez etiquetados los hogares con su respectiva clase IVLE, se entrena un clasificador ensamblado **LightGBM (Gradient Boosting Machine)** optimizado para problemas multiclase ordinales. 
Este modelo captura las interacciones no lineales complejas entre las características originales del hogar y su nivel de vulnerabilidad final.

---

## 3. Inteligencia Artificial Explicable (XAI) y Simulador DSS

Para que el modelo pase de ser un experimento de laboratorio a un verdadero **Sistema de Apoyo a Decisiones (DSS)** orientado a políticas públicas, la interfaz de inferencia integra **SHAP (SHapley Additive exPlanations)**.

Durante la evaluación de un hogar específico (Simulador):
- El modelo LightGBM emite una predicción probabilística sobre el nivel de riesgo del hogar.
- El intérprete de Teoría de Juegos de SHAP deconstruye la predicción.
- Genera una descomposición exacta, indicando con valores positivos y negativos cuánto contribuyó cada factor (ej. falta de instrucción, número de menores en el hogar, entorno informal) a empujar al hogar hacia una zona de vulnerabilidad.
- Un motor de inferencia traduce estos impactos matemáticos en **recomendaciones de política pública accionables**.

---

## 4. Estructura del Repositorio

- `/dashboard`: Contiene la aplicación web (Streamlit) y los scripts de arquitectura predictiva.
  - `app.py`: Archivo principal del DSS interactivo.
  - `src/model_trainer.py`: Algoritmos de entrenamiento, Pipeline de sklearn, PCA, GMM, BIC Grid Search, y LightGBM.
  - `src/data_processor.py`: ETL y módulos de Fusión Estadística (Statistical Matching).
  - `src/decision_support.py`: Motor de inferencia y generación de recomendaciones.
  - `src/interpreter.py`: Motor XAI utilizando la librería SHAP adaptada a salidas multiclase.
- `/Analisis`: Notebooks académicos.
  - `evaluacion_IVLE.ipynb`: Notebook de evaluación del IVLE, incluyendo matriz de confusión multiclase, optimización paramétrica del GMM ($k=5$) mediante curvas BIC, y visualización de cargas factoriales (Loadings del PCA).
- `setup_dashboard.py`: Script de instanciación que ejecuta la fusión y el entrenamiento dinámico (Pipeline fit).

---

## 5. Ejecución del Entorno

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. (Opcional si no se han generado los modelos) Entrenar la arquitectura:
   ```bash
   python setup_dashboard.py
   ```
3. Levantar el Sistema de Apoyo a Decisiones (DSS):
   ```bash
   streamlit run dashboard/app.py
   ```

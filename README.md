# DAT4CCIÓN: Análisis y Predicción de Desigualdad Laboral Femenina

Este proyecto es una plataforma interactiva diseñada para visibilizar, analizar y proponer soluciones ante las brechas de género en el mercado laboral y la economía del hogar en México. El objetivo central es analizar, a partir de datos, las barreras y desafíos que enfrentan las mujeres para ejercer sus derechos vinculados al trabajo remunerado, permitiendo comprender los matices y consecuencias de estas desigualdades. Este trabajo sirve como punto de partida para proponer soluciones innovadoras en el marco del **DAT4CCIÓN: Datatón regional para la igualdad 2026**.

---

## Contexto del Problema
En México, la desigualdad de género no es solo una percepción; es una barrera estadística. Las mujeres enfrentan:
- **Techos de cristal**: Menor probabilidad de alcanzar puestos directivos.
- **Pisos pegajosos**: Responsabilidades de cuidados no remuneradas que limitan su entrada al mercado laboral.
- **Brecha Salarial**: Diferencias de ingreso incluso al comparar personas con el mismo nivel educativo.

Este dashboard utiliza **Ciencia de Datos** y **Machine Learning** para cuantificar estos fenómenos.

---

## Metodologia Tecnica (Sin Suposiciones)

### 1. Fuentes de Datos
Utilizamos dos de las encuestas más robustas de México:
- **ENOE (Encuesta Nacional de Ocupación y Empleo)**: Fotografía del mercado laboral (salarios, jornadas).
- **ENIGH (Encuesta Nacional de Ingresos y Gastos de los Hogares)**: Dinámica económica familiar.

### 2. Procesamiento de Datos (Pipeline)
- **Merge de Tablas**: Unión de archivos mediante llave compuesta de 9 variables.
- **Standard de Medianas**: Priorizamos la **Mediana** sobre el promedio para evitar sesgos por salarios atípicos.
- **Filtro de Outliers**: Limpieza estadística (Percentil 95) para centrar visualizaciones en la mayoría de la población.

### 3. Modelos de Machine Learning
| Modelo | Tipo | Objetivo | Variables Predictoras |
| :--- | :--- | :--- | :--- |
| **M1: Participacion** | Clasificación (LightGBM) | Probabilidad de entrar al mercado laboral. | Edad, Escolaridad, Estado Civil, Zona, Genero. |
| **M2: Ingresos** | Regresión (Ridge) | Estimar el salario justo por perfil. | Capital Humano, Sector SCIAN, Horas, Genero. |
| **M3: Informalidad** | Clasificación (LightGBM) | Riesgo de empleo sin prestaciones. | Sector, Escolaridad, Entidad, Genero. |
| **M4: Diagnostico** | Clasificación (LGBM) | Identificar barreras estructurales (Determinantes de brecha). | Ingresos, Cuidados, Transferencias, Negocio. |

---

## Arquitectura y Despliegue (Streamlit Cloud)

Para garantizar la viabilidad técnica y rapidez del dashboard en la nube, implementamos:

1. **Optimizacion con Parquet**: Redujimos el peso de los microdatos en un 90% mediante archivos vinculados `.parquet`.
2. **Setup Automatico**: `setup_dashboard.py` automatiza la limpieza y el entrenamiento de los 4 modelos de IA.

---

## Guia de Instalacion y Uso

### Instalacion Rapida
1. **Crear entorno**: `conda create -n dataton python=3.10`
2. **Instalar dependencias**: `pip install -r requirements.txt`

### Ejecucion
1. **Configuracion Inicial**: `python setup_dashboard.py`
2. **Lanzar Dashboard**: `streamlit run dashboard/app.py`

---

## Hallazgos del Diagnostico de IA
- **M4 - Barreras de Cuidado**: La presencia de menores es el predictor más fuerte de la vulnerabilidad económica en hogares con jefatura femenina.
- **Techos de Cristal**: La brecha salarial absoluta es más persistente en perfiles profesionales, evidenciando barreras no académicas.
- **Participacion**: El matrimonio suele correlacionarse con una mayor participación masculina pero una menor femenina.

---

## Colaboradores
- Fernández Córdova Jonathan
- Ximena Zaleta Hernández
- Martínez Domínguez Diego 
- Chama Aguilar Jessica Pola

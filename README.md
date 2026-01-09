
# 🏙️ TFM — Análisis del mercado inmobiliario en el área metropolitana de Barcelona

Aplicación interactiva desarrollada con **Streamlit** como parte del **Trabajo Final de Máster Ciencia de Datos (TFM)** de **Abel Mora Vázquez** (UOC, curso 2025–2026).

La aplicación permite analizar el mercado inmobiliario del área metropolitana de Barcelona desde una
perspectiva **objetiva y basada en datos**, integrando información urbana, sociodemográfica y modelos de
*Machine Learning*.

---

## 🚀 Aplicación (Streamlit Cloud)

- **App:** https://tfm-abel-mora-vazquez.streamlit.app/

---
## 🔍 Funcionalidades principales

- Recomendación de distritos según preferencias del usuario (seguridad, limpieza, transporte, tranquilidad).
- Visualización de servicios urbanos y movilidad mediante mapas interactivos.
- Comparación entre **valor percibido** y **valor real** del mercado inmobiliario.
- Exploración de listados inmobiliarios (compra, alquiler y habitaciones).
- Identificación de viviendas “joya” y viviendas sobrevaloradas.
- Predicción orientativa del precio de una vivienda mediante un modelo **CatBoost**.

---

```text
.
├── app.py
├── requirements.txt
├── models/
│   └── catboost_model.cbm
└── data_app/
    ├── metrics_app.parquet
    ├── perfil_app.parquet
    ├── geo_app.parquet
    ├── macro_valor_app.parquet
    ├── viviendas_app.parquet
    ├── importancia_app.parquet
    ├── rmse_app.parquet
    ├── compra_app.parquet
    ├── alquiler_app.parquet
    └── habitaciones_app.parquet

```

---

## 🧰 Requisitos

- **Python 3.10 o superior**
- Dependencias especificadas en `requirements.txt`

Librerías principales utilizadas:
- streamlit
- pandas
- numpy
- folium
- streamlit-folium
- catboost
- matplotlib
- pyarrow

---

## ▶️ Ejecución en local

1️⃣ Instalar las dependencias:

pip install -r requirements.txt

2️⃣ Ejecutar la aplicación:
streamlit run app.py

---

## 📊 Fuentes de datos

Datos sociodemográficos y urbanos:
Open Data del Ajuntament de Barcelona.

Datos inmobiliarios:
Datos obtenidos mediante la API de Idealista (Diciembre de 2025).

Los datos han sido procesados, agregados y anonimizados para su uso académico.

---

## 🧠 Modelo predictivo

Modelo final: CatBoostRegressor

Objetivo: estimación orientativa del precio de viviendas.

Variables utilizadas: características estructurales del inmueble, localización y componentes
principales derivados de variables urbanas y encuesta a vecinos.

---

## 📄 Licencia

Este proyecto está bajo una
Licencia Creative Commons Atribución 4.0 Internacional (CC BY 4.0).

Se permite copiar, redistribuir y adaptar el material, siempre que se reconozca la autoría
y se indique si se han realizado cambios.

🔗 https://creativecommons.org/licenses/by/4.0

---

## 👤 Autor

Abel Mora Vázquez

Graduado en Ciencia de Datos Aplicada – Universitat Oberta de Catalunya (UOC)

Estudiante del Máster en Ciencia de Datos – UOC

🔗 LinkedIn: https://www.linkedin.com/in/abelmoravazquez

🔗 GitHub: https://github.com/Abelibz


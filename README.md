# 🎯 Proyecto de Futbol – Visualización e Integración (Altair + Streamlit)

Este proyecto integra los resultados de análisis y modelado realizados en etapas anteriores.  
Incluye **visualizaciones interactivas** con Altair y una **aplicación en Streamlit** que permite explorar los datos y **probar el modelo entrenado** con nuevos inputs.

---

## 🧭 Estructura del proyecto

```
.
├─ streamlit_app.py                    # App principal (Altair + Streamlit)
├─ data/
│  └─ datos_procesados_modelo_v2.csv   # Dataset base
├─ notebooks/
│  └─ 01_altair_exploracion.ipynb      # (opcional) análisis exploratorio
├─ requirements.txt
└─ README.md
```

## ⚙️ Instalación y uso local

### 1. Clonar el repositorio
```bash
git clone https://github.com/agudgithub/streamlit-futbol.git
cd <streamlit-futbol>
```

### 2. Instalar dependencias
Asegurate de tener Python 3.10+ y ejecutá:
```bash
pip install -r requirements.txt
```

> ⚠️ Importante: el `.pkl` del modelo fue entrenado con una versión específica de `scikit-learn` y `imbalanced-learn`.  
> Si cambia la versión, la app podría mostrar errores al cargarlo.  
> Ajustá el `requirements.txt` según la versión usada en tu entrenamiento.

Ejemplo de archivo recomendado:

```txt
streamlit>=1.38
altair>=5.2
pandas>=2.1
numpy>=1.26
scikit-learn==1.6.1
imbalanced-learn==0.12.3
gdown>=5.1
joblib>=1.3
```

### 3. Obtener los datos
Podés copiar manualmente tu CSV o descargarlo desde Google Drive:

```bash
python - <<'PY'
import gdown, os
os.makedirs("data", exist_ok=True)
gdown.download(id="1t3zZh2CV5IBEV3Jwp1mBF0RoRridPg5-", 
               output="data/datos_procesados_modelo_v2.csv", quiet=False)
PY
```

### 4. Ejecutar la aplicación
```bash
streamlit run streamlit_app.py
```

---

## 🧩 Uso de la aplicación

### 🔹 Pestaña **Exploración**
- Permite **filtrar por equipo** y visualizar:
  - **Ventaja de winrate vs. diferencia de goles**  
  - **Distribución acumulada (ECDF) de goles por resultado**  
  - **Posesión local vs. visitante (facetado por resultado)**  

### 🔹 Pestaña **Probar modelo**
- Carga el modelo entrenado (`modelo_final.pkl`) desde Google Drive.
- Permite ingresar nuevos valores (equipos y variables numéricas clave).
- Genera una predicción de resultado (`Local`, `Empate`, `Visitante`)  
  con gráfico de probabilidades si el modelo lo soporta.

### 🔹 Pestaña **Acerca de**
- Resume dependencias, fuentes de datos y consejos para reproducir el entorno.

---

## 📊 Visualizaciones (Altair)

| Gráfico | Tipo | Propósito | Interactividad |
|----------|------|------------|----------------|
| **Ventaja de winrate vs. diferencia de goles** | Dispersión | Relación entre desempeño previo y resultado | Selector de equipo, tooltip |
| **ECDF de goles por resultado** | Curva acumulada | Comparar distribuciones de goles | Tooltip |
| **Posesión local vs visitante (facet)** | Facet + tendencia | Ver diferencias en estilo de juego según resultado | Facetas por clase, línea de tendencia |

---

## 🌐 Deploy en Streamlit Cloud

1. Subí el repo completo a GitHub.
2. Entrá en [Streamlit Community Cloud](https://share.streamlit.io/).
3. Clic en **“New app”** y configurá:
   - **Repository:** `<tu-usuario>/<tu-repo>`
   - **Branch:** `main`
   - **Main file:** `streamlit_app.py`
4. Deploy → la app quedará disponible en  
   `https://<tu-usuario>-<tu-repo>.streamlit.app/`

---

## 🧾 Criterios de evaluación

| Criterio | Cumplimiento |
|-----------|---------------|
| 2–3 visualizaciones interactivas (Altair) | ✅ |
| Aplicación Streamlit funcional | ✅ |
| Integración del modelo predictivo | ✅ |
| Comunicación clara y reproducible | ✅ |
| Deploy en Streamlit Cloud + README detallado | ✅ |

---

## 🧠 Créditos

- **Autor/es:** [Tu nombre / grupo]
- **Materia:** Ciencia de Datos – UTN FRM  
- **Docente:** [nombre del docente si aplica]  
- **Herramientas:** Python, Altair, Streamlit, scikit-learn, imbalanced-learn  
- **Dataset:** `data/datos_procesados_modelo_v2.csv`  
- **Modelo:** `modelo_final.pkl` (Google Drive)

---

> 🧩 *Este proyecto integra análisis, modelado y visualización, aplicando los principios de la gramática de gráficos y de la comunicación efectiva de resultados mediante una interfaz reproducible.*

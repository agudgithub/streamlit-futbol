# streamlit_app.py
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import re, os, joblib

st.set_page_config(page_title="Análisis y Modelo", layout="wide")
alt.data_transformers.disable_max_rows()

# =========================
# DATA
# =========================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, dayfirst=True)
    # normalizaciones básicas
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    # features derivados
    if {'local_lastN_winrate','visitante_lastN_winrate'}.issubset(df.columns):
        df['delta_winrate'] = df['local_lastN_winrate'] - df['visitante_lastN_winrate']
    if {'local_lastN_possession_avg','visitante_lastN_possession_avg'}.issubset(df.columns):
        df['delta_possession'] = df['local_lastN_possession_avg'] - df['visitante_lastN_possession_avg']
    return df

DATA_PATH = 'data/datos_procesados_modelo_v2.csv'
df = load_data(DATA_PATH)

st.title("Proyecto de Futbol – Visualización e Integración")
st.caption("Altair + Streamlit • Exploración y prueba de modelo")

tab1, tab2, tab3 = st.tabs(["Exploración", "Probar modelo", "Acerca de"])

# =========================
# TAB 1: EXPLORACIÓN (Altair)
# =========================
with tab1:
    # valores para filtro por equipo
    if {'equipo_local_norm','equipo_visitante_norm'}.issubset(df.columns):
        equipos = sorted(set(df['equipo_local_norm']).union(set(df['equipo_visitante_norm'])))
    else:
        equipos = []

    equipo = st.selectbox("Filtrar por equipo (local o visitante)", ['(todos)'] + equipos if equipos else ['(todos)'])

    if equipo == '(todos)' or not equipos:
        data = df
    else:
        data = df[(df['equipo_local_norm'] == equipo) | (df['equipo_visitante_norm'] == equipo)]

    # Chart 1: ventaja winrate vs diferencia de goles
    if {'delta_winrate','diferencia_goles_partido','resultado_texto'}.issubset(data.columns):
        chart1 = (alt.Chart(data).mark_circle(opacity=0.6)
            .encode(
                x=alt.X('delta_winrate:Q', title='Ventaja de winrate (local - visitante)'),
                y=alt.Y('diferencia_goles_partido:Q', title='Diferencia de goles'),
                color=alt.Color('resultado_texto:N', title='Resultado'),
                tooltip=['date:T','equipo_local_norm:N','equipo_visitante_norm:N',
                         'delta_winrate:Q','diferencia_goles_partido:Q','resultado_texto:N']
            ).properties(height=340, title='Ventaja reciente vs. diferencia de goles'))
        st.altair_chart(chart1, use_container_width=True)
    else:
        st.info("Faltan columnas para el Chart 1 (delta_winrate, diferencia_goles_partido, resultado_texto).")

    # ECDF: distribución de goles por resultado
    if {'total_goles_partido','resultado_texto'}.issubset(data.columns):
        base = data[['total_goles_partido','resultado_texto']].dropna().copy().sort_values('total_goles_partido')
        ecdf = (alt.Chart(base)
            .transform_window(
                sort=[{'field':'total_goles_partido'}],
                frame=[None, 0],
                cumulative_count='count(*)'
            )
            .transform_joinaggregate(total='count(*)', groupby=['resultado_texto'])
            .transform_calculate(p='datum.cumulative_count / datum.total')
            .mark_line(point=True)
            .encode(
                x=alt.X('total_goles_partido:Q', title='Goles'),
                y=alt.Y('p:Q', title='Proporción acumulada'),
                color=alt.Color('resultado_texto:N', title='Resultado'),
                tooltip=['total_goles_partido:Q','p:Q','resultado_texto:N']
            ).properties(height=340, title='ECDF: Goles por resultado'))
        st.altair_chart(ecdf, use_container_width=True)
    else:
        st.info("Faltan columnas para el ECDF (total_goles_partido, resultado_texto).")

    # Facet: posesión local vs visitante + tendencia
    if {'local_lastN_possession_avg','visitante_lastN_possession_avg','resultado_texto'}.issubset(data.columns):
        scatter = (alt.Chart(data).mark_point(opacity=0.45)
            .encode(
                x=alt.X('local_lastN_possession_avg:Q', title='Posesión local (prom. N)'),
                y=alt.Y('visitante_lastN_possession_avg:Q', title='Posesión visitante (prom. N)'),
                tooltip=['equipo_local_norm:N','equipo_visitante_norm:N','resultado_texto:N',
                         'local_lastN_possession_avg:Q','visitante_lastN_possession_avg:Q']
            ).properties(width=260, height=220))

        trend = (alt.Chart(data)
            .transform_regression('local_lastN_possession_avg', 'visitante_lastN_possession_avg')
            .mark_line()
            .encode(x='local_lastN_possession_avg:Q', y='visitante_lastN_possession_avg:Q'))

        facet = (scatter + trend).facet(column=alt.Column('resultado_texto:N', title=''),
                                        title='Posesión local vs. visitante (facet por resultado)')
        st.altair_chart(facet, use_container_width=True)
    else:
        st.info("Faltan columnas para el facet (posesiones y resultado_texto).")

# =========================
# UTIL: Drive downloader
# =========================
def _drive_id_from_url(url: str):
    m = re.search(r'/d/([^/]+)/', url)
    return m.group(1) if m else None

@st.cache_resource(show_spinner=False)
def load_model_from_drive(url_or_id: str, out_path='model_final.pkl'):
    import gdown
    file_id = url_or_id if re.fullmatch(r'[A-Za-z0-9_-]{25,}', url_or_id) else _drive_id_from_url(url_or_id)
    if not file_id:
        raise ValueError("URL/ID de Drive inválida.")
    gdown.download(id=file_id, output=out_path, quiet=True)
    return joblib.load(out_path)

# =========================
# HELPERS: armar fila 1xN con todas las features
# =========================
def infer_feature_names(model, fallback_cols):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(fallback_cols)

def build_one_row(df_full, feat_names, overrides: dict):
    """
    Crea una fila (1xN) con TODAS las columnas que el modelo espera.
    - Numéricas: mediana
    - Categóricas/objeto: moda (valor más frecuente)
    - Luego aplica 'overrides' (inputs del usuario)
    Mantiene dtypes compatibles.
    """
    data = {}
    for col in feat_names:
        if col in df_full.columns:
            s = df_full[col]
            if pd.api.types.is_numeric_dtype(s):
                val = float(s.median()) if s.notna().any() else 0.0
            else:
                # modo (valor más frecuente) si existe, sino string vacío
                val = s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else ""
            data[col] = val
        else:
            # columna que el modelo espera pero no está en df -> default
            data[col] = 0.0
    # aplicar overrides del usuario
    for k, v in overrides.items():
        if k in data:
            data[k] = v
    # DataFrame 1xN y castear dtypes similares a df_full
    row = pd.DataFrame([data])
    for col in row.columns:
        if col in df_full.columns:
            if pd.api.types.is_categorical_dtype(df_full[col]):
                row[col] = row[col].astype('category')
            elif pd.api.types.is_numeric_dtype(df_full[col]):
                row[col] = pd.to_numeric(row[col], errors='coerce')
            else:
                # usar dtype original cuando sea posible
                try:
                    row[col] = row[col].astype(df_full[col].dtype)
                except Exception:
                    pass
    return row

# =========================
# TAB 2: PROBAR MODELO
# =========================
with tab2:
    st.subheader("Cargar modelo desde Google Drive")
    url = st.text_input(
        "Pega aquí la URL de tu modelo_final.pkl",
        value="https://drive.google.com/file/d/1nypgIlyosdGE-WCcH_EKDyMhd-oFGTWm/view?usp=drive_link"
    )
    load_btn = st.button("Cargar modelo")
    model = None
    if load_btn:
        with st.spinner("Descargando y cargando el modelo..."):
            try:
                model = load_model_from_drive(url)
                st.success("Modelo cargado.")
            except Exception as e:
                st.error(f"No pude cargar el modelo: {e}")

    st.markdown("### Ingresar datos nuevos")
    st.caption("Completá algunos features clave (incluye equipos). El resto se completa con valores típicos del dataset.")

    # --- Controles de equipos (categóricos obligatorios del pipeline) ---
    if {'equipo_local_norm','equipo_visitante_norm'}.issubset(df.columns):
        equipos_all = sorted(set(df['equipo_local_norm']).union(set(df['equipo_visitante_norm'])))
    else:
        equipos_all = []

    c1, c2 = st.columns(2)
    # Si tenemos lista de equipos, evitamos que el visitante pueda ser el mismo que el local
    if equipos_all:
        sel_local = c1.selectbox("equipo_local_norm", equipos_all)
        visit_options = [e for e in equipos_all if e != sel_local]
        if visit_options:
            sel_visit = c2.selectbox("equipo_visitante_norm", visit_options)
        else:
            # Caso raro: solo hay un equipo en la lista
            sel_visit = c2.selectbox("equipo_visitante_norm", ["(No hay otro equipo disponible)"])
            st.warning("No hay otro equipo distinto disponible para seleccionar como visitante.")
    else:
        # fallback a text inputs (validación se hará antes de predecir)
        sel_local = c1.text_input("equipo_local_norm")
        sel_visit = c2.text_input("equipo_visitante_norm")

    # --- Controles numéricos básicos ---
    cols_simple = [
        'local_lastN_winrate','visitante_lastN_winrate',
        'local_lastN_goals_for','visitante_lastN_goals_for',
        'local_lastN_possession_avg','visitante_lastN_possession_avg',
        'local_lastN_remates_puerta','visitante_lastN_remates_puerta'
    ]
    med = df.median(numeric_only=True).to_dict()
    user_vals = {'equipo_local_norm': sel_local, 'equipo_visitante_norm': sel_visit}

    # Validación en caliente: mostrar aviso si los equipos son iguales (o placeholder)
    teams_different = True
    if isinstance(sel_local, str) and isinstance(sel_visit, str):
        if sel_local.strip() == sel_visit.strip():
            teams_different = False
            st.error("El equipo local y el equipo visitante deben ser diferentes. Por favor corregí la selección.")
        if sel_visit == "(No hay otro equipo disponible)":
            teams_different = False
            st.error("No es posible seleccionar el mismo equipo como visitante. Añadí más equipos al dataset o usa otro CSV.")
    colA, colB = st.columns(2)
    for i, c in enumerate(cols_simple):
        w = colA if i % 2 == 0 else colB
        default = float(med.get(c, 0.0)) if c in med else 0.0
        user_vals[c] = w.number_input(c, value=default)

    if st.button("Predecir"):
        try:
            # bloquear predicción si equipos iguales
            if not teams_different:
                st.error("No se puede predecir: el equipo local y visitante deben ser diferentes.")
                raise SystemExit()

            # si el user no apretó "Cargar modelo" antes
            if model is None:
                model = load_model_from_drive(url)

            # columnas esperadas por el modelo
            feat = infer_feature_names(model, df.columns)

            # debug opcional: mostrar features esperadas y faltantes respecto al DataFrame base
            with st.expander("Ver columnas esperadas por el modelo"):
                expected = set(feat)
                have = set(df.columns)
                missing_in_df = sorted(expected - have)
                st.write("Total esperadas:", len(feat))
                st.write("Algunas (primeras 30):", feat[:30])
                if missing_in_df:
                    st.warning(f"Columnas esperadas que no están en el CSV: {missing_in_df}")

            # construir fila 1xN
            row = build_one_row(df, feat, user_vals)

            # asegurar orden de columnas
            X = row[feat] if all(f in row.columns for f in feat) else row

            # predicción
            y_pred = model.predict(X)[0]
            st.success(f"Predicción: **{y_pred}**")

            # probabilidades, si existen
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                if hasattr(model, "classes_"):
                    proba_df = pd.DataFrame({"clase": model.classes_, "prob": proba})
                    st.altair_chart(
                        alt.Chart(proba_df).mark_bar().encode(x='clase:N', y='prob:Q', tooltip=['clase','prob']),
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error al predecir: {e}")

# =========================
# TAB 3: ACERCA DE
# =========================
with tab3:
    st.markdown("""
**Datos**: `data/datos_procesados_modelo_v2.csv`  
**Visualizaciones**: Altair (interactivas, comparables, y con facetado)  
**Modelo**: se carga desde Google Drive y se prueba con inputs del usuario.

> Consejo: fijá en `requirements.txt` la **misma versión de scikit-learn** e **imbalanced-learn** con la que entrenaste el modelo para evitar incompatibilidades al cargar el .pkl.
""")

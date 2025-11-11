# streamlit_app.py
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import re, os, joblib, unicodedata

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
    # (no normalization needed — CSV tiene nombres finales)
    return df


# Nota: ya no se necesita normalizar nombres de equipos — el CSV viene correcto

# Placeholder for df; will cargar según la elección de modelo más abajo
df = None

st.title("Proyecto de Futbol – Visualización e Integración")
st.caption("Altair + Streamlit • Exploración y prueba de modelo")

# Selección global de modelo/dataset (afecta los CSVs que se cargan)
MODEL_CSV_MAP = {
    'Gradiente': 'df_grid_procesado_gradiente.csv',
    'Regresion': 'df_grid_procesado_regresion.csv',
    'Ridge': 'df_grid_procesado_ridge.csv'
}
MODEL_FILE_MAP = {
    'Gradiente': 'modelo_final_gradiente.pkl',
    'Regresion': 'modelo_final_regresion.pkl',
    'Ridge': 'modelo_final_ridge.pkl'
}

# Cargar dataset por defecto (para la exploración). El selector de modelo/dataset
# fue movido a la pestaña 'Probar modelo' para no interferir la exploración.
DEFAULT_MODEL_CHOICE = 'Gradiente'
DATA_PATH = os.path.join('data', MODEL_CSV_MAP[DEFAULT_MODEL_CHOICE])
df = load_data(DATA_PATH)

# asegurar existencia de session_state para el modelo
if 'model' not in st.session_state:
    st.session_state['model'] = None

tab1, tab2, tab3 = st.tabs(["Exploración", "Probar modelo", "Acerca de"])

# =========================
# TAB 1: EXPLORACIÓN (Altair)
# =========================
with tab1:
    # valores para filtro por equipo
    if {'equipo_local_norm','equipo_visitante_norm'}.issubset(df.columns):
        raw = list(df['equipo_local_norm'].dropna().astype(str)) + list(df['equipo_visitante_norm'].dropna().astype(str))
        equipos = sorted(set(raw))
    else:
        equipos = []

    if not equipos:
        st.info("No hay equipos disponibles en el CSV para filtrar.")
        data = df
        equipo = None
    else:
        # Ya no mostramos la opción '(todos)'; el usuario debe elegir un equipo
        equipo = st.selectbox("Elegí un equipo", equipos)
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

    # --- UNIFICADO: gráfico parametrizable (Eficiencia / Posesión) ---
    st.markdown("### Visualización parametrizable: Eficiencia / Posesión")
    metric = st.selectbox("Métrica a visualizar", ["Eficiencia", "Posesión"])
    side = st.selectbox("Filtrar por lado del equipo seleccionado", ["Ambos", "Local", "Visitante"])

    # chequear columnas necesarias según métrica
    if metric == "Eficiencia":
        needed = {
            'resultado_texto',
            'local_lastN_goals_for','local_lastN_remates_puerta',
            'visitante_lastN_goals_for','visitante_lastN_remates_puerta'
        }
    else:  # Posesión
        needed = {
            'resultado_texto',
            'local_lastN_possession_avg','visitante_lastN_possession_avg'
        }

    if not needed.issubset(df.columns):
        st.info(f"Faltan columnas necesarias para '{metric}'. Columnas requeridas: {sorted(needed)}")
    elif equipo is None:
        st.info("Seleccioná un equipo para ver el gráfico parametrizable.")
    else:
        # seleccionar subconjunto según 'side' y 'equipo'
        if side == "Local":
            data_sub = df[df['equipo_local_norm'] == equipo].copy()
            title_side = f"partidos donde {equipo} fue LOCAL"
        elif side == "Visitante":
            data_sub = df[df['equipo_visitante_norm'] == equipo].copy()
            title_side = f"partidos donde {equipo} fue VISITANTE"
        else:
            data_sub = df[(df['equipo_local_norm'] == equipo) | (df['equipo_visitante_norm'] == equipo)].copy()
            title_side = f"partidos donde {equipo} fue LOCAL o VISITANTE"

        if data_sub.empty:
            st.info(f"No hay partidos para {title_side}.")
        else:
            # preparar columnas para graficar
            if metric == "Eficiencia":
                # calcular eficiencia en pandas (evita repetir Altair transform_calculate)
                def safe_eff(gf_col, rp_col):
                    gf = pd.to_numeric(data_sub.get(gf_col), errors='coerce').fillna(0)
                    rp = pd.to_numeric(data_sub.get(rp_col), errors='coerce')
                    rp = rp.replace(0, np.nan)
                    eff = gf / rp
                    eff = eff.clip(upper=1)
                    return eff

                data_sub['local_eff'] = safe_eff('local_lastN_goals_for', 'local_lastN_remates_puerta')
                data_sub['visit_eff'] = safe_eff('visitante_lastN_goals_for', 'visitante_lastN_remates_puerta')
                x_field, y_field = 'local_eff', 'visit_eff'
                x_title, y_title = 'Eficiencia Local (goles / remates a puerta)', 'Eficiencia Visitante (goles / remates a puerta)'
                x_scale = alt.Scale(domain=[0,1])
                y_scale = alt.Scale(domain=[0,1])
            else:
                # Posesión (valores ya esperados como promedios)
                data_sub['local_poss'] = pd.to_numeric(data_sub.get('local_lastN_possession_avg'), errors='coerce')
                data_sub['visit_poss'] = pd.to_numeric(data_sub.get('visitante_lastN_possession_avg'), errors='coerce')

                # Algunas fuentes guardan posesión como 0-1; si ese es el caso convertir a 0-100
                max_val = pd.concat([data_sub['local_poss'].dropna(), data_sub['visit_poss'].dropna()]).max() if not data_sub.empty else None
                if max_val is not None and max_val <= 1.01:
                    data_sub['local_poss'] = data_sub['local_poss'] * 100.0
                    data_sub['visit_poss'] = data_sub['visit_poss'] * 100.0

                # campos para mostrar en tooltip ya formateados
                data_sub['local_poss_pct'] = data_sub['local_poss'].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")
                data_sub['visit_poss_pct'] = data_sub['visit_poss'].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")

                x_field, y_field = 'local_poss', 'visit_poss'
                x_title, y_title = 'Posesión Local (%)', 'Posesión Visitante (%)'
                # for possession, fix axis to 0-100 for clarity
                x_scale = alt.Scale(domain=[0, 100])
                y_scale = alt.Scale(domain=[0, 100])

            plot_df = data_sub.dropna(subset=[x_field, y_field])
            if plot_df.empty:
                st.info("No hay filas válidas con valores numéricos para graficar.")
            else:
                # preparar tooltip y tamaño de puntos según métrica
                if metric == "Posesión":
                    tooltip = ['date:T','equipo_local_norm:N','equipo_visitante_norm:N','local_poss_pct:N','visit_poss_pct:N','resultado_texto:N']
                    point_size = 60
                    x_axis = alt.Axis(format='.0f')
                    y_axis = alt.Axis(format='.0f')
                else:
                    tooltip = ['date:T','equipo_local_norm:N','equipo_visitante_norm:N',f'{x_field}:Q', f'{y_field}:Q', 'resultado_texto:N']
                    point_size = 80
                    # Para eficiencia aseguramos ejes visibles con formato y ticks
                    x_axis = alt.Axis(format='.2f', tickCount=5, grid=True)
                    y_axis = alt.Axis(format='.2f', tickCount=5, grid=True)

                scatter = (alt.Chart(plot_df).mark_point(opacity=0.7, size=point_size)
                           .encode(
                               x=alt.X(f'{x_field}:Q', title=x_title, scale=x_scale, axis=x_axis),
                               y=alt.Y(f'{y_field}:Q', title=y_title, scale=y_scale, axis=y_axis),
                               color=alt.Color('resultado_texto:N', title='Resultado'),
                               tooltip=tooltip
                           ))

                trend = (alt.Chart(plot_df)
                         .transform_regression(x_field, y_field, method="linear")
                         .mark_line(color='black')
                         .encode(x=f'{x_field}:Q', y=f'{y_field}:Q'))

                inner = (scatter + trend).properties(width=260, height=240)
                unified = inner.facet(
                    column=alt.Column('resultado_texto:N', title='Resultado del Partido'),
                    title=f"{metric} — {title_side}"
                )

                st.altair_chart(unified, use_container_width=True)

# =========================
# UTIL: Drive downloader
# =========================
def _drive_id_from_url(url: str):
    m = re.search(r'/d/([^/]+)/', url)
    return m.group(1) if m else None

@st.cache_resource(show_spinner=False)
def load_model_from_drive(url_or_id: str, out_path='modelo_final_gradiente.pkl'):
    import gdown
    file_id = url_or_id if re.fullmatch(r'[A-Za-z0-9_-]{25,}', url_or_id) else _drive_id_from_url(url_or_id)
    if not file_id:
        raise ValueError("URL/ID de Drive inválida.")
    gdown.download(id=file_id, output=out_path, quiet=True)
    return joblib.load(out_path)

# =========================
# HELPERS: armar fila 1xN con todas las features
# =========================
# ======= Preprocessing helpers (same logic used in training preprocessing) =======
def normalizar_texto(s):
    """Quita acentos, minúsculas, espacios y números."""
    if pd.isna(s) or s == "":
        return ""
    s = str(s).strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r'\d', '', s)  # quita números
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('.', '').replace('-', '')
    s = s.strip()
    return s

def to_int_safe(value):
    try:
        if value is None:
            return 0
        match = re.search(r'\d+', str(value))
        return int(match.group()) if match else 0
    except (ValueError, TypeError):
        return 0

def to_float_safe(value):
    try:
        if value is None:
            return 0.0
        value_str = str(value).replace('%', '').strip()
        match = re.search(r'[\d.]+', value_str)
        return float(match.group()) if match else 0.0
    except (ValueError, TypeError):
        return 0.0

def to_float_first_number(x):
    if pd.isna(x) or x == "":
        return 0.0
    s = str(x).replace('%', '').replace('％', '').replace(',', '.').strip()
    m = re.search(r'-?\d+(\.\d+)?', s)
    return float(m.group(0)) if m else 0.0


def calc_metrics(prev_df, equipo_norm):
    resumen = { 'matches': 0, 'wins': 0, 'winrate': 0.0, 'goals_for': 0, 'goals_against': 0,
                'remates_total': 0, 'remates_puerta': 0, 'possession_avg': 0.0 }
    if prev_df is None or prev_df.shape[0] == 0:
        return resumen
    resumen['matches'] = prev_df.shape[0]
    gf = ga = rt = rp = 0
    poss = []
    wins = 0
    for _, r in prev_df.iterrows():
        # determine which side corresponds to the equipo_norm
        local_norm = r.get('equipo_local_norm') if 'equipo_local_norm' in r else normalizar_texto(r.get('equipo_local', ''))
        if pd.isna(local_norm):
            local_norm = ''
        if str(local_norm) == equipo_norm:
            goals_for = to_int_safe(r.get('goles_local_num') or r.get('goles_local') or r.get('resultado_local'))
            goals_against = to_int_safe(r.get('goles_visitante_num') or r.get('goles_visitante') or r.get('resultado_visitante'))
            rem_total = to_int_safe(r.get('remates_total_local_num') or r.get('remates_total_local'))
            rem_puerta = to_int_safe(r.get('remates_puerta_local_num') or r.get('remates_puerta_local'))
            possession = to_float_safe(r.get('posesion_local_num') or r.get('posesion_local'))
        else:
            goals_for = to_int_safe(r.get('goles_visitante_num') or r.get('goles_visitante') or r.get('resultado_visitante'))
            goals_against = to_int_safe(r.get('goles_local_num') or r.get('goles_local') or r.get('resultado_local'))
            rem_total = to_int_safe(r.get('remates_total_visitante_num') or r.get('remates_total_visitante'))
            rem_puerta = to_int_safe(r.get('remates_puerta_visitante_num') or r.get('remates_puerta_visitante'))
            possession = to_float_safe(r.get('posesion_visitante_num') or r.get('posesion_visitante'))
        gf += goals_for
        ga += goals_against
        rt += rem_total
        rp += rem_puerta
        poss.append(possession)
        if goals_for > goals_against:
            wins += 1
    resumen['wins'] = wins
    resumen['winrate'] = (wins / resumen['matches']) if resumen['matches'] > 0 else 0.0
    resumen['goals_for'] = gf
    resumen['goals_against'] = ga
    resumen['remates_total'] = rt
    resumen['remates_puerta'] = rp
    resumen['possession_avg'] = (sum(poss)/len(poss)) if len(poss) > 0 else 0.0
    return resumen


def build_match_features(df_raw, equipo_local, equipo_visitante, N=6):
    """Genera las features agregadas para la pareja (local, visitante) usando las N últimas apariciones de cada equipo.
    Devuelve un dict con claves tipo 'local_lastN_winrate', 'visitante_lastN_winrate', etc., compatibles con el pipeline.
    Si no puede computar (faltan columnas), devuelve {} para indicar fallback.
    """
    if equipo_local is None or equipo_visitante is None:
        return {}
    local_norm = normalizar_texto(equipo_local)
    visit_norm = normalizar_texto(equipo_visitante)

    df2 = df_raw.copy()
    # ensure normalized team name columns
    if 'equipo_local_norm' not in df2.columns and 'equipo_local' in df2.columns:
        df2['equipo_local_norm'] = df2['equipo_local'].apply(normalizar_texto)
        df2['equipo_visitante_norm'] = df2['equipo_visitante'].apply(normalizar_texto)

    # ensure numeric goal columns
    if 'goles_local_num' not in df2.columns and 'resultado_local' in df2.columns:
        df2['goles_local_num'] = df2['resultado_local'].apply(to_int_safe)
        df2['goles_visitante_num'] = df2['resultado_visitante'].apply(to_int_safe)

    # remates and possession
    num_feats = [
        ('remates_total_local', 'remates_total_local_num'),
        ('remates_total_visitante', 'remates_total_visitante_num'),
        ('remates_puerta_local', 'remates_puerta_local_num'),
        ('remates_puerta_visitante', 'remates_puerta_visitante_num')
    ]
    for col, col_num in num_feats:
        if col in df2.columns:
            df2[col_num] = pd.to_numeric(df2[col], errors='coerce').fillna(0).astype(int)
        elif col_num not in df2.columns:
            df2[col_num] = 0

    if 'posesion_local_num' not in df2.columns:
        if 'posesion_local' in df2.columns:
            df2['posesion_local_num'] = df2['posesion_local'].apply(to_float_first_number)
            df2['posesion_visitante_num'] = df2['posesion_visitante'].apply(to_float_first_number)
        else:
            df2['posesion_local_num'] = 0.0
            df2['posesion_visitante_num'] = 0.0

    # ensure date for ordering
    if 'date' not in df2.columns:
        if 'fecha' in df2.columns:
            try:
                df2['date'] = pd.to_datetime(df2['fecha'], dayfirst=True, errors='coerce')
            except Exception:
                df2['date'] = pd.NaT
        else:
            df2['date'] = pd.NaT

    # find last N matches for each team (sorted by date descending)
    def last_n(team_norm):
        mask = (df2.get('equipo_local_norm') == team_norm) | (df2.get('equipo_visitante_norm') == team_norm)
        sub = df2.loc[mask].copy()
        if 'date' in sub.columns and sub['date'].notna().any():
            sub = sub.sort_values('date', ascending=False)
        else:
            sub = sub.sort_index(ascending=False)
        return sub.head(N)

    prev_local = last_n(local_norm)
    prev_visit = last_n(visit_norm)

    # compute metrics
    metrics_local = calc_metrics(prev_local, local_norm)
    metrics_visit = calc_metrics(prev_visit, visit_norm)

    features = {
        'local_lastN_matches': metrics_local['matches'],
        'local_lastN_wins': metrics_local['wins'],
        'local_lastN_winrate': round(metrics_local['winrate'], 3),
        'local_lastN_goals_for': metrics_local['goals_for'],
        'local_lastN_goals_against': metrics_local['goals_against'],
        'local_lastN_remates_total': metrics_local['remates_total'],
        'local_lastN_remates_puerta': metrics_local['remates_puerta'],
        'local_lastN_possession_avg': round(metrics_local['possession_avg'], 3),
        'visitante_lastN_matches': metrics_visit['matches'],
        'visitante_lastN_wins': metrics_visit['wins'],
        'visitante_lastN_winrate': round(metrics_visit['winrate'], 3),
        'visitante_lastN_goals_for': metrics_visit['goals_for'],
        'visitante_lastN_goals_against': metrics_visit['goals_against'],
        'visitante_lastN_remates_total': metrics_visit['remates_total'],
        'visitante_lastN_remates_puerta': metrics_visit['remates_puerta'],
        'visitante_lastN_possession_avg': round(metrics_visit['possession_avg'], 3)
    }
    return features
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
    st.subheader("Modelo seleccionado")
    # Selector de tipo de modelo/dataset movido aquí (afecta sólo la sección 'Probar modelo')
    model_choice = st.selectbox('Seleccioná el tipo de modelo/dataset', list(MODEL_CSV_MAP.keys()))

    # Intent: cargar automáticamente el modelo asociado al tipo seleccionado.
    # Mostramos mensajes de carga aquí en la pestaña.
    model = None
    default_fname = MODEL_FILE_MAP.get(model_choice)
    model_paths_to_try = [os.path.join('models', default_fname), default_fname]
    model_loaded_path = None
    model_load_messages = []
    for p in model_paths_to_try:
        if p and os.path.exists(p):
            try:
                model = joblib.load(p)
                model_loaded_path = p
                st.session_state['model'] = model
                break
            except Exception as e:
                model_load_messages.append(f"Encontré '{p}' pero no pude cargarlo: {e}")
    if model_loaded_path is None:
        model_load_messages.append(f"No se encontró el archivo de modelo local '{default_fname}'. Colocá el .pkl en la carpeta 'models/' o en la raíz del proyecto con ese nombre para que se cargue automáticamente.")

    # Mostrar estado del intento de carga automática realizado al elegir el modelo.
    if model_loaded_path is not None:
        st.success(f"Modelo cargado automáticamente: {model_loaded_path}")
    else:
        for m in model_load_messages:
            st.warning(m)

    st.markdown("### Ingresar datos nuevos")
    st.caption("Se completan los datos con los preprocesados.")

    # --- Controles de equipos (categóricos obligatorios del pipeline) ---
    if {'equipo_local_norm','equipo_visitante_norm'}.issubset(df.columns):
        raw_all = list(df['equipo_local_norm'].dropna().astype(str)) + list(df['equipo_visitante_norm'].dropna().astype(str))
        equipos_all = sorted(set(raw_all))
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

    # Opciones avanzadas: por defecto están ocultas. Solo se usan si el usuario marca 'usar_advanced'.
    advanced_defaults = {
        'local_lastN_winrate': 0.42,
        'local_lastN_goals_for': 0.31,
        'local_lastN_possession_avg': 0.82,
        'local_lastN_remates_puerta': 0.35,
        'visitante_lastN_winrate': 0.40,
        'visitante_lastN_goals_for': 0.36,
        'visitante_lastN_possession_avg': 0.83,
        'visitante_lastN_remates_puerta': 0.40
    }

    advanced_values = {}
    with st.expander("Opciones avanzadas", expanded=False):
        use_advanced = st.checkbox("Usar opciones avanzadas para la predicción", value=False)
        colA, colB = st.columns(2)
        for i, c in enumerate(cols_simple):
            w = colA if i % 2 == 0 else colB
            default = float(advanced_defaults.get(c, med.get(c, 0.0)))
            # mostrar el control pero no aplicarlo hasta que use_advanced sea True
            advanced_values[c] = w.number_input(c, value=default)

    # si el usuario eligió usar opciones avanzadas, incorporarlas a user_vals
    if 'use_advanced' in locals() and use_advanced:
        user_vals.update(advanced_values)

    if st.button("Predecir"):
        try:
            # bloquear predicción si equipos iguales
            if not teams_different:
                st.error("No se puede predecir: el equipo local y visitante deben ser diferentes.")
                raise SystemExit()

            # Si no hay modelo cargado, intentar cargar el archivo por defecto (models/ -> raíz)
            if model is None:
                default_fname = MODEL_FILE_MAP.get(model_choice)
                tried = False
                for p in [os.path.join('models', default_fname), default_fname]:
                    if p and os.path.exists(p):
                        try:
                            model = joblib.load(p)
                            st.info(f"Modelo cargado desde: {p}")
                            tried = True
                            break
                        except Exception as e:
                            st.warning(f"Encontré '{p}' pero no pude cargarlo: {e}")
                            tried = True
                if not tried:
                    st.error(f"No se cargó ningún modelo. Colocá el archivo '{default_fname}' en 'models/' o en la raíz del proyecto.")
                    raise

            # use model from session_state if present
            if st.session_state.get('model') is not None:
                model = st.session_state.get('model')

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

            # intentar construir features históricas basadas en los equipos seleccionados
            N_map = {'Gradiente': 6, 'Regresion': 9, 'Ridge': 9}
            N = N_map.get(model_choice, 6)
            try:
                computed_features = build_match_features(df, sel_local, sel_visit, N)
            except Exception as e:
                computed_features = {}
                st.warning(f"No pude generar features históricas dinámicas: {e}")

            if computed_features:
                # incorporar las features calculadas como overrides (tienen prioridad sobre medianas)
                user_vals.update(computed_features)
            else:
                st.info("No se pudieron calcular las features históricas para los equipos seleccionados; usaré valores típicos del dataset.")

            # construir fila 1xN
            row = build_one_row(df, feat, user_vals)

            # asegurar orden de columnas
            X = row[feat] if all(f in row.columns for f in feat) else row

            # predicción
            y_pred = model.predict(X)[0]
            st.success(f"Predicción: **{y_pred}**")

            # probabilidades, si existen (mostrar en formato 'Local / Empate / Visitante' cuando sea posible)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes = list(getattr(model, "classes_", list(range(len(proba)))))
                proba_df = pd.DataFrame({"clase": classes, "prob": proba})

                # intentar interpretar etiquetas para mostrar prob. Local/Empate/Visitante
                def interpret_label(cls):
                    s = str(cls).lower()
                    if any(k in s for k in ("local", "home", "casa")):
                        return "Local"
                    if any(k in s for k in ("visit", "away", "fuera")):
                        return "Visitante"
                    if any(k in s for k in ("draw", "empate", "tie")):
                        return "Empate"
                    return None

                mapped = [interpret_label(c) for c in classes]
                if any(m is not None for m in mapped):
                    agg = {"Local": 0.0, "Empate": 0.0, "Visitante": 0.0}
                    for cls, p, m in zip(classes, proba, mapped):
                        if m is None:
                            continue
                        agg[m] += float(p)

                    # mostrar probabilidades con nombres de equipos
                    local_name = user_vals.get('equipo_local_norm', 'Local')
                    visit_name = user_vals.get('equipo_visitante_norm', 'Visitante')
                    st.write(f"Probabilidad que {local_name} (Local) gane: {agg['Local']:.1%}")
                    st.write(f"Probabilidad de empate: {agg['Empate']:.1%}")
                    st.write(f"Probabilidad que {visit_name} (Visitante) gane: {agg['Visitante']:.1%}")

                    prob_display_df = pd.DataFrame({"resultado": [f"{local_name} (Local)", "Empate", f"{visit_name} (Visitante)"],
                                                    "prob": [agg['Local'], agg['Empate'], agg['Visitante']]})
                    st.altair_chart(
                        alt.Chart(prob_display_df).mark_bar().encode(x='resultado:N', y='prob:Q', tooltip=['resultado','prob']),
                        use_container_width=True
                    )
                else:
                    # fallback: mostrar probabilidades por clase tal como las devuelve el modelo
                    proba_df['prob_pct'] = (proba_df['prob'] * 100).round(1).astype(str) + '%'
                    st.write("Probabilidades por clase:")
                    st.table(proba_df)
                    st.altair_chart(
                        alt.Chart(proba_df).mark_bar().encode(x='clase:N', y='prob:Q', tooltip=['clase','prob']),
                        use_container_width=True
                    )
            else:
                st.info("El modelo no provee probabilidades (no implementa predict_proba). Se muestra la predicción númerica o de clase.")
        except Exception as e:
            st.error(f"Error al predecir: {e}")

# =========================
# TAB 3: ACERCA DE
# =========================
with tab3:
    st.markdown("""
**Datos**: `data/df_grid_procesado_gradiente.csv`  
**Visualizaciones**: Altair (interactivas, comparables, y con facetado)  
**Modelo**: se intenta cargar automáticamente el archivo local asociado al tipo de modelo (p. ej. `modelo_final_gradiente.pkl`). Si no está presente, colocá el .pkl en `models/` o en la raíz del proyecto.

> Consejo: fijá en `requirements.txt` la **misma versión de scikit-learn** e **imbalanced-learn** con la que entrenaste el modelo para evitar incompatibilidades al cargar el .pkl.
""")

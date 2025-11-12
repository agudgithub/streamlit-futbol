# streamlit_app.py
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import re, os, joblib, unicodedata, datetime

# =========================
# HELPERS de preprocesamiento (antes de load_data)
# =========================
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

# ========== Fechas completas y N dinámico ==========
def unir_fecha_anio(df):
    df['fecha_str'] = df['fecha'].astype(str).str.zfill(5) + df['anio'].astype(str)
    df['fecha_completa'] = pd.to_datetime(df['fecha_str'], format='%d/%m/%Y', errors='coerce')
    return df

def calcular_N_dinamico(df, fecha_usuario, ventana_meses=12):
    fecha_usuario = pd.to_datetime(fecha_usuario)
    fecha_max = df['fecha_completa'].max()
    if fecha_usuario > fecha_max:
        raise ValueError(f"La fecha elegida ({fecha_usuario.date()}) supera la fecha máxima disponible ({fecha_max.date()})")
    ventana_atras = fecha_usuario - pd.DateOffset(months=ventana_meses)
    rango = df[(df['fecha_completa'] <= fecha_usuario) & (df['fecha_completa'] >= ventana_atras)]
    equipos = pd.concat([rango['equipo_local'], rango['equipo_visitante']]).unique() if not rango.empty else []
    partidos_por_equipo = [rango[(rango['equipo_local']==eq)|(rango['equipo_visitante']==eq)].shape[0] for eq in equipos]
    return min(partidos_por_equipo) if partidos_por_equipo else 1

# ========== Generar features dinámicos (para exploración) ==========
def generar_df_features(df_raw, fecha_referencia=None):
    df = unir_fecha_anio(df_raw.copy())
    if fecha_referencia is not None:
        N = calcular_N_dinamico(df, fecha_referencia, 24)  # 2 años hacia atrás si se provee fecha
    else:
        N = 3
    df['equipo_local_norm'] = df['equipo_local'].apply(normalizar_texto)
    df['equipo_visitante_norm'] = df['equipo_visitante'].apply(normalizar_texto)
    df['goles_local_num'] = df['resultado_local'].apply(to_int_safe)
    df['goles_visitante_num'] = df['resultado_visitante'].apply(to_int_safe)
    num_feats = [
        ('remates_total_local', 'remates_total_local_num'),
        ('remates_total_visitante', 'remates_total_visitante_num'),
        ('remates_puerta_local', 'remates_puerta_local_num'),
        ('remates_puerta_visitante', 'remates_puerta_visitante_num')
    ]
    for col, col_num in num_feats:
        if col in df.columns:
            df[col_num] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col_num] = 0
    df['posesion_local_num'] = df['posesion_local'].apply(to_float_first_number) if 'posesion_local' in df.columns else 0.0
    df['posesion_visitante_num'] = df['posesion_visitante'].apply(to_float_first_number) if 'posesion_visitante' in df.columns else 0.0

    def _calc_metrics(prev_df, equipo_norm):
        resumen = { 'matches': 0, 'wins': 0, 'winrate': 0.0, 'goals_for': 0, 'goals_against': 0,
                    'remates_total': 0, 'remates_puerta': 0, 'possession_avg': 0.0 }
        if prev_df is None or prev_df.shape[0] == 0:
            return resumen
        resumen['matches'] = prev_df.shape[0]
        gf = ga = rt = rp = 0
        poss = []
        wins = 0
        for _, r in prev_df.iterrows():
            if normalizar_texto(r['equipo_local']) == equipo_norm:
                goals_for = to_int_safe(r.get('goles_local_num'))
                goals_against = to_int_safe(r.get('goles_visitante_num'))
                rem_total = to_int_safe(r.get('remates_total_local_num'))
                rem_puerta = to_int_safe(r.get('remates_puerta_local_num'))
                possession = to_float_safe(r.get('posesion_local_num'))
            else:
                goals_for = to_int_safe(r.get('goles_visitante_num'))
                goals_against = to_int_safe(r.get('goles_local_num'))
                rem_total = to_int_safe(r.get('remates_total_visitante_num'))
                rem_puerta = to_int_safe(r.get('remates_puerta_visitante_num'))
                possession = to_float_safe(r.get('posesion_visitante_num'))
            gf += goals_for; ga += goals_against; rt += rem_total; rp += rem_puerta; poss.append(possession)
            if goals_for > goals_against: wins += 1
        resumen['wins'] = wins
        resumen['winrate'] = (wins / resumen['matches']) if resumen['matches'] > 0 else 0.0
        resumen['goals_for'] = gf; resumen['goals_against'] = ga
        resumen['remates_total'] = rt; resumen['remates_puerta'] = rp
        resumen['possession_avg'] = (sum(poss)/len(poss)) if len(poss) > 0 else 0.0
        return resumen

    out_cols = [
        'fecha_completa', 'equipo_local_norm', 'equipo_visitante_norm', 'goles_local_num', 'goles_visitante_num'
    ]
    cols_features = [
        'local_lastN_matches', 'local_lastN_wins', 'local_lastN_winrate',
        'local_lastN_goals_for', 'local_lastN_remates_puerta', 'local_lastN_possession_avg',
        'visitante_lastN_matches', 'visitante_lastN_wins',
        'visitante_lastN_winrate', 'visitante_lastN_goals_for', 'visitante_lastN_remates_puerta',
        'visitante_lastN_possession_avg',
        'total_goles_partido', 'diferencia_goles_partido', 'resultado_texto'
    ]
    df_out = pd.DataFrame(columns=out_cols + cols_features)
    for idx, row in df.iterrows():
        team_local = row['equipo_local_norm']
        team_visit = row['equipo_visitante_norm']
        prev_local = df.loc[((df['equipo_local_norm'] == team_local) | (df['equipo_visitante_norm'] == team_local)) & (df.index > idx)].head(N)
        prev_visit = df.loc[((df['equipo_visitante_norm'] == team_visit) | (df['equipo_local_norm'] == team_visit)) & (df.index > idx)].head(N)
        metrics_local = _calc_metrics(prev_local, team_local)
        metrics_visit = _calc_metrics(prev_visit, team_visit)
        if int(row.get('goles_local_num', 0)) > int(row.get('goles_visitante_num', 0)):
            resultado_texto = 'gana_local'
        elif int(row.get('goles_local_num', 0)) < int(row.get('goles_visitante_num', 0)):
            resultado_texto = 'gana_visitante'
        else:
            resultado_texto = 'empate'
        fila = {
            'fecha_completa': row.get('fecha_completa'),
            'equipo_local_norm': team_local,
            'equipo_visitante_norm': team_visit,
            'goles_local_num': int(row.get('goles_local_num', 0)),
            'goles_visitante_num': int(row.get('goles_visitante_num', 0)),
            'local_lastN_matches': metrics_local['matches'],
            'local_lastN_wins': metrics_local['wins'],
            'local_lastN_winrate': round(metrics_local['winrate'], 3),
            'local_lastN_goals_for': metrics_local['goals_for'],
            'local_lastN_remates_puerta': metrics_local['remates_puerta'],
            'local_lastN_possession_avg': round(metrics_local['possession_avg'], 3),
            'visitante_lastN_matches': metrics_visit['matches'],
            'visitante_lastN_wins': metrics_visit['wins'],
            'visitante_lastN_winrate': round(metrics_visit['winrate'], 3),
            'visitante_lastN_goals_for': metrics_visit['goals_for'],
            'visitante_lastN_remates_puerta': metrics_visit['remates_puerta'],
            'visitante_lastN_possession_avg': round(metrics_visit['possession_avg'], 3),
            'total_goles_partido': int(row.get('goles_local_num', 0)) + int(row.get('goles_visitante_num', 0)),
            'diferencia_goles_partido': int(row.get('goles_local_num', 0)) - int(row.get('goles_visitante_num', 0)),
            'resultado_texto': resultado_texto
        }
        df_out = pd.concat([df_out, pd.DataFrame([fila])], ignore_index=True)
    df_out = df_out.apply(pd.to_numeric, errors='ignore')
    df_out['fecha_completa'] = pd.to_datetime(df_out['fecha_completa'])
    df_out.dropna(inplace=True)
    return df_out

st.set_page_config(page_title="Análisis y Modelo", layout="wide")
alt.data_transformers.disable_max_rows()

# Estilos: hacer que los controles ocupen solo su contenido (no todo el ancho)
st.markdown(
        """
        <style>
        /* Selectbox: inline y ancho ajustado al contenido */
        div[data-testid="stSelectbox"] { display: inline-block; margin-right: 12px; }
        div[data-testid="stSelectbox"] > div { width: fit-content; min-width: 220px; }

        /* DateInput: inline y compacto */
        div[data-testid="stDateInput"] { display: inline-block; margin-right: 12px; }
        div[data-testid="stDateInput"] > div { width: fit-content; min-width: 220px; }
        div[data-testid="stDateInput"] input { width: 160px; }

        /* NumberInput: inline y compacto */
        div[data-testid="stNumberInput"] { display: inline-block; margin-right: 12px; }
        div[data-testid="stNumberInput"] > div { width: fit-content; min-width: 180px; }
        div[data-testid="stNumberInput"] input { width: 110px; text-align: center; }
        /* Responsivo: en pantallas chicas reducir mínimo */
        @media (max-width: 768px) {
            div[data-testid="stSelectbox"] > div,
            div[data-testid="stDateInput"] > div,
            div[data-testid="stNumberInput"] > div { min-width: 160px; }
            div[data-testid="stDateInput"] input { width: 140px; }
            div[data-testid="stNumberInput"] input { width: 100px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
)

# =========================
# DATA
# =========================
@st.cache_data
def load_data(path):
    # Cargar datos crudos y generar features para exploración
    df_raw = pd.read_csv(path, dayfirst=True)
    df_feat = generar_df_features(df_raw, fecha_referencia=None)
    # compatibilidad: charts usan 'date'
    df_feat['date'] = df_feat['fecha_completa']
    # features derivados para gráficos
    if {'local_lastN_winrate','visitante_lastN_winrate'}.issubset(df_feat.columns):
        df_feat['delta_winrate'] = df_feat['local_lastN_winrate'] - df_feat['visitante_lastN_winrate']
    if {'local_lastN_possession_avg','visitante_lastN_possession_avg'}.issubset(df_feat.columns):
        df_feat['delta_possession'] = df_feat['local_lastN_possession_avg'] - df_feat['visitante_lastN_possession_avg']
    return df_feat


# Nota: ya no se necesita normalizar nombres de equipos — el CSV viene correcto

# Placeholder for df; will cargar según la elección de modelo más abajo
df = None

st.title("Proyecto de Futbol – Visualización e Integración")
st.caption("Altair + Streamlit • Exploración y prueba de modelo")

# Selección global de modelo/dataset (afecta los CSVs que se cargan)
MODEL_CSV_MAP = {
    'Random Forest': 'datos_procesados_modelo.csv',
    'Regresion': 'datos_procesados_modelo.csv',
    'Ridge': 'datos_procesados_modelo.csv'
}
MODEL_FILE_MAP = {
    'Random Forest': 'modelo_final_random.pkl',
    'Regresion': 'modelo_final_regresion.pkl',
    'Ridge': 'modelo_final_ridge.pkl'
}

# Cargar dataset por defecto (para la exploración). El selector de modelo/dataset
# fue movido a la pestaña 'Probar modelo' para no interferir la exploración.
DEFAULT_MODEL_CHOICE = 'Random Forest'
DATA_PATH = os.path.join('data', MODEL_CSV_MAP[DEFAULT_MODEL_CHOICE])
df = load_data(DATA_PATH)

# conservar una copia cruda para cálculos dinámicos en predicción
if 'df_source' not in st.session_state:
    try:
        st.session_state['df_source'] = pd.read_csv(DATA_PATH, dayfirst=True)
    except Exception:
        st.session_state['df_source'] = None

# asegurar existencia de session_state para el modelo
if 'model' not in st.session_state:
    st.session_state['model'] = None

# valor por defecto para cantidad de partidos a mostrar en las gráficas
if 'matches_limit' not in st.session_state:
    st.session_state['matches_limit'] = 30

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
def load_model_from_drive(url_or_id: str, out_path='modelo_final_random.pkl'):
    import gdown
    file_id = url_or_id if re.fullmatch(r'[A-Za-z0-9_-]{25,}', url_or_id) else _drive_id_from_url(url_or_id)
    if not file_id:
        raise ValueError("URL/ID de Drive inválida.")
    gdown.download(id=file_id, output=out_path, quiet=True)
    return joblib.load(out_path)

## (helpers y funciones de features ya están definidas arriba)

# Util: obtener el estimador final (desenvolver Pipeline/GridSearchCV, etc.)
def get_final_estimator(model):
    est = model
    # Desenrollar meta-estimadores comunes
    for attr in ("best_estimator_", "estimator_"):
        if hasattr(est, attr):
            try:
                est = getattr(est, attr)
            except Exception:
                pass
    # Desenrollar Pipeline (sklearn o imblearn)
    try:
        if hasattr(est, 'steps') and est.steps:
            est = est.steps[-1][1]
        elif hasattr(est, 'named_steps') and est.named_steps:
            est = list(est.named_steps.values())[-1]
    except Exception:
        pass
    # Desenrollar base_estimator si existe
    if hasattr(est, 'base_estimator_'):
        try:
            est = getattr(est, 'base_estimator_')
        except Exception:
            pass
    return est

# Util: intentar obtener los nombres de features transformados (p. ej. ColumnTransformer/OneHotEncoder)
def get_transformed_feature_names(model, input_feature_names=None):
    # 1) Si el modelo expone get_feature_names_out, intentarlo directamente
    try:
        if hasattr(model, 'get_feature_names_out'):
            try:
                return list(model.get_feature_names_out())
            except TypeError:
                # Algunos requieren input_feature_names
                if input_feature_names is not None:
                    try:
                        return list(model.get_feature_names_out(input_feature_names))
                    except Exception:
                        pass
    except Exception:
        pass

    # 2) Si es un Pipeline, buscar el último transformador que exponga get_feature_names_out
    steps = []
    try:
        if hasattr(model, 'steps') and model.steps:
            steps = [s[1] for s in model.steps]
        elif hasattr(model, 'named_steps') and model.named_steps:
            steps = list(model.named_steps.values())
    except Exception:
        steps = []

    for step in reversed(steps):
        if hasattr(step, 'get_feature_names_out'):
            try:
                return list(step.get_feature_names_out())
            except TypeError:
                if input_feature_names is not None:
                    try:
                        return list(step.get_feature_names_out(input_feature_names))
                    except Exception:
                        pass
        # Si el step es a su vez un Pipeline anidado, intentar recursivamente
        try:
            nested = get_transformed_feature_names(step, input_feature_names)
            if nested:
                return list(nested)
        except Exception:
            pass

    # 3) Buscar atributos comunes para preprocesadores
    for attr in ('preprocessor', 'transformer', 'columntransformer'):
        tr = getattr(model, attr, None)
        if tr is not None and hasattr(tr, 'get_feature_names_out'):
            try:
                return list(tr.get_feature_names_out())
            except Exception:
                pass

    return None

# Util: detectar nombres genéricos tipo x0, f1, etc.
def _is_generic_names(names):
    try:
        return all(re.match(r'^[xf]\d+$', str(n)) for n in names)
    except Exception:
        return False

# Util: agregar importancias por feature original cuando el modelo expande columnas (OneHot/transformers)
def aggregate_importance_to_original(weights, names, original_feats):
    # Retorna lista de pesos alineados a original_feats sumando contribuciones de columnas transformadas
    try:
        orig_list = list(original_feats)
        orig_low = [(f, f.lower()) for f in orig_list]
        # ordenar por largo desc para preferir match de nombres largos
        orig_low.sort(key=lambda x: -len(x[1]))
        agg = {f: 0.0 for f in orig_list}
        for w, n in zip(weights, names):
            s = str(n).lower()
            matched = None
            # match directo por substring
            for f, f_low in orig_low:
                if f_low and f_low in s:
                    matched = f
                    break
            # intento adicional por tokens separados por '__' o '_'
            if matched is None:
                tokens = re.split(r'__|_', s)
                for f, f_low in orig_low:
                    if f_low in tokens or any(tok == f_low for tok in tokens):
                        matched = f
                        break
            if matched is not None:
                agg[matched] += float(w)
        return [agg[f] for f in orig_list]
    except Exception:
        return None

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

# ========= Features para predicción por ventana temporal (6 meses) =========
def build_match_features_by_date_window(df_raw, equipo_local, equipo_visitante, fecha_referencia, months_window=6):
    if equipo_local is None or equipo_visitante is None:
        return {}
    df2 = unir_fecha_anio(df_raw.copy())
    # asegurar columnas necesarias
    if 'equipo_local_norm' not in df2.columns and 'equipo_local' in df2.columns:
        df2['equipo_local_norm'] = df2['equipo_local'].apply(normalizar_texto)
        df2['equipo_visitante_norm'] = df2['equipo_visitante'].apply(normalizar_texto)
    if 'goles_local_num' not in df2.columns and 'resultado_local' in df2.columns:
        df2['goles_local_num'] = df2['resultado_local'].apply(to_int_safe)
        df2['goles_visitante_num'] = df2['resultado_visitante'].apply(to_int_safe)
    for col, col_num in [('remates_total_local', 'remates_total_local_num'),
                         ('remates_total_visitante', 'remates_total_visitante_num'),
                         ('remates_puerta_local', 'remates_puerta_local_num'),
                         ('remates_puerta_visitante', 'remates_puerta_visitante_num')]:
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

    local_norm = normalizar_texto(equipo_local)
    visit_norm = normalizar_texto(equipo_visitante)
    fecha_ref = pd.to_datetime(fecha_referencia)
    fecha_ini = fecha_ref - pd.DateOffset(months=months_window)
    rango = df2[(df2['fecha_completa'] <= fecha_ref) & (df2['fecha_completa'] >= fecha_ini)]

    prev_local = rango[(rango['equipo_local_norm'] == local_norm) | (rango['equipo_visitante_norm'] == local_norm)]
    prev_visit = rango[(rango['equipo_local_norm'] == visit_norm) | (rango['equipo_visitante_norm'] == visit_norm)]

    metrics_local = calc_metrics(prev_local, local_norm)
    metrics_visit = calc_metrics(prev_visit, visit_norm)

    return {
        'local_lastN_matches': metrics_local['matches'],
        'local_lastN_wins': metrics_local['wins'],
        'local_lastN_winrate': round(metrics_local['winrate'], 3),
        'local_lastN_goals_for': metrics_local['goals_for'],
        'local_lastN_remates_puerta': metrics_local['remates_puerta'],
        'local_lastN_possession_avg': round(metrics_local['possession_avg'], 3),
        'visitante_lastN_matches': metrics_visit['matches'],
        'visitante_lastN_wins': metrics_visit['wins'],
        'visitante_lastN_winrate': round(metrics_visit['winrate'], 3),
        'visitante_lastN_goals_for': metrics_visit['goals_for'],
        'visitante_lastN_remates_puerta': metrics_visit['remates_puerta'],
        'visitante_lastN_possession_avg': round(metrics_visit['possession_avg'], 3)
    }


## (calc_metrics y build_match_features ya están definidos anteriormente; se elimina duplicado)
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
        sel_local = c1.selectbox("Equipo local", equipos_all)
        visit_options = [e for e in equipos_all if e != sel_local]
        if visit_options:
            sel_visit = c2.selectbox("Equipo visitante", visit_options)
        else:
            # Caso raro: solo hay un equipo en la lista
            sel_visit = c2.selectbox("Equipo visitante", ["(No hay otro equipo disponible)"])
            st.warning("No hay otro equipo distinto disponible para seleccionar como visitante.")
    else:
        # fallback a text inputs (validación se hará antes de predecir)
        sel_local = c1.text_input("Equipo local")
        sel_visit = c2.text_input("Equipo visitante")

    # (Se reubica la fecha de referencia junto al botón de predecir más abajo)

    # Valores base del usuario (equipos seleccionados)
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

    # Se removieron las opciones avanzadas para simplificar la interfaz

    # =========================
    # ESTADÍSTICAS PARAMETRIZABLES DE LOS EQUIPOS SELECCIONADOS
    # =========================
    st.markdown("### Estadísticas de los equipos seleccionados")
    st.caption("Se usan las columnas de features ya preprocesadas (df) para generar las métricas.")

    def _team_metric_df(df_src: pd.DataFrame, team_norm: str, metric_key: str, side_filter: str, year_filter: str, max_rows=None):
        """Devuelve un DataFrame con una columna 'metric_value' para el equipo dado,
        tomando la columna correcta según juegue de local o visitante. Permite filtrar por lado.
        metric_key puede ser: winrate, possession, efficiency, goals_for, shots_on_target, matches.
        """
        if not team_norm:
            return pd.DataFrame()

        # Seleccionar filas donde participa el equipo
        mask_local = df_src['equipo_local_norm'] == team_norm
        mask_visit = df_src['equipo_visitante_norm'] == team_norm

        if side_filter == "Solo local":
            df_team = df_src[mask_local].copy()
        elif side_filter == "Solo visitante":
            df_team = df_src[mask_visit].copy()
        else:
            df_team = df_src[mask_local | mask_visit].copy()

        if df_team.empty:
            return pd.DataFrame()

    # Indicar lado
        df_team['home_away'] = np.where(df_team['equipo_local_norm'] == team_norm, 'Local', 'Visitante')

        # Seleccionar columna de métrica según lado
        if metric_key == 'winrate':
            df_team['metric_value'] = np.where(
                df_team['home_away'] == 'Local',
                pd.to_numeric(df_team.get('local_lastN_winrate'), errors='coerce'),
                pd.to_numeric(df_team.get('visitante_lastN_winrate'), errors='coerce')
            )
            y_title = 'Winrate (0-1)'
            scale = alt.Scale(domain=[0, 1])
            fmt = '.2f'
        elif metric_key == 'possession':
            df_team['metric_value'] = np.where(
                df_team['home_away'] == 'Local',
                pd.to_numeric(df_team.get('local_lastN_possession_avg'), errors='coerce'),
                pd.to_numeric(df_team.get('visitante_lastN_possession_avg'), errors='coerce')
            )
            # Si parece estar en 0-1, convertir a porcentaje
            max_val = df_team['metric_value'].max()
            if pd.notna(max_val) and max_val <= 1.01:
                df_team['metric_value'] = df_team['metric_value'] * 100.0
            y_title = 'Posesión (%)'
            scale = alt.Scale(domain=[0, 100])
            fmt = '.0f'
        elif metric_key == 'efficiency':
            # goles a favor / remates a puerta, acotado a 1
            local_eff = (pd.to_numeric(df_team.get('local_lastN_goals_for'), errors='coerce') /
                         pd.to_numeric(df_team.get('local_lastN_remates_puerta'), errors='coerce').replace(0, np.nan))
            visit_eff = (pd.to_numeric(df_team.get('visitante_lastN_goals_for'), errors='coerce') /
                         pd.to_numeric(df_team.get('visitante_lastN_remates_puerta'), errors='coerce').replace(0, np.nan))
            df_team['metric_value'] = np.where(df_team['home_away'] == 'Local', local_eff, visit_eff)
            df_team['metric_value'] = df_team['metric_value'].clip(upper=1)
            y_title = 'Eficiencia (goles/remates a puerta)'
            scale = alt.Scale(domain=[0, 1])
            fmt = '.2f'
        elif metric_key == 'goals_for':
            df_team['metric_value'] = np.where(
                df_team['home_away'] == 'Local',
                pd.to_numeric(df_team.get('local_lastN_goals_for'), errors='coerce'),
                pd.to_numeric(df_team.get('visitante_lastN_goals_for'), errors='coerce')
            )
            y_title = 'Goles a favor'
            scale = alt.Scale(zero=True)
            fmt = '.0f'
        elif metric_key == 'shots_on_target':
            df_team['metric_value'] = np.where(
                df_team['home_away'] == 'Local',
                pd.to_numeric(df_team.get('local_lastN_remates_puerta'), errors='coerce'),
                pd.to_numeric(df_team.get('visitante_lastN_remates_puerta'), errors='coerce')
            )
            y_title = 'Remates a puerta'
            scale = alt.Scale(zero=True)
            fmt = '.0f'
        else:
            # métrica no soportada
            return pd.DataFrame(), '', None, '.0f'

        # Orden temporal y columnas útiles
        df_team = df_team.dropna(subset=['metric_value'])
        if 'date' not in df_team.columns and 'fecha_completa' in df_team.columns:
            df_team['date'] = pd.to_datetime(df_team['fecha_completa'])
        # Filtro por año si corresponde
        if year_filter and isinstance(year_filter, str) and year_filter != "Todos":
            try:
                yr = int(year_filter)
                df_team = df_team[pd.to_datetime(df_team['date']).dt.year == yr]
            except Exception:
                pass
        df_team = df_team.sort_values('date')
        # limitar cantidad de partidos mostrados (los más recientes)
        try:
            if max_rows is not None and int(max_rows) > 0:
                df_team = df_team.tail(int(max_rows))
        except Exception:
            pass
        df_team['oponente'] = np.where(df_team['home_away'] == 'Local', df_team['equipo_visitante_norm'], df_team['equipo_local_norm'])
        return df_team, y_title, scale, fmt

    # Controles de parametrización
    # Construir lista de años disponibles desde el df preprocesado
    if 'date' in df.columns:
        try:
            years = sorted(pd.to_datetime(df['date']).dropna().dt.year.unique().tolist())
        except Exception:
            years = []
    elif 'fecha_completa' in df.columns:
        try:
            years = sorted(pd.to_datetime(df['fecha_completa']).dropna().dt.year.unique().tolist())
        except Exception:
            years = []
    else:
        years = []

    m1, m2, m3, m4, m5 = st.columns([1.2, 1.1, 1.0, 1.0, 1.2])
    metric_label = m1.selectbox(
        "Métrica",
        [
            "Winrate",
            "Posesión",
            "Eficiencia",
            "Goles a favor",
            "Remates a puerta"
        ],
        index=0
    )
    side_choice = m2.selectbox("Lado", ["Ambos", "Solo local", "Solo visitante"], index=0)
    year_options = ["Todos"] + [str(y) for y in years]
    year_choice = m3.selectbox("Año", year_options, index=0)
    chart_type = m4.selectbox("Tipo", ["Línea", "Puntos"], index=0)

    # Nuevo: cantidad de partidos a mostrar (misma fila que los demás parámetros)
    matches_limit_input = m5.number_input(
        "Partidos a mostrar",
        min_value=5,
        max_value=200,
        step=5,
        key="matches_limit",
        help="Limita cuántos partidos recientes se grafican en las tarjetas superiores"
    )

    metric_map = {
        "Winrate": "winrate",
        "Posesión": "possession",
        "Eficiencia": "efficiency",
        "Goles a favor": "goals_for",
        "Remates a puerta": "shots_on_target"
    }
    metric_key = metric_map[metric_label]

    c_left, c_right = st.columns(2)
    matches_limit = int(st.session_state.get('matches_limit', matches_limit_input))
    if isinstance(sel_local, str) and sel_local and isinstance(sel_visit, str) and sel_visit and teams_different:
        # Panel izquierdo: equipo local seleccionado
        ldf, y_title_l, scale_l, fmt_l = _team_metric_df(df, sel_local, metric_key, side_choice, year_choice, matches_limit)
        if not ldf.empty:
            c_left.markdown(f"**{metric_label} — {sel_local}**")
            if chart_type == 'Línea':
                base_l = alt.Chart(ldf).mark_line()
            else:
                base_l = alt.Chart(ldf).mark_point(size=60, opacity=0.8)
            chart_l = (base_l
                       .encode(
                           x=alt.X('date:T', title='Fecha'),
                           y=alt.Y('metric_value:Q', title=y_title_l, scale=scale_l, axis=alt.Axis(format=fmt_l)),
                           color=alt.Color('home_away:N', title='Condición'),
                           tooltip=['date:T', 'home_away:N', 'oponente:N', alt.Tooltip('metric_value:Q', title=metric_label, format=fmt_l)]
                       )
                       .properties(height=260))
            c_left.altair_chart(chart_l, use_container_width=True)
        else:
            c_left.info("Sin datos preprocesados para el equipo local seleccionado con el filtro actual.")

        # Panel derecho: equipo visitante seleccionado
        rdf, y_title_r, scale_r, fmt_r = _team_metric_df(df, sel_visit, metric_key, side_choice, year_choice, matches_limit)
        if not rdf.empty:
            c_right.markdown(f"**{metric_label} — {sel_visit}**")
            if chart_type == 'Línea':
                base_r = alt.Chart(rdf).mark_line()
            else:
                base_r = alt.Chart(rdf).mark_point(size=60, opacity=0.8)
            chart_r = (base_r
                       .encode(
                           x=alt.X('date:T', title='Fecha'),
                           y=alt.Y('metric_value:Q', title=y_title_r, scale=scale_r, axis=alt.Axis(format=fmt_r)),
                           color=alt.Color('home_away:N', title='Condición'),
                           tooltip=['date:T', 'home_away:N', 'oponente:N', alt.Tooltip('metric_value:Q', title=metric_label, format=fmt_r)]
                       )
                       .properties(height=260))
            c_right.altair_chart(chart_r, use_container_width=True)
        else:
            c_right.info("Sin datos preprocesados para el equipo visitante seleccionado con el filtro actual.")
    else:
        st.info("Seleccioná dos equipos distintos para ver sus estadísticas.")

    # --- Fecha de referencia + cantidad de partidos (misma fila), botón debajo ---
    pred_row_l, pred_row_r = st.columns([1.0, 0.6])
    with pred_row_l:
        # Fecha de referencia (6 meses hacia atrás), limitada entre 2022-01-01 y 2025-09-28
        min_d = datetime.date(2022, 1, 1)
        max_d = datetime.date(2025, 9, 28)
        selected_date = st.date_input(
            "Fecha de referencia (historial 6 meses hacia atrás)",
            value=max_d,
            min_value=min_d,
            max_value=max_d,
            key="pred_ref_date"
        )
    with pred_row_r:
        # (Reservado para futuros controles si los necesitás)
        pass
    predict_clicked = st.button("Predecir", use_container_width=False)

    if predict_clicked:
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

            # intentar construir features históricas basadas en ventana de 6 meses desde la fecha elegida
            try:
                df_source = st.session_state.get('df_source')
                if df_source is None:
                    df_source = pd.read_csv(DATA_PATH, dayfirst=True)
                computed_features = build_match_features_by_date_window(df_source, sel_local, sel_visit, pd.to_datetime(selected_date), months_window=6)
            except Exception as e:
                computed_features = {}
                st.warning(f"No pude generar features históricas dinámicas (6 meses): {e}")

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
            
            # Se elimina la visualización de importancia de features a pedido del usuario.
        except Exception as e:
            st.error(f"Error al predecir: {e}")

# =========================
# TAB 3: ACERCA DE
# =========================
with tab3:
    st.markdown("""
**Datos**: `data/datos_procesados_modelo.csv`  
**Visualizaciones**: Altair (interactivas, comparables, y con facetado)  
**Modelo**: se intenta cargar automáticamente el archivo local asociado al tipo de modelo (p. ej. `modelo_final_random.pkl`). Si no está presente, colocá el .pkl en `models/` o en la raíz del proyecto.

> Consejo: fijá en `requirements.txt` la **misma versión de scikit-learn** e **imbalanced-learn** con la que entrenaste el modelo para evitar incompatibilidades al cargar el .pkl.
""")

from pathlib import Path
import textwrap
import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# =========================
# 0) CONFIG + CSS
# =========================
st.set_page_config(
    page_title="TFM Abel Mora Vázquez",
    page_icon="🏙️",
    layout="wide"
)

st.markdown("""
<style>
.block-container{
  padding-top: 2rem !important;     
  padding-bottom: 1rem !important;
}
.stTabs [data-baseweb="tab"]{
  white-space: normal !important;
  word-break: break-word !important;
  height: auto !important;
  padding: 8px 12px !important;
  line-height: 1.1 !important;
}
.stTabs [data-baseweb="tab"] *{
  font-size: 19px !important;
  font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data_app"
MODEL_PATH = "models/catboost_model.cbm"

# =========================
# 1) LOADERS (CACHE)
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    metrics = pd.read_parquet(f"{DATA_DIR}/metrics_app.parquet")
    perfil  = pd.read_parquet(f"{DATA_DIR}/perfil_app.parquet")
    geo     = pd.read_parquet(f"{DATA_DIR}/geo_app.parquet")
    macro   = pd.read_parquet(f"{DATA_DIR}/macro_valor_app.parquet")
    viv     = pd.read_parquet(f"{DATA_DIR}/viviendas_app.parquet")
    imp     = pd.read_parquet(f"{DATA_DIR}/importancia_app.parquet")
    rmse    = pd.read_parquet(f"{DATA_DIR}/rmse_app.parquet")
    compra   = pd.read_parquet(f"{DATA_DIR}/compra_app.parquet")
    alquiler = pd.read_parquet(f"{DATA_DIR}/alquiler_app.parquet")
    hab      = pd.read_parquet(f"{DATA_DIR}/habitaciones_app.parquet")

    return metrics, perfil, geo, macro, viv, imp, rmse, compra, alquiler, hab

@st.cache_resource(show_spinner=False)
def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model

metrics, perfil, geo, macro, viv, imp, rmse, compra, alquiler, hab = load_data()
model = load_model()

# =========================
# 2) HELPERS
# =========================
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def cap_pois(df, max_per_service=250, seed=42):
    if df is None or df.empty:
        return df
    out = []
    for (district, service), g in df.groupby(["district", "service"], dropna=False):
        if len(g) > max_per_service:
            g = g.sample(max_per_service, random_state=seed)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df

def eur(x):
    try:
        return f"{float(x):,.0f} €".replace(",", ".")
    except Exception:
        return "—"

def m2_fmt(x):
    try:
        return f"{float(x):,.0f} m²".replace(",", ".")
    except Exception:
        return "—"

def wrap_label(text, width=12):
    return "\n".join(textwrap.wrap(str(text), width=width))

def minmax_norm(s: pd.Series, floor=0.18, invert=False) -> pd.Series:
    """
    Normaliza a [floor, 1]. Si invert=True, hace que "más alto sea peor"
    pase a "más alto sea mejor", sin colapsar en 0.
    """
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(), s.max()

    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        
        return pd.Series([0.5] * len(s), index=s.index)

    x = (s - mn) / (mx - mn)  # [0,1]

    if invert:
        x = 1 - x  # invertimos en [0,1]
    # aplicamos suelo para evitar 0 "invisible"
    x = floor + (1 - floor) * x
    return x

def radar_plot(labels, series_dict, title, figsize=(4.8, 4.2), legend=True):
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for name, vals in series_dict.items():
        vals = list(vals) + [vals[0]]
        ax.plot(angles, vals, linewidth=2, label=name)
        ax.fill(angles, vals, alpha=0.10)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    pretty_labels = [wrap_label(l, 10) for l in labels]  # antes 12
    ax.set_thetagrids(np.degrees(angles[:-1]), pretty_labels)
    ax.tick_params(axis='x', labelsize=9, pad=8)  # pad separa etiquetas del círculo
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(20)

    ax.set_title(title, pad=8, fontsize=12, fontweight="bold", y=1.20)

    if legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.10, 0.50),  # a la derecha, centrada
            frameon=True,
            fontsize=9
        )

    fig.subplots_adjust(left=0.06, right=0.78, top=0.74, bottom=0.06)

    return fig


def plot_macro_scatter(df_macro, figsize=(6.8, 4.4)):
    fig, ax = plt.subplots(figsize=figsize)

    clusters = sorted(df_macro["Cluster"].dropna().unique())
    for c in clusters:
        d = df_macro[df_macro["Cluster"] == c]
        ax.scatter(d["price_m2_idealista"], d["satisfaccion_index"], label=f"Cluster {c}", s=95)

    x_med = np.median(df_macro["price_m2_idealista"])
    y_med = np.median(df_macro["satisfaccion_index"])
    ax.axvline(x_med, linestyle="--", alpha=0.6)
    ax.axhline(y_med, linestyle="--", alpha=0.6)

    ax.set_title("Valor percibido vs valor real por distrito", fontweight="bold")
    ax.set_xlabel("Precio observado por m² (Idealista) [€/m²]")
    ax.set_ylabel("Índice de satisfacción")
    ax.legend()
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig

def listing_filters(df, modo_label, default_max_from_quantile=True):
    """
    Devuelve (df_filtrado, config) con filtros comunes y ordenación.
    """
    if df is None or df.empty:
        return df, {}

    df = df.copy()

    # Normalizar columnas
    df = to_num(df, ["price", "size", "rooms", "bathrooms", "tenantNumber"])

    if "district" in df.columns:
        df = df.dropna(subset=["district"])
    if "price" in df.columns:
        df = df.dropna(subset=["price"])

    c1, c2, c3, c4 = st.columns([1.1, 1.0, 1.0, 1.1])

    with c1:
        dist = st.selectbox(
            f"Distrito ({modo_label}):",
            sorted(df["district"].dropna().unique()),
            key=f"dist_{modo_label}"
        )
    df = df[df["district"] == dist].copy()

    # Defaults
    if "price" in df.columns and len(df) > 10 and default_max_from_quantile:
        suggested_max = float(df["price"].quantile(0.95))
    else:
        suggested_max = float(df["price"].max()) if "price" in df.columns and not df.empty else 0.0

    with c2:
        min_price = st.number_input("Precio mín.", value=0.0, step=50.0, key=f"minp_{modo_label}")
    with c3:
        max_price = st.number_input("Precio máx.", value=float(suggested_max), step=50.0, key=f"maxp_{modo_label}")
    with c4:
        sort_mode = st.selectbox(
            "Ordenar por",
            ["Precio ↑", "Precio ↓", "m² ↑", "m² ↓", "€/m² ↑", "€/m² ↓"],
            index=0,
            key=f"sort_{modo_label}"
        )

    if "price" in df.columns:
        df = df[(df["price"] >= min_price) & (df["price"] <= max_price)].copy()

    # Filtros
    extra_cols = st.columns(3)
    with extra_cols[0]:
        min_m2 = st.number_input("m² mín.", value=0.0, step=5.0, key=f"minm2_{modo_label}")
    with extra_cols[1]:
        min_rooms = st.selectbox("Hab. mín.", [0,1,2,3,4,5], index=0, key=f"minr_{modo_label}")
    with extra_cols[2]:
        min_baths = st.selectbox("Baños mín.", [0,1,2,3,4], index=0, key=f"minb_{modo_label}")

    if "size" in df.columns:
        df = df[df["size"] >= min_m2]
    if "rooms" in df.columns:
        df = df[df["rooms"].fillna(0) >= min_rooms]
    if "bathrooms" in df.columns:
        df = df[df["bathrooms"].fillna(0) >= min_baths]

    # €/m²
    if "price" in df.columns and "size" in df.columns:
        df["eur_m2"] = df["price"] / df["size"].replace(0, np.nan)

    # Ordenación
    def safe_sort(col, ascending=True):
        if col in df.columns:
            return df.sort_values(col, ascending=ascending)
        return df

    if sort_mode == "Precio ↑":
        df = safe_sort("price", True)
    elif sort_mode == "Precio ↓":
        df = safe_sort("price", False)
    elif sort_mode == "m² ↑":
        df = safe_sort("size", True)
    elif sort_mode == "m² ↓":
        df = safe_sort("size", False)
    elif sort_mode == "€/m² ↑":
        df = safe_sort("eur_m2", True)
    elif sort_mode == "€/m² ↓":
        df = safe_sort("eur_m2", False)

    config = {
        "district": dist,
        "min_price": min_price,
        "max_price": max_price,
        "sort_mode": sort_mode,
        "min_m2": min_m2,
        "min_rooms": min_rooms,
        "min_baths": min_baths
    }
    return df, config

def render_listing_grid_paginated(df, title, cols=4, key_prefix="grid"):
    """
    - Paginación + selector de nº por página.
    """
    st.subheader(title)

    if df is None or df.empty:
        st.info("No hay resultados para mostrar con estos filtros.")
        return

    df = df.copy()

    if "url" in df.columns:
        df = df.dropna(subset=["url"])
    if "price" in df.columns:
        df = df.dropna(subset=["price"])

    # Controles de paginación
    top = st.columns([1.0, 1.0, 2.0])
    with top[0]:
        per_page = st.selectbox("Anuncios por página", [8, 12, 16, 20, 24], index=1, key=f"{key_prefix}_perpage")
    with top[1]:
        total = len(df)
        n_pages = max(int(np.ceil(total / per_page)), 1)
        page = st.number_input("Página", min_value=1, max_value=n_pages, value=1, step=1, key=f"{key_prefix}_page")
    with top[2]:
        st.caption(f"Mostrando {per_page} por página · Total resultados: {len(df)}")

    start = (page - 1) * per_page
    end = start + per_page
    df_page = df.iloc[start:end].copy()

    rows = int(np.ceil(len(df_page) / cols))
    idx = 0

    for _ in range(rows):
        cc = st.columns(cols)
        for j in range(cols):
            if idx >= len(df_page):
                break
            r = df_page.iloc[idx]
            with cc[j]:
                thumb = r.get("thumbnail", None)
                if isinstance(thumb, str) and thumb.startswith("http"):
                    st.image(thumb, use_container_width=True)
                else:
                    st.caption("Sin imagen")

                price = eur(r.get("price", None))
                size  = m2_fmt(r.get("size", None))
                rooms = r.get("rooms", None)
                baths = r.get("bathrooms", None)

                meta = [f"**{price}**", size]
                if rooms is not None and not pd.isna(rooms):
                    try:
                        meta.append(f"{int(rooms)} hab.")
                    except Exception:
                        pass
                if baths is not None and not pd.isna(baths):
                    try:
                        meta.append(f"{int(baths)} baños")
                    except Exception:
                        pass

                st.write(" · ".join(meta))

                url = r.get("url", None)
                if isinstance(url, str) and url.startswith("http"):
                    st.link_button("🔗 Ver anuncio", url)
                else:
                    st.caption("Sin URL")

            idx += 1
def apply_sidebar_filters(df, modo, presupuesto, m2):
    """
    Aplica filtros globales del sidebar.
    - presupuesto: umbral máximo de price
    - m2: umbral mínimo de size
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Precio
    if "price" in out.columns:
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
        out = out.dropna(subset=["price"])
        out = out[out["price"] <= float(presupuesto)]

    # Superficie
    if "size" in out.columns:
        out["size"] = pd.to_numeric(out["size"], errors="coerce")
        out = out.dropna(subset=["size"])
        out = out[out["size"] >= float(m2)]

    return out
# =========================
# 3) SIDEBAR
# =========================
with st.sidebar:
    st.header("🎚️ Perfil de Vivienda")

    with st.form("perfil_form"):
        modo = st.selectbox("Operación", ["Compra", "Alquiler", "Habitaciones"], index=0)
        m2 = st.slider("Superficie deseada (m²)", 30, 200, 60)

        if modo == "Compra":
            presupuesto = st.number_input("Presupuesto Máximo (€)", value=350000, step=5000)
        elif modo == "Alquiler":
            presupuesto = st.number_input("Presupuesto Máximo (€/mes)", value=1200, step=50)
        else:
            presupuesto = st.number_input("Presupuesto Máximo (habitación €/mes)", value=700, step=25)

        st.divider()
        st.subheader("Prioridades del entorno")

        imp_seg = st.slider("Seguridad", 1, 10, 8)
        imp_lim = st.slider("Limpieza urbana", 1, 10, 7)
        imp_tra = st.slider("Transporte público", 1, 10, 7)
        imp_trq = st.slider("Tranquilidad (ruido y circulación)", 1, 10, 6)
        tiene_coche = st.checkbox("Tengo vehículo propio")

        aplicar = st.form_submit_button("Aplicar filtros")

# =========================
# 4) RECOMENDADOR
# =========================
col_p = "precio_m2_compra" if modo == "Compra" else "precio_m2_alquiler"
precio_m2_max = presupuesto / max(m2, 1)

df_result = metrics.copy()
df_result = to_num(df_result, ["precio_m2_compra", "precio_m2_alquiler"]).dropna(subset=["district"])

# Habitaciones:
if modo == "Habitaciones":
    df_result = df_result[df_result["precio_m2_alquiler"] <= precio_m2_max].copy()
else:
    df_result = df_result[df_result[col_p] <= precio_m2_max].copy()

if not df_result.empty:
    # Índices (0–10) basados en la encuesta 
    seguridad_idx = df_result["seguridad_barrio"]
    limpieza_idx  = (df_result["limpieza_calles"] + df_result["recogida_basura"]) / 2

    # Transporte público (percepción)
    cols_tp = [c for c in ["metro", "autobus", "tranvia", "bicing"] if c in df_result.columns]
    transporte_idx = df_result[cols_tp].mean(axis=1) if len(cols_tp) > 0 else 0

    # Tranquilidad: gestión en ruido + gestión en circulación
    tranquilidad_idx = (df_result["ruido"] + df_result["circulacion"]) / 2

    # Score: matching por mínimos (no compensatorio) 
    max_score = imp_seg + imp_lim + imp_tra + imp_trq + (5 if tiene_coche else 0)

    score = (
        np.minimum(seguridad_idx, imp_seg) +
        np.minimum(limpieza_idx, imp_lim) +
        np.minimum(transporte_idx, imp_tra) +
        np.minimum(tranquilidad_idx, imp_trq)
    )

    if tiene_coche:
        score = score + np.minimum(df_result["aparcamiento"], 5)

    df_result["Match %"] = (score / max_score) * 100
    df_result = df_result.sort_values("Match %", ascending=False)

# =========================
# 5) UI TABS (fusionados)
# =========================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Inicio",
    "🎯 Recomendador",
    "🗺️ Mapa y Servicios",
    "📡 Radares",
    "📌 Valor Percibido vs Real",
    "💎 Listados",
    "🤖 Modelos y Métricas",
    "🔮 Predicción de Precio",
    "ℹ️ Sobre la app"
])

# -------------------------
# TAB 0
# -------------------------
with tab0:
    st.title("🧠 Análisis del mercado inmobiliario en el área metropolitana de Barcelona")

    st.markdown("""
Elegir dónde vivir en Barcelona no es solo una cuestión de precio. La seguridad, la movilidad,
el acceso a servicios y la calidad de vida varían notablemente entre distritos.

Esta aplicación mediante datos sociodemográficos, de transporte urbano y modelos de *Machine Learning*
te ayuda a conocer el mercado inmobiliario de Barcelona de forma objetiva y basada en datos.
""")
    st.subheader("¿Qué puedes hacer aquí?")
    st.markdown("""
- **Recomendación de distritos** según tus prioridades personales.  
- **Explorar mapas y servicios urbanos** con datos reales por distrito. 
- **Comparar valor percibido y valor real** del mercado inmobiliario.  
- **Predecir el precio de una vivienda**
""")
    st.subheader("Desplázate por las pestañas de arriba. Pero antes, ¿Que valoran los vecinos? ")
    flourish_html = """
<div class="flourish-embed flourish-radar" data-src="visualisation/27078996"></div>
<script src="https://public.flourish.studio/resources/embed.js"></script>
"""
    components.html(
        flourish_html,
        height=1000,       
        scrolling=False   
    )

    with st.expander("📘 Nota metodológica y fuentes de datos"):
        st.markdown("""
**Origen de los datos**  
Datos sociodemográficos y de servicios urbanos procedentes del Open Data del Ajuntament de Barcelona.
Datos inmobiliarios obtenidos de portal inmobiliario Idealista mediante API. (Diciembre 2025).

**Metodología de predicción**  
Modelo de *Machine Learning* CatBoost entrenado con múltiples variables urbanas, económicas y estructurales
para estimar el precio de viviendas de forma orientativa.
""")

# -------------------------
# TAB 1
# -------------------------
with tab1:
    st.title("🏆 Distritos recomendados")
    with st.expander("ℹ️ ¿Cómo se calcula el Match %?"):
        st.markdown("""
- Cada prioridad (0–10) limita cuánto puede aportar esa dimensión.
- Para cada dimensión se calcula: **aporte = min(valor_del_distrito, tu_prioridad)**.
- El porcentaje final es la suma de aportes dividida por el máximo posible (tus prioridades + coche).
""")
    if df_result.empty:
        st.warning("No hay distritos que cumplan tu presupuesto. Sube presupuesto o reduce m².")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Distritos aptos", len(df_result))
        c2.metric("Afinidad máxima", f"{df_result['Match %'].max():.1f}%")
        if modo == "Habitaciones":
            c3.metric("Precio m² (proxy alquiler) medio", f"{df_result['precio_m2_alquiler'].mean():.1f}€")
        else:
            c3.metric("Precio m² medio", f"{int(df_result[col_p].mean())}€")

        cols_show = ["district", col_p if modo != "Habitaciones" else "precio_m2_alquiler",
                     "limpieza_calles", "metro", "aparcamiento", "Match %"]
        cols_show = [c for c in cols_show if c in df_result.columns]
        st.dataframe(df_result[cols_show].head(15), use_container_width=True)

        st.subheader("📋 Perfil rápido (niveles) del mejor distrito")
        best = df_result.iloc[0]["district"]
        rowp = perfil[perfil["district"] == best]
        if not rowp.empty:
            st.dataframe(rowp, use_container_width=True)
        else:
            st.info("No hay perfil_niveles para ese distrito.")
    # ✅ Debajo de TODO lo anterior (siempre)
    st.divider()
    st.subheader("🌳 Valoraciones y Calidad de Vida")

    flourish_hierarchy = """
<div class="flourish-embed flourish-hierarchy" data-src="visualisation/27190574"></div>
<script src="https://public.flourish.studio/resources/embed.js"></script>
"""
    components.html(flourish_hierarchy, height=1500, scrolling=False)

# -------------------------
# TAB 2: MAPA
# -------------------------
with tab2:
    st.title("🗺️ Mapa de servicios")
    st.caption("Filtra por distrito y por tipo de servicio. Se limita el nº de puntos por servicio para rendimiento.")

    distrito_sel = st.selectbox("Selecciona distrito:", sorted(metrics["district"].dropna().unique()))
    servicios = sorted(geo["service"].dropna().unique())
    servicios_sel = st.multiselect(
        "Servicios a mostrar:",
        options=servicios,
        default=[s for s in servicios if "Metro" in s or "Bicing" in s][:2]
    )

    geo_f = geo[(geo["district"] == distrito_sel) & (geo["service"].isin(servicios_sel))].copy()
    geo_f = geo_f.dropna(subset=["lat", "lon"])
    geo_f = cap_pois(geo_f, max_per_service=250)

    rowm = metrics[metrics["district"] == distrito_sel].iloc[0]
    center = [41.387, 2.17]
    if "lat" in metrics.columns and "lon" in metrics.columns:
        try:
            if float(rowm["lat"]) != 0 and float(rowm["lon"]) != 0:
                center = [float(rowm["lat"]), float(rowm["lon"])]
        except Exception:
            pass

    col_map, col_info = st.columns([2.05, 1])

    with col_map:
        m = folium.Map(location=center, zoom_start=13)
        cluster = MarkerCluster().add_to(m)

        for _, r in geo_f.iterrows():
            folium.Marker(
                location=[r["lat"], r["lon"]],
                tooltip=r["service"],
                popup=r.get("desc", r["service"])
            ).add_to(cluster)

        st_folium(m, width="100%", height=520)

    with col_info:
        st.subheader(f"📊 {distrito_sel}")
        if "ingresos_mensuales" in metrics.columns:
            st.metric("Ingresos medios", f"{rowm['ingresos_mensuales']:.0f}€")
        st.metric("€/m² compra", f"{rowm['precio_m2_compra']:.0f}")
        st.metric("€/m² alquiler", f"{rowm['precio_m2_alquiler']:.1f}")

        st.write("**Percepción (0-10):**")
        for k, label in [("limpieza_calles", "Limpieza"), ("seguridad_barrio", "Seguridad"), ("ruido", "Ruido")]:
            if k in metrics.columns:
                val = float(rowm[k])
                st.progress(min(val/10, 1.0), text=f"{label}: {val:.1f}")

    st.divider()
    st.markdown("### 🚦 Servicios de movilidad urbana por distrito")

    components.iframe(
    "https://public.flourish.studio/visualisation/27091950/embed",
    height=1200,
    scrolling=False
    )

# -------------------------
# TAB 3: RADARES (Servicios / Movilidad / Seguridad)
# -------------------------
with tab3:
    st.title("📡 Radares comparativos")
    st.caption("Comparación visual normalizada (0–1). Máximo 4 distritos para mejorar legibilidad.")

    radar_tipo = st.radio(
        "Tipo de radar:",
        ["Servicios", "Movilidad", "Seguridad (tasas)"],
        horizontal=True,
        key="radar_tipo_tab3"
    )

    if radar_tipo == "Servicios":
        radar_dims = [
            ("n_metro", "Metro"),
            ("n_bus", "Bus"),
            ("n_bicing", "Bicing"),
            ("n_aparcamiento", "Aparcamientos"),
            ("Parades Taxi", "Taxi"),
            ("Punts de recàrrega de vehicles elèctrics", "Recarga EV"),
        ]

    elif radar_tipo == "Movilidad":
        radar_dims = [
            ("Tasa_Trans_Publico", "Transp. público / km²"),
            ("Tasa_Trans_Privado", "Transp. privado / km²"),
            ("Tasa_Movilidad_Sup", "Movilidad total / km²"),
        ]

    else:  # Seguridad (tasas)
        radar_dims = [
            ("Tasa_C_Personas", "Delitos personas (x1000)"),
            ("Tasa_C_Patrimonio", "Delitos patrimonio (x1000)"),
            ("Tasa_C_Otros", "Otros delitos (x1000)"),
        ]

    # Filtra columnas
    radar_dims = [(c, lab) for c, lab in radar_dims if c in metrics.columns]
    cols = [c for c, _ in radar_dims]
    labels = [lab for _, lab in radar_dims]

    if len(cols) == 0:
        st.info("No hay columnas disponibles para este radar en `metrics`.")
    else:
        
        norm_df = metrics[["district"] + cols].copy()

        for c in cols:
            norm_df[c] = pd.to_numeric(norm_df[c], errors="coerce")

        norm_df[cols] = norm_df[cols].fillna(0)

        # Normalizar. Se añade un suelo para poder ver los que menos tienen ya que sino, se vería como 0.
        for c in cols:
            norm_df[c] = minmax_norm(norm_df[c], floor=0.18)

        selected = st.multiselect(
            "Selecciona distritos (máx 4):",
            options=sorted(norm_df["district"].dropna().unique()),
            default=sorted(norm_df["district"].dropna().unique())[:2],
            key=f"radar_distritos_{radar_tipo}")[:4]

        if len(selected) == 0:
            st.info("Selecciona al menos 1 distrito.")
        else:
            left, right = st.columns([1.25, 1])

            with left:
                series_dict = {}
                for d in selected:
                    row = norm_df[norm_df["district"] == d].iloc[0]
                    series_dict[d] = [float(row[c]) for c in cols]

                titulo = f"Comparativa (normalizado): {radar_tipo}"
                fig = radar_plot(labels, series_dict, titulo, figsize=(4.8, 4.2), legend=True)
                st.pyplot(fig, use_container_width=False)

                if radar_tipo == "Seguridad (tasas)":
                    st.caption(
                        "En este radar, valores más altos indican mayor incidencia delictiva "
                        "(tasas por cada 1.000 habitantes)."
                )

            with right:
                st.subheader("Valores reales (sin normalizar)")
                df_real = metrics[metrics["district"].isin(selected)][["district"] + cols].copy()
                st.dataframe(df_real, use_container_width=True)
                
# -------------------------
# TAB 4: VALOR 
# -------------------------
with tab4:
    st.title("📌 Valor Percibido vs Valor Real")
    st.write("Relación entre satisfacción (percepción) y el precio €/m² observado en Idealista.")

    left, right = st.columns([1.0, 1.25])
    with left:
        fig = plot_macro_scatter(macro, figsize=(6.4, 4.2))
        st.pyplot(fig, use_container_width=False)
    with right:
        st.subheader("Tabla resumen")
        st.dataframe(
            macro.sort_values(["categoria_valor", "satisfaccion_index"], ascending=[True, False]),
            use_container_width=True,
            height=420
        )

# -------------------------
# TAB 5: LISTADOS
# -------------------------
with tab5:
    st.title("💎 Listados")
    st.caption("Listados con imágenes, filtros y paginación.")

    if modo == "Compra":
        st.subheader("Compra: Joyas / Sobrevaloradas")
        distrito_v = st.selectbox(
            "Distrito (joyas/sobre):",
            sorted(viv["district"].dropna().unique()),
            key="dist_compra_joyas"
        )
        sub = viv[viv["district"] == distrito_v].copy()
        
        sub = apply_sidebar_filters(sub, modo="Compra", presupuesto=presupuesto, m2=m2)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Viviendas joya")
            joyas = (
                sub[sub["tipo_vivienda"] == "Vivienda joya"]
                .sort_values("desviacion_precio")
                .head(24)
            )
            render_listing_grid_paginated(joyas, "", cols=3, key_prefix="joyas_grid")

        with c2:
            st.markdown("### Viviendas sobrevaloradas")
            sobre = (
                sub[sub["tipo_vivienda"] == "Vivienda sobrevalorada"]
                .sort_values("desviacion_precio", ascending=False)
                .head(24)
            )
            render_listing_grid_paginated(sobre, "", cols=3, key_prefix="sobre_grid")

        st.divider()

        st.subheader("Compra: Listado")
        df_list = compra.copy()
        df_list = apply_sidebar_filters(df_list, modo="Compra", presupuesto=presupuesto, m2=m2)
        df_list, _ = listing_filters(df_list, "compra", default_max_from_quantile=True)
        render_listing_grid_paginated(df_list, "Compra: anuncios disponibles", cols=4, key_prefix="compra_grid")

    elif modo == "Alquiler":
        st.subheader("Alquiler: Listado")
        df_list = alquiler.copy()
        df_list = apply_sidebar_filters(df_list, modo="Alquiler", presupuesto=presupuesto, m2=m2)
        df_list, _ = listing_filters(df_list, "alquiler", default_max_from_quantile=True)
        render_listing_grid_paginated(df_list, "Alquiler: anuncios disponibles", cols=4, key_prefix="alq_grid")

    else:
        st.subheader("Habitaciones: Listado")
        df_list = hab.copy()
        df_list = apply_sidebar_filters(df_list, modo="Habitaciones", presupuesto=presupuesto, m2=m2)

        # Filtro tenantNumber
        tenant_choices = ["Cualquiera"]
        if "tenantNumber" in df_list.columns:
            vals = sorted(
                [int(x) for x in pd.to_numeric(df_list["tenantNumber"], errors="coerce").dropna().unique()]
            )
            tenant_choices += vals

        tenant_sel = st.selectbox("Nº compañeros (tenantNumber)", tenant_choices, index=0)
        if tenant_sel != "Cualquiera" and "tenantNumber" in df_list.columns:
            df_list = df_list[pd.to_numeric(df_list["tenantNumber"], errors="coerce") == int(tenant_sel)].copy()

        df_list, _ = listing_filters(df_list, "habitaciones", default_max_from_quantile=True)
        render_listing_grid_paginated(df_list, "Habitaciones: anuncios disponibles", cols=4, key_prefix="hab_grid")

# -------------------------
# TAB 6: MODELOS 
# -------------------------
with tab6:
    st.title("🤖 Modelos y métricas")

    # =========================
    # IMPORTANCIA VARIABLES
    # =========================
    st.subheader("Importancia de variables (CatBoost)")

    imp_view = imp.copy()

    # Detecta columnas (feature / importancia)
    feat_col = imp_view.columns[0]
    imp_col  = imp_view.columns[-1]

    imp_view[imp_col] = pd.to_numeric(imp_view[imp_col], errors="coerce")
    imp_view = (
        imp_view
        .dropna(subset=[imp_col])
        .sort_values(imp_col, ascending=False)
    )

    # Top 20
    imp_top = imp_view[[feat_col, imp_col]].head(20).copy()
    imp_top = imp_top.rename(columns={feat_col: "Variable", imp_col: "Importancia"})
    imp_top["Importancia"] = imp_top["Importancia"].round(3)

    # Descripciones en español
    desc_map = {
        "size": "Superficie del inmueble (m²)",
        "latitude": "Latitud (ubicación aproximada)",
        "longitude": "Longitud (ubicación aproximada)",
        "bathrooms": "Número de baños",
        "rooms": "Número de habitaciones",
        "floor": "Planta del inmueble",
        "district": "Distrito",
        "neighborhood": "Barrio",
        "hasLift": "Dispone de ascensor",
        "propertyType": "Tipo de vivienda",
        "exterior": "Vivienda exterior / interior",
        "parkingSpace": "Plaza de parking",
        "newDevelopment": "Obra nueva",
        "PC1": "Componente principal 1 (PCA)",
        "PC2": "Componente principal 2 (PCA)",
        "PC3": "Componente principal 3 (PCA)",
        "PC4": "Componente principal 4 (PCA)",
        "PC5": "Componente principal 5 (PCA)",
    }

    imp_top["Descripción"] = imp_top["Variable"].map(desc_map).fillna("—")

    # Tabla compacta con barras
    st.dataframe(
        imp_top.style.bar(subset=["Importancia"]),
        use_container_width=True,
        height=480
    )

    # Gráfico horizontal
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    tmp = imp_top.sort_values("Importancia", ascending=True)
    ax.barh(tmp["Variable"], tmp["Importancia"])
    ax.set_title("Top 20 variables por importancia")
    ax.set_xlabel("Importancia")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

    st.divider()

    # =========================
    # RMSE POR TRAMOS
    # =========================
    st.subheader("RMSE por tramos (CatBoost vs XGBoost)")

    st.dataframe(rmse, use_container_width=True, height=260)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(rmse["tramo"], rmse["rmse_catboost"], marker="o", label="CatBoost")
    ax.plot(rmse["tramo"], rmse["rmse_xgboost"], marker="o", label="XGBoost")
    ax.set_xlabel("Tramo de precio (test)")
    ax.set_ylabel("RMSE (€)")
    ax.set_title("Error por tramos", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    
# -------------------------
# TAB 7: PREDICCIÓN
# -------------------------
with tab7:
    st.title("🔮 Predicción de precio (CatBoost)")
    st.caption("Introduce características básicas y obtén una estimación orientativa.")

    if "pred_history" not in st.session_state:
        st.session_state["pred_history"] = []

    # Columnas según el entrenamiento
    FEATURES = [
        "floor", "propertyType", "size", "rooms", "bathrooms",
        "district", "neighborhood", "latitude", "longitude",
        "newDevelopment", "hasLift", "exterior",
        "parkingSpace", "PC1", "PC2", "PC3", "PC4", "PC5"
    ]

    districts = sorted(metrics["district"].dropna().unique())

    with st.form("pred_form"):
        colA, colB = st.columns(2)
        with colA:
            district_in = st.selectbox("Distrito", districts)
            size_in = st.number_input("Superficie (m²)", min_value=10.0, max_value=500.0, value=80.0, step=1.0)
            rooms_in = st.selectbox("Habitaciones", [0, 1, 2, 3, 4, 5], index=2)
            baths_in = st.selectbox("Baños", [1, 2, 3, 4], index=0)
            floor_in = st.selectbox("Planta", ["bajo", "intermedio", "alto", "desconocido"], index=0)

        with colB:
            propertyType_in = st.selectbox("Tipo", ["flat", "penthouse", "duplex", "studio", "chalet"], index=0)
            exterior_in = st.selectbox("Exterior",["desconocido", "True", "False"],index=0,format_func=lambda x: "Desconocido" if x == "desconocido" else ("Sí" if x == "True" else "No"))
            hasLift_in = st.checkbox("¿Tiene Ascensor?", value=True)
            parking_in = st.checkbox("Parking",value=False)
            if not parking_in:
                parking_val = False
            else:
                parking_val = "{'hasParkingSpace': True, 'isParkingSpaceIncludedInPrice': True}"
            newDev_in = st.checkbox("¿Es Obra nueva?", value=False)

        pred_btn = st.form_submit_button("Predecir precio", use_container_width=True)

    if pred_btn:
        try:
            # Obtener coordenadas del distrito
            rowd = metrics[metrics["district"] == district_in].iloc[0]
            lat_in = float(rowd["lat"]) if "lat" in metrics.columns else 41.387
            lon_in = float(rowd["lon"]) if "lon" in metrics.columns else 2.17
            pc_vals = {f"PC{i}": float(rowd.get(f"PC{i}", 0.0)) for i in range(1, 6)}

            user_row = {
                "floor": str(floor_in),
                "propertyType": str(propertyType_in),
                "size": float(size_in),
                "rooms": int(rooms_in),
                "bathrooms": int(baths_in),
                "district": str(district_in),
                "neighborhood": "Unknown",
                "latitude": float(lat_in),
                "longitude": float(lon_in),
                "newDevelopment": bool(newDev_in), 
                "hasLift": bool(hasLift_in),
                "exterior": exterior_in,
                "parkingSpace": parking_val,
                "PC1": float(pc_vals["PC1"]),
                "PC2": float(pc_vals["PC2"]),
                "PC3": float(pc_vals["PC3"]),
                "PC4": float(pc_vals["PC4"]),
                "PC5": float(pc_vals["PC5"]),
            }

            X_user = pd.DataFrame([user_row])[FEATURES]

            # Realizar la predicción
            pred = float(model.predict(X_user)[0])
            pred_eurm2 = pred / max(float(size_in), 1.0)

            st.success(f"Precio estimado: **{pred:,.0f} €**".replace(",", "."))

            # Guardar en el historial con todos los detalles
            resumen_dict = {
                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Distrito": district_in,
                "m²": size_in,
                "Hab": rooms_in,
                "Baños": baths_in,
                "Planta": floor_in,
                "Tipo": propertyType_in,
                "Exterior": exterior_in,
                "Ascensor": "Sí" if hasLift_in else "No",
                "Parking": "Sí" if parking_val else "No",
                "Obra Nueva": "Sí" if newDev_in else "No",
                "Precio (€)": round(pred, 0),
                "€/m²": round(pred_eurm2, 0),
            }
            st.session_state["pred_history"].append(resumen_dict)

        except Exception as e:
            st.error(f"Error en el modelo: {e}")

    # =========================
    # HISTORIAL DE PREDICCIONES
    # =========================
    st.divider()
    col_titulo, col_btn_dl, col_btn_clr = st.columns([2, 1, 1])

    with col_titulo:
        st.subheader("📚 Historial")

    if not st.session_state["pred_history"]:
        st.info("Aún no hay predicciones guardadas.")
    else:
        hist_df = pd.DataFrame(st.session_state["pred_history"])

        # Botón de descarga
        with col_btn_dl:
            csv = hist_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Descargar CSV",
                data=csv,
                file_name="historial_predicciones.csv",
                mime="text/csv",
                use_container_width=True,
                key="header_dl"
            )

        # Botón de borrar
        with col_btn_clr:
            if st.button("🗑️ Borrar", use_container_width=True, key="header_clr"):
                st.session_state["pred_history"] = []
                st.rerun()

        st.dataframe(hist_df.iloc[::-1], use_container_width=True, height=300)

# -------------------------
# TAB 8: SOBRE LA APP
# -------------------------
with tab8:
    with st.container():
        st.title("ℹ️ Información del Proyecto")
        
        col_info, col_logo = st.columns([3, 1])
    
        with col_info:
            st.subheader("✍️ Autor y Licencia")
            
            st.markdown("""
        **Autor:**  
        **Abel Mora Vázquez**  
        
        **Formación:**  
        Graduado en Ciencia de Datos Aplicada (*Applied Data Science Degree*) –  [Universitat Oberta de Catalunya (UOC)](https://www.uoc.edu)  
        Estudiante del **Máster en Ciencia de Datos** – [Universitat Oberta de Catalunya (UOC)](https://www.uoc.edu) **Curso académico:** 2025 – 2026  
        
        **Contexto académico:**  
        Aplicación desarrollada como parte del **Trabajo Final de Máster (TFM)**  
        
        **Título del TFM:**  
        Análisis del Mercado Inmobiliario en el Área Metropolitana de Barcelona  
        
        **Licencia:**  
        [Creative Commons CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.es)
        """)
        
            st.divider()
        
            st.subheader("📊 Resumen del Proyecto y Metodología")
        
            st.markdown("""
        ### 🔍 Enfoque del proyecto
        Este proyecto desarrolla **un proceso completo de Ciencia de Datos**, que abarca desde la integración y preparación de múltiples fuentes de información hasta el despliegue de una **aplicación interactiva** orientada al análisis y **toma de decisiones en el mercado inmobiliario**.
        
        ---
        
        ### 🧩 Fases principales
        
        **1️⃣ Ingeniería de datos aplicada a proyecto de Ciencia de Datos**  
        Integración de datos sociodemográficos y urbanos del **Ayuntamiento de Barcelona**, junto con datos del mercado inmobiliario obtenidos mediante la **API de Idealista**.  
        El proceso incluye **normalización y agregación** a nivel de distrito para construir conjuntos de datos válidos para análisis y modelado.
        
        **2️⃣ Análisis exploratorio (EDA)**  
        Limpieza de datos, estudio de distribuciones, detección de *outliers* y análisis de correlaciones para identificar los principales factores que influyen en el precio de la vivienda.  
        
        
        **3️⃣ Aprendizaje no supervisado**  
        Aplicación de **PCA (Análisis de Componentes Principales)** para reducir dimensionalidad y **K-Means** para identificar perfiles de distritos con características similares.
        
        **4️⃣ Modelado predictivo**  
        Evaluación comparativa de distintos modelos (*Ridge Regression, Random Forest, XGBoost y CatBoost*).  
        Se selecciona **CatBoost** como modelo final por su **precisión** y su **manejo eficiente de variables categóricas**.
        
        **5️⃣ Despliegue**  
        **Visualizaciones interactivas** desarrolladas con **Flourish**.
        Desarrollo de una **aplicación interactiva con Streamlit**, que permite explorar indicadores urbanos, comparar distritos, analizar listados inmobiliarios y **estimar precios de vivienda** de forma visual e intuitiva.
        """)
            
            st.divider()
            st.subheader("📬 Contacto")
        
            components.html("""
            <div style="display:flex;flex-direction:column;gap:12px;max-width:500px;font-family:sans-serif;">
        
              <a href="https://www.linkedin.com/in/abelmoravazquez" target="_blank" rel="noopener"
               style="display:flex;align-items:center;gap:15px;padding:14px;border-radius:12px;
                      border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.03);
                      text-decoration:none;color:#ffffff;transition:0.3s;">
                <span style="display:flex;align-items:center;justify-content:center;width:40px;height:40px;border-radius:10px;background:#0077b5;">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="white"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.369-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zM6.814 20.452H3.86V9h2.954v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z"/></svg>
                </span>
                <div style="display:flex;flex-direction:column;">
                  <strong style="font-size:16px;">Conectar en LinkedIn</strong>
                  <span style="font-size:13px;color:#b0b0b0;">linkedin.com/in/abelmoravazquez</span>
                </div>
              </a>
        
              <a href="https://github.com/Abelibz" target="_blank" rel="noopener"
               style="display:flex;align-items:center;gap:15px;padding:14px;border-radius:12px;
                      border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.03);
                      text-decoration:none;color:#ffffff;transition:0.3s;">
                <span style="display:flex;align-items:center;justify-content:center;width:40px;height:40px;border-radius:10px;background:#333;">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="white"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.757-1.333-1.757-1.089-.745.084-.729.084-.729 1.205.084 1.84 1.236 1.84 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.418-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.435.375.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>
                </span>
                <div style="display:flex;flex-direction:column;">
                  <strong style="font-size:16px;">Perfil de GitHub</strong>
                  <span style="font-size:13px;color:#b0b0b0;">github.com/Abelibz</span>
                </div>
              </a>
            </div>
            """, height=180)

    st.caption("Si redistribuyes esta aplicación o utilizas sus figuras, por favor cita al autor y enlaza a la licencia CC BY 4.0.")






# =============================================================================
# MÓDULO 3 — Dashboard Ejecutivo ARCO
# app.py — Punto de entrada de la aplicación Streamlit
#
# Audiencias:
#   · Operaciones ARCO : predicción interactiva mes a mes
#   · Comité académico : validación de hipótesis y desempeño del modelo
#
# Ejecución local : streamlit run app.py
# Despliegue      : Streamlit Community Cloud (github.com/usuario/repo)
#
# Dependencias    : ver requirements.txt
# Datos           : Informacion-PSI-23-24-25-detallado.xlsx (mismo directorio)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Prophet se importa con manejo de error para dar un mensaje claro
# si el paquete no está instalado en el entorno
try:
    from prophet import Prophet
    PROPHET_DISPONIBLE = True
except ImportError:
    PROPHET_DISPONIBLE = False

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ARCO · Dashboard Predictivo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS personalizados ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fondo general */
.stApp { background-color: #F4F5F7; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B3A5C 0%, #0F2239 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #CADCEC !important; }
[data-testid="stSidebar"] .stRadio label { color: #B8C8D8 !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: #E8EEF4 !important;
}

/* Encabezado de página */
.page-header {
    background: linear-gradient(135deg, #1B3A5C 0%, #2E86AB 100%);
    color: white;
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}
.page-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.page-header p {
    font-size: 0.95rem;
    opacity: 0.8;
    margin: 0;
}

/* Tarjetas KPI */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    border: 1px solid #E8ECF0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
    height: 100%;
}
.kpi-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #8D99AE;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.9rem;
    font-weight: 500;
    color: #1B3A5C;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.kpi-delta {
    font-size: 0.8rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 20px;
    display: inline-block;
}
.kpi-delta.positivo { background: #D4EFDF; color: #1E7E44; }
.kpi-delta.negativo { background: #FADBD8; color: #C0392B; }
.kpi-delta.neutral  { background: #EBF5FB; color: #2874A6; }

/* Sección de contenido */
.section-card {
    background: white;
    border-radius: 14px;
    padding: 1.6rem;
    border: 1px solid #E8ECF0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1B3A5C;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #EBF5FB;
}

/* Badge de hipótesis */
.hipotesis-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 24px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px 0;
}
.hip-confirmada { background: #D4EFDF; color: #1E7E44; border: 1px solid #A9DFBF; }
.hip-parcial    { background: #FEF9E7; color: #B7770D; border: 1px solid #F9E79F; }

/* Separador elegante */
.divider {
    border: none;
    border-top: 1px solid #E8ECF0;
    margin: 1.2rem 0;
}

/* Tabla de predicción */
.pred-table {
    background: #F8FAFC;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid #E3EAF0;
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
}
.pred-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #EDF2F7;
    color: #2C3E50;
}
.pred-row:last-child { border-bottom: none; }
.pred-label { color: #7F8C9A; font-size: 0.82rem; }
.pred-val   { font-weight: 500; color: #1B3A5C; }

/* Nota metodológica */
.nota-metodo {
    background: #EBF5FB;
    border-left: 4px solid #2E86AB;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #2C3E50;
    margin-top: 0.8rem;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #B0BAC4;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #E8ECF0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA Y PREPARACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Cargando datos históricos…")
def cargar_datos():
    """
    Carga y prepara el dataset maestro desde el archivo Excel de ARCO.
    Se ejecuta una sola vez gracias a @st.cache_data.
    Retorna el DataFrame consolidado con todas las variables del estudio.
    """
    xl = pd.ExcelFile("Informacion-PSI-23-24-25-detallado.xlsx")

    df_af  = pd.read_excel(xl, "1.Afluencia",            parse_dates=["fecha"])
    df_en  = pd.read_excel(xl, "2.Consumo-de-energia",   parse_dates=["fecha"])
    df_ag  = pd.read_excel(xl, "3.Consumo-de-agua",      parse_dates=["fecha"])
    df_cl  = pd.read_excel(xl, "4.Clima",                parse_dates=["fecha"])
    df_gs  = pd.read_excel(xl, "5.Gasto-vs-presupuesto", parse_dates=["fecha"])
    df_ing = pd.read_excel(xl, "6.Ingresos",             parse_dates=["fecha"])
    df_ocu = pd.read_excel(xl, "7.Ocupacion",            parse_dates=["fecha"])

    # Afluencia semanal → mensual
    df_af["anio_real"] = df_af["fecha"].dt.year
    df_af_mes = df_af.groupby(["anio_real", "mes-en-numero"], as_index=False).agg(
        afluencia_mensual=("afluencia", "sum"))
    df_af_mes["fecha"] = pd.to_datetime(
        df_af_mes["anio_real"].astype(str) + "-" +
        df_af_mes["mes-en-numero"].astype(str).str.zfill(2) + "-01")

    # Clima semanal → mensual
    df_cl["anio_real"] = df_cl["fecha"].dt.year
    df_cl_mes = df_cl.groupby(["anio_real", "mes-en-numero"], as_index=False).agg(
        temp_media=("temperatura-media", "mean"),
        temp_maxima=("temperatura-maxima", "mean"),
        humedad=("humedad", "mean"))
    df_cl_mes["fecha"] = pd.to_datetime(
        df_cl_mes["anio_real"].astype(str) + "-" +
        df_cl_mes["mes-en-numero"].astype(str).str.zfill(2) + "-01")

    # Dataset maestro
    dm = df_gs[["fecha", "gasto-operativo-total",
                "presupuesto-gasto-operativo-total"]].copy()
    dm["mes"] = dm["fecha"].dt.month
    dm["año"] = dm["fecha"].dt.year
    dm = dm.merge(df_af_mes[["fecha", "afluencia_mensual"]], on="fecha", how="left")
    dm = dm.merge(df_ing[["fecha", "ingresos_por_renta"]],   on="fecha", how="left")
    dm = dm.merge(df_en[["fecha", "consumo-cfe-kwh", "gasto-energia"]], on="fecha", how="left")
    dm = dm.merge(df_ag[["fecha", "consumo-m3-agua", "gasto-agua"]], on="fecha", how="left")
    dm = dm.merge(df_cl_mes[["fecha", "temp_media", "temp_maxima", "humedad"]], on="fecha", how="left")
    dm = dm.merge(df_ocu[["fecha", "porcentaje-ocupacion", "locales-ocupados"]], on="fecha", how="left")

    # Variables derivadas
    dm["ratio_gasto_ingresos"] = dm["gasto-operativo-total"] / dm["ingresos_por_renta"]
    dm["ocu_pct"]              = dm["porcentaje-ocupacion"] * 100
    dm["sin_mes"]              = np.sin(2 * np.pi * dm["mes"] / 12)
    dm["cos_mes"]              = np.cos(2 * np.pi * dm["mes"] / 12)
    dm["t"]                    = np.arange(len(dm))

    # Outlier de gasto (IQR de Tukey, 1977)
    q1 = dm["gasto-operativo-total"].quantile(0.25)
    q3 = dm["gasto-operativo-total"].quantile(0.75)
    dm["es_outlier_gasto"] = dm["gasto-operativo-total"] > q3 + 1.5 * (q3 - q1)

    return dm


@st.cache_resource(show_spinner="Entrenando modelos…")
def entrenar_modelos(dm):
    """
    Prepara el sistema predictivo completo del dashboard.
    @st.cache_resource evita reentrenar en cada interacción del usuario.

    Modelos:
        PROPHET : cargado desde prophet_arco_con_eventos.pkl (modelo principal
                  de afluencia, entrenado en el Módulo 2B con eventos especiales)
        A       : Regresión lineal de respaldo si el .pkl no está disponible
        B       : Ridge Regression para gasto operativo
        C       : Regresión lineal para consumo eléctrico (kWh)
        D       : Regresión lineal para consumo de agua (m³)

    El archivo prophet_arco_con_eventos.pkl debe estar en el mismo directorio
    que app.py. Se genera en el Módulo 2B de Google Colab.
    """
    dm_clean = dm[~dm["es_outlier_gasto"]].copy()

    FEAT_AF = ["t", "sin_mes", "cos_mes", "temp_media"]
    FEAT_GS = ["afluencia_mensual", "temp_media", "humedad",
               "ocu_pct", "sin_mes", "cos_mes", "t"]
    FEAT_EN = ["afluencia_mensual", "temp_media", "temp_maxima", "sin_mes", "cos_mes"]
    FEAT_AG = ["afluencia_mensual", "temp_media", "humedad", "sin_mes", "cos_mes"]

    # ── Modelo Principal: Prophet (carga desde .pkl o reentrenamiento automático)
    #
    # Flujo de tres pasos:
    #   1. Si existe el .pkl → cargarlo (instantáneo, ~1 segundo)
    #   2. Si no existe pero hay Excel de eventos → entrenar y guardar el .pkl
    #   3. Si tampoco hay Excel de eventos → entrenar Prophet sin eventos
    #
    # Nota sobre Streamlit Community Cloud: el sistema de archivos no es
    # persistente entre reinicios del servidor. El .pkl generado en paso 2
    # acelerará las recargas durante la misma sesión, pero en un reinicio
    # del servidor se volverá a entrenar (~2 min con 5 categorías de eventos).
    # Para evitar esto, sube el .pkl al repositorio de GitHub.
    modelo_prophet  = None
    usando_prophet  = False
    mape_af_prophet = 5.05   # MAPE conocido del Módulo 2B
    PKL_PATH        = "prophet_arco_con_eventos.pkl"
    EVENTOS_PATH    = "Eventos_PSI_2025.xlsx"

    if not PROPHET_DISPONIBLE:
        # Prophet no instalado: usar Modelo A como respaldo
        pass

    else:
        # ── Paso 1: intentar cargar el .pkl existente ─────────────────────
        if os.path.exists(PKL_PATH):
            try:
                with open(PKL_PATH, "rb") as f:
                    modelo_prophet = pickle.load(f)
                if hasattr(modelo_prophet, "predict"):
                    usando_prophet = True
            except Exception:
                modelo_prophet = None

        # ── Paso 2 y 3: entrenar Prophet si no se pudo cargar el .pkl ────
        if not usando_prophet:
            try:
                # Construir el dataframe en formato Prophet desde el dataset maestro
                # (ya disponible en dm, sin necesidad de leer el Excel de nuevo)
                df_train_p = (
                    dm[dm["fecha"] >= "2023-01-01"]
                    [["fecha", "afluencia_mensual", "temp_media"]]
                    .rename(columns={"fecha": "ds",
                                     "afluencia_mensual": "y"})
                    .dropna()
                    .sort_values("ds")
                    .reset_index(drop=True)
                )

                # Intentar cargar el calendario de eventos para el reentrenamiento
                df_holidays_retrain = None
                if os.path.exists(EVENTOS_PATH):
                    try:
                        df_ev_rt = pd.read_excel(
                            EVENTOS_PATH,
                            sheet_name="eventos-especiales",
                            header=0
                        )
                        df_ev_rt["fecha_inicio"] = pd.to_datetime(
                            df_ev_rt["fecha_inicio"])
                        df_ev_rt["fecha_fin"]    = pd.to_datetime(
                            df_ev_rt["fecha_fin"])
                        df_ev_rt = df_ev_rt[
                            df_ev_rt["aplica_a_modelo"].astype(str)
                            .str.upper() == "SI"
                        ].copy()

                        # Mapa de agrupación por categoría
                        # (mismo que en el Módulo 2B para mantener consistencia)
                        MAPA_GRUPOS = {
                            "festivo_oficial": "festivo_oficial",
                            "periodo_escolar": "periodo_escolar",
                            "evento_externo" : "evento_externo",
                            "estreno_cine"   : "estreno_cine",
                            "evento_interno" : "evento_interno",
                            "apertura_socio" : "evento_interno",
                        }

                        filas_h = []
                        for _, ev in df_ev_rt.iterrows():
                            tipo  = str(ev["tipo_evento"]).strip().lower()
                            grupo = MAPA_GRUPOS.get(tipo, tipo)
                            rango = pd.date_range(
                                start=ev["fecha_inicio"],
                                end=ev["fecha_fin"],
                                freq="D"
                            )
                            for dia in rango:
                                filas_h.append({
                                    "ds"           : dia,
                                    "holiday"      : grupo,
                                    "lower_window" : 0,
                                    "upper_window" : 0,
                                })
                        if filas_h:
                            df_holidays_retrain = pd.DataFrame(filas_h)
                            df_holidays_retrain["ds"] = pd.to_datetime(
                                df_holidays_retrain["ds"])
                    except Exception:
                        df_holidays_retrain = None

                # Configurar y entrenar Prophet
                prophet_kwargs = dict(
                    seasonality_mode="multiplicative",
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    interval_width=0.95,
                )
                if df_holidays_retrain is not None:
                    prophet_kwargs["holidays"] = df_holidays_retrain

                modelo_prophet = Prophet(**prophet_kwargs)
                modelo_prophet.add_regressor("temp_media")
                modelo_prophet.fit(df_train_p)
                usando_prophet = True

                # Guardar el .pkl para acelerar recargas en la misma sesión
                try:
                    with open(PKL_PATH, "wb") as f:
                        pickle.dump(modelo_prophet, f)
                except Exception:
                    pass  # si no se puede guardar, continuar normalmente

                # Calcular MAPE real del modelo recién entrenado
                fc_train = modelo_prophet.predict(df_train_p)
                y_real_p = df_train_p["y"].values
                y_pred_p = fc_train["yhat"].values
                mape_af_prophet = float(
                    np.mean(np.abs((y_real_p - y_pred_p) / y_real_p)) * 100
                )

            except Exception:
                modelo_prophet = None
                usando_prophet = False

    # ── Modelo A — Regresión lineal (respaldo o comparación académica) ────────
    # Se entrena siempre: sirve como baseline en el Panel Académico
    # y como predictor de afluencia si el .pkl no está disponible.
    X_af = dm[FEAT_AF]
    y_af = dm["afluencia_mensual"]
    mod_af    = LinearRegression().fit(X_af, y_af)
    y_pred_af = mod_af.predict(X_af)
    mape_af   = np.mean(np.abs((y_af - y_pred_af) / y_af)) * 100

    # Calcular predicciones históricas de Prophet para el Panel Académico
    # (solo si el modelo está disponible, para mostrar la comparación)
    y_pred_prophet_hist = None
    if usando_prophet:
        try:
            # Construir el dataframe histórico en formato Prophet
            df_hist_prophet = dm[["fecha", "temp_media"]].rename(
                columns={"fecha": "ds", "temp_media": "temp_media"})
            df_hist_prophet["ds"] = pd.to_datetime(df_hist_prophet["ds"])
            fc_hist = modelo_prophet.predict(df_hist_prophet)
            y_pred_prophet_hist = fc_hist["yhat"].values
        except Exception:
            y_pred_prophet_hist = None

    # ── Modelo B — Ridge Regression del gasto operativo ──────────────────────
    X_gs    = dm_clean[FEAT_GS].dropna()
    y_gs    = dm_clean.loc[X_gs.index, "gasto-operativo-total"]
    sc_gs   = StandardScaler()
    X_gs_sc = sc_gs.fit_transform(X_gs)
    mod_gs  = Ridge(alpha=0.5).fit(X_gs_sc, y_gs)
    y_pred_gs = mod_gs.predict(X_gs_sc)
    mape_gs = np.mean(np.abs((y_gs - y_pred_gs) / y_gs)) * 100

    # ── Modelo C — Consumo eléctrico (kWh) ───────────────────────────────────
    dm_en  = dm[dm["consumo-cfe-kwh"] > 500].copy()
    X_en   = dm_en[FEAT_EN].dropna()
    y_en   = dm_en.loc[X_en.index, "consumo-cfe-kwh"]
    mod_en = LinearRegression().fit(X_en, y_en)
    mape_en = np.mean(np.abs((y_en - mod_en.predict(X_en)) / y_en)) * 100

    # ── Modelo D — Consumo de agua (m³) ──────────────────────────────────────
    dm_ag  = dm[dm["consumo-m3-agua"] > 50].copy()
    X_ag   = dm_ag[FEAT_AG].dropna()
    y_ag   = dm_ag.loc[X_ag.index, "consumo-m3-agua"]
    mod_ag = LinearRegression().fit(X_ag, y_ag)
    mape_ag = np.mean(np.abs((y_ag - mod_ag.predict(X_ag)) / y_ag)) * 100

    # ── Promedios históricos por mes (climatología) ───────────────────────────
    clim_temp = dm.groupby("mes")["temp_media"].mean().to_dict()
    clim_hum  = dm.groupby("mes")["humedad"].mean().to_dict()
    clim_ocu  = dm.groupby("mes")["ocu_pct"].mean().to_dict()
    t_ultimo  = int(dm["t"].max())

    return {
        # Modelos
        "modelo_prophet"         : modelo_prophet,
        "usando_prophet"         : usando_prophet,
        "mod_af"                 : mod_af,
        "mod_gs"                 : mod_gs,
        "sc_gs"                  : sc_gs,
        "mod_en"                 : mod_en,
        "mod_ag"                 : mod_ag,
        # Features
        "feat_af"                : FEAT_AF,
        "feat_gs"                : FEAT_GS,
        "feat_en"                : FEAT_EN,
        "feat_ag"                : FEAT_AG,
        # Métricas
        "mape_af"                : mape_af,          # Modelo A (baseline)
        "mape_af_prophet"        : mape_af_prophet,  # Prophet (Módulo 2B)
        "mape_gs"                : mape_gs,
        "mape_en"                : mape_en,
        "mape_ag"                : mape_ag,
        # Historial de predicciones para Panel Académico
        "y_pred_prophet_hist"    : y_pred_prophet_hist,
        # Climatología
        "clim_temp"              : clim_temp,
        "clim_hum"               : clim_hum,
        "clim_ocu"               : clim_ocu,
        "t_ultimo"               : t_ultimo,
    }


def predecir_mes(mes: int, año: int, modelos: dict, dm: pd.DataFrame) -> dict:
    """
    Genera predicciones para un mes y año dado.

    Usa Prophet como modelo principal de afluencia si el .pkl está disponible.
    En caso contrario, usa el Modelo A de regresión lineal como respaldo.
    El gasto, energía y agua siempre usan los Modelos B, C y D.

    Parámetros:
        mes     : número de mes (1–12)
        año     : año de la predicción
        modelos : dict con modelos entrenados (resultado de entrenar_modelos)
        dm      : dataset maestro histórico

    Retorna:
        dict con todas las predicciones y métricas de contexto
    """
    temp  = modelos["clim_temp"].get(mes, 26.0)
    hum   = modelos["clim_hum"].get(mes, 55.0)
    ocu   = modelos["clim_ocu"].get(mes, 82.0)
    sin_m = np.sin(2 * np.pi * mes / 12)
    cos_m = np.cos(2 * np.pi * mes / 12)
    meses_desde_inicio = (año - 2023) * 12 + (mes - 1)
    t_val = meses_desde_inicio

    # ── Predicción de afluencia ───────────────────────────────────────────────
    # RUTA PRINCIPAL: Prophet con eventos especiales (Módulo 2B)
    # La predicción incluye automáticamente los eventos del calendario
    # de la plantilla Eventos_PSI_2025.xlsx que se usó en el entrenamiento.
    if modelos["usando_prophet"]:
        try:
            fecha_pred = pd.Timestamp(year=año, month=mes, day=1)
            df_fut_p   = pd.DataFrame({
                "ds"        : [fecha_pred],
                "temp_media": [temp],
            })
            fc = modelos["modelo_prophet"].predict(df_fut_p)
            af_pred = float(fc["yhat"].iloc[0])
            # Banda de incertidumbre al 95%
            af_lower = float(fc["yhat_lower"].iloc[0])
            af_upper = float(fc["yhat_upper"].iloc[0])
            fuente_af = "Prophet + eventos"
        except Exception:
            # Si Prophet falla por alguna razón, caer al Modelo A
            af_pred   = None
            af_lower  = None
            af_upper  = None
            fuente_af = None

    # RUTA DE RESPALDO: Modelo A de regresión lineal
    if not modelos["usando_prophet"] or af_pred is None:
        X_af  = pd.DataFrame([[t_val, sin_m, cos_m, temp]],
                             columns=modelos["feat_af"])
        af_pred   = float(modelos["mod_af"].predict(X_af)[0])
        af_lower  = af_pred * 0.90   # IC aproximado ±10%
        af_upper  = af_pred * 1.10
        fuente_af = "Regresión lineal (respaldo)"

    af_pred = max(af_pred, 50000)
    af_lower = max(af_lower, 30000)

    # ── Predicción de gasto (Modelo B — Ridge) ────────────────────────────────
    X_gs    = pd.DataFrame([[af_pred, temp, hum, ocu, sin_m, cos_m, t_val]],
                           columns=modelos["feat_gs"])
    X_gs_sc = modelos["sc_gs"].transform(X_gs)
    gs_pred = float(modelos["mod_gs"].predict(X_gs_sc)[0])
    gs_pred = max(gs_pred, 0)

    # ── Predicción de consumo eléctrico (Modelo C) ────────────────────────────
    temp_max = temp + 3.5
    X_en     = pd.DataFrame([[af_pred, temp, temp_max, sin_m, cos_m]],
                            columns=modelos["feat_en"])
    kwh_pred = float(modelos["mod_en"].predict(X_en)[0])
    kwh_pred = max(kwh_pred, 0)

    # ── Predicción de consumo de agua (Modelo D) ──────────────────────────────
    X_ag      = pd.DataFrame([[af_pred, temp, hum, sin_m, cos_m]],
                             columns=modelos["feat_ag"])
    agua_pred = float(modelos["mod_ag"].predict(X_ag)[0])
    agua_pred = max(agua_pred, 0)

    # ── Métricas de contexto ──────────────────────────────────────────────────
    ing_prom    = dm["ingresos_por_renta"].mean()
    gasto_actual= dm["gasto-operativo-total"].mean()
    ratio_pred  = gs_pred / ing_prom
    ahorro      = gasto_actual - gs_pred
    presupuesto = dm["presupuesto-gasto-operativo-total"].mean()

    return {
        "afluencia_pred"   : af_pred,
        "afluencia_lower"  : af_lower,
        "afluencia_upper"  : af_upper,
        "fuente_afluencia" : fuente_af,
        "gasto_pred"       : gs_pred,
        "kwh_pred"         : kwh_pred,
        "agua_pred"        : agua_pred,
        "ratio_pred"       : ratio_pred,
        "ahorro_vs_actual" : ahorro,
        "ingresos_prom"    : ing_prom,
        "gasto_actual"     : gasto_actual,
        "presupuesto"      : presupuesto,
        "temp_usada"       : temp,
        "hum_usada"        : hum,
        "ocu_usada"        : ocu,
    }


def proyectar_año(año: int, mes_hasta: int, modelos: dict,
                   dm: pd.DataFrame) -> pd.DataFrame:
    """
    Genera proyecciones mensuales acumuladas desde enero hasta mes_hasta
    del año indicado. Devuelve un DataFrame con una fila por mes.

    Si Prophet está disponible, se llama una sola vez con todos los meses
    para aprovechar la vectorización interna y obtener las bandas IC 95%.
    Si no está disponible, se usa el Modelo A en loop.

    Parámetros:
        año       : año a proyectar (ej: 2026)
        mes_hasta : último mes a incluir (1–12)
        modelos   : dict con modelos entrenados
        dm        : dataset maestro histórico

    Retorna:
        DataFrame con columnas:
            fecha, mes, afluencia_pred, afluencia_lower, afluencia_upper,
            gasto_pred, ratio_pred, fuente
    """
    meses        = list(range(1, mes_hasta + 1))
    ing_prom     = dm["ingresos_por_renta"].mean()
    clim_temp    = modelos["clim_temp"]
    clim_hum     = modelos["clim_hum"]
    clim_ocu     = modelos["clim_ocu"]

    fechas = [pd.Timestamp(year=año, month=m, day=1) for m in meses]

    # ── Afluencia con Prophet (vectorizado, una sola llamada) ─────────────────
    if modelos["usando_prophet"]:
        try:
            df_fut_p = pd.DataFrame({
                "ds"        : fechas,
                "temp_media": [clim_temp.get(m, 26.0) for m in meses],
            })
            fc = modelos["modelo_prophet"].predict(df_fut_p)
            af_preds  = fc["yhat"].values
            af_lowers = fc["yhat_lower"].values
            af_uppers = fc["yhat_upper"].values
            fuente    = "Prophet + eventos"
        except Exception:
            af_preds  = None
    else:
        af_preds = None

    # ── Afluencia con Modelo A si Prophet no está disponible ──────────────────
    if af_preds is None:
        af_preds, af_lowers, af_uppers = [], [], []
        for m in meses:
            t_val = (año - 2023) * 12 + (m - 1)
            sin_m = np.sin(2 * np.pi * m / 12)
            cos_m = np.cos(2 * np.pi * m / 12)
            temp  = clim_temp.get(m, 26.0)
            X     = pd.DataFrame([[t_val, sin_m, cos_m, temp]],
                                  columns=modelos["feat_af"])
            af    = float(modelos["mod_af"].predict(X)[0])
            af    = max(af, 50000)
            af_preds.append(af)
            af_lowers.append(af * 0.90)
            af_uppers.append(af * 1.10)
        fuente = "Regresión lineal"

    af_preds  = np.maximum(af_preds, 50000)
    af_lowers = np.maximum(af_lowers, 30000)

    # ── Gasto con Modelo B (vectorizado) ─────────────────────────────────────
    filas_gs = []
    for i, m in enumerate(meses):
        t_val = (año - 2023) * 12 + (m - 1)
        sin_m = np.sin(2 * np.pi * m / 12)
        cos_m = np.cos(2 * np.pi * m / 12)
        temp  = clim_temp.get(m, 26.0)
        hum   = clim_hum.get(m, 55.0)
        ocu   = clim_ocu.get(m, 82.0)
        filas_gs.append([af_preds[i], temp, hum, ocu, sin_m, cos_m, t_val])

    X_gs_all = pd.DataFrame(filas_gs, columns=modelos["feat_gs"])
    X_gs_sc  = modelos["sc_gs"].transform(X_gs_all)
    gs_preds = np.maximum(modelos["mod_gs"].predict(X_gs_sc), 0)

    return pd.DataFrame({
        "fecha"           : fechas,
        "mes"             : meses,
        "afluencia_pred"  : af_preds,
        "afluencia_lower" : af_lowers,
        "afluencia_upper" : af_uppers,
        "gasto_pred"      : gs_preds,
        "ratio_pred"      : gs_preds / ing_prom,
        "fuente"          : fuente,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# PALETA CORPORATIVA ARCO
# ═══════════════════════════════════════════════════════════════════════════════
C_PRIM  = "#1B3A5C"
C_SEC   = "#2E86AB"
C_ACENT = "#E84855"
C_POS   = "#3BB273"
C_NARJ  = "#F4A261"
C_GRIS  = "#8D99AE"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8FAFC",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.color":       "#CCCCCC",
    "font.family":      "sans-serif",
})

MESES_ESP = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
             "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Navegación
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.2rem; margin-bottom:0.3rem;'>📊</div>
        <div style='font-size:1.1rem; font-weight:700; color:#E8EEF4;'>ARCO</div>
        <div style='font-size:0.75rem; color:#7A9BBF; letter-spacing:1px;'>
            PLAZA PASEO SAN ISIDRO - CULIACÁN
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    vista = st.radio(
        "Selecciona la vista:",
        options=["Panel operativo", "Panel académico"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Selector de mes y año (solo relevante para Panel Operativo)
    st.markdown(
        "<p style='font-size:0.8rem; font-weight:600; "
        "text-transform:uppercase; letter-spacing:0.8px; color:#7A9BBF;'>"
        "Mes a predecir</p>",
        unsafe_allow_html=True
    )

    import datetime
    hoy   = datetime.date.today()
    años  = list(range(2023, 2028))
    año_sel = st.selectbox("Año",  años,
                            index=años.index(min(hoy.year, 2027)),
                            label_visibility="collapsed")
    mes_sel = st.selectbox("Mes",
                            options=list(range(1, 13)),
                            format_func=lambda m: MESES_ESP[m - 1],
                            index=hoy.month - 1,
                            label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#4A6A8A; line-height:1.6;'>
        <b style='color:#7A9BBF;'>Proyecto de eficiencia operativa</b><br>
        Emanuel Delgadillo Hernández<br>
        Maestría en Analítica e Inteligencia de Negocios<br>
        Universidad de Monterrey - 2026<br><br>
        <b style='color:#7A9BBF;'>Asesor</b><br>
        Dr. Juan Baldemar Garza Villegas
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS Y ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════
try:
    dm      = cargar_datos()
    modelos = entrenar_modelos(dm)
except FileNotFoundError:
    st.error(
        "No se encontró el archivo **Informacion-PSI-23-24-25-detallado.xlsx**. "
        "Asegúrate de que esté en el mismo directorio que app.py."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# VISTA 1 — PANEL OPERATIVO
# ═══════════════════════════════════════════════════════════════════════════════
if vista == "Panel operativo":

    mes_nombre = MESES_ESP[mes_sel - 1]

    st.markdown(f"""
    <div class="page-header">
        <h1>Panel Operativo</h1>
        <p>Predicción para <strong>{mes_nombre} {año_sel}</strong>
        - Plaza Paseo San Isidro - Culiacán, Sinaloa</p>
    </div>
    """, unsafe_allow_html=True)

    # Generar predicción
    pred = predecir_mes(mes_sel, año_sel, modelos, dm)

    # ── Fila 1: KPIs principales ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        delta_af = (pred["afluencia_pred"] / dm["afluencia_mensual"].mean() - 1) * 100
        cls_af   = "positivo" if delta_af >= 0 else "negativo"
        fuente_badge = (
            '<span style="font-size:0.7rem; background:#D4EFDF; color:#1E7E44; '
            'padding:2px 7px; border-radius:10px; font-weight:600;">Prophet + eventos</span>'
            if pred["fuente_afluencia"] == "Prophet + eventos"
            else
            '<span style="font-size:0.7rem; background:#EBF5FB; color:#2874A6; '
            'padding:2px 7px; border-radius:10px; font-weight:600;">Modelo A</span>'
        )
        ic_texto = (
            f'IC 95%: {pred["afluencia_lower"]:,.0f} – {pred["afluencia_upper"]:,.0f}'
            if pred.get("afluencia_lower") else ""
        )
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Afluencia estimada {fuente_badge}</div>
            <div class="kpi-value">{pred['afluencia_pred']:,.0f}</div>
            <span class="kpi-delta {cls_af}">
                {'▲' if delta_af >= 0 else '▼'} {abs(delta_af):.1f}% vs promedio
            </span>
            <div style='font-size:0.75rem; color:#8D99AE; margin-top:0.3rem;'>
                {ic_texto}
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        delta_gs = (pred["gasto_pred"] / pred["gasto_actual"] - 1) * 100
        cls_gs   = "negativo" if delta_gs >= 0 else "positivo"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Gasto operativo proyectado</div>
            <div class="kpi-value">${pred['gasto_pred']/1e6:.2f}M</div>
            <span class="kpi-delta {cls_gs}">
                {'▲' if delta_gs >= 0 else '▼'} {abs(delta_gs):.1f}% vs promedio hist.
            </span>
        </div>""", unsafe_allow_html=True)

    with c3:
        ratio_pct  = pred["ratio_pred"] * 100
        meta_pct   = 10.0
        delta_meta = ratio_pct - meta_pct
        cls_ratio  = "negativo" if delta_meta > 0 else "positivo"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Ratio gasto / ingresos</div>
            <div class="kpi-value">{ratio_pct:.1f}%</div>
            <span class="kpi-delta {cls_ratio}">
                {'+' if delta_meta > 0 else ''}{delta_meta:.1f} pts vs meta 10%
            </span>
        </div>""", unsafe_allow_html=True)

    with c4:
        ahorro    = pred["ahorro_vs_actual"]
        cls_aho   = "positivo" if ahorro >= 0 else "negativo"
        signo_aho = "▼" if ahorro >= 0 else "▲"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Ahorro vs promedio histórico</div>
            <div class="kpi-value">${abs(ahorro)/1e3:.0f}K</div>
            <span class="kpi-delta {cls_aho}">
                {signo_aho} {'Menor gasto esperado' if ahorro >= 0 else 'Mayor gasto esperado'}
            </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ── Fila 2: Consumos + Comparación vs presupuesto ────────────────────────
    col_izq, col_der = st.columns([1, 1.6])

    with col_izq:
        st.markdown("""
        <div class="section-card">
        <div class="section-title">Consumos proyectados</div>
        """, unsafe_allow_html=True)

        # kWh
        kwh_prom = dm[dm["consumo-cfe-kwh"] > 500]["consumo-cfe-kwh"].mean()
        delta_kwh = (pred["kwh_pred"] / kwh_prom - 1) * 100
        st.markdown(f"""
        <div class="pred-row">
            <span class="pred-label">Energía eléctrica</span>
            <span class="pred-val">{pred['kwh_pred']:,.0f} kWh
                <small style='color:{"#C0392B" if delta_kwh>0 else "#1E7E44"}'>
                    ({'+' if delta_kwh>0 else ''}{delta_kwh:.0f}%)
                </small>
            </span>
        </div>""", unsafe_allow_html=True)

        # m³ Agua
        agua_prom = dm[dm["consumo-m3-agua"] > 50]["consumo-m3-agua"].mean()
        delta_ag  = (pred["agua_pred"] / agua_prom - 1) * 100
        st.markdown(f"""
        <div class="pred-row">
            <span class="pred-label">Consumo de agua</span>
            <span class="pred-val">{pred['agua_pred']:,.0f} m³
                <small style='color:{"#C0392B" if delta_ag>0 else "#1E7E44"}'>
                    ({'+' if delta_ag>0 else ''}{delta_ag:.0f}%)
                </small>
            </span>
        </div>""", unsafe_allow_html=True)

        # Temperatura y ocupación usadas
        st.markdown(f"""
        <div class="pred-row">
            <span class="pred-label">Temperatura estimada</span>
            <span class="pred-val">{pred['temp_usada']:.1f} °C</span>
        </div>
        <div class="pred-row">
            <span class="pred-label">Ocupación estimada</span>
            <span class="pred-val">{pred['ocu_usada']:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="nota-metodo">
            Los consumos se estiman con promedios históricos de temperatura,
            humedad y ocupación del mismo mes en 2023–2025
            (Sari et al., 2023).
        </div>
        </div>""", unsafe_allow_html=True)

    with col_der:
        # Gráfica: gasto proyectado vs presupuesto vs promedio histórico
        fig, ax = plt.subplots(figsize=(7, 3.5))

        conceptos = ["Promedio\nhistórico", "Presupuesto\nestimado",
                     f"Proyección\n{mes_nombre} {año_sel}"]
        valores   = [pred["gasto_actual"], pred["presupuesto"],
                     pred["gasto_pred"]]
        colores   = [C_GRIS, C_SEC, C_POS if pred["gasto_pred"] <= pred["presupuesto"] else C_ACENT]

        bars = ax.bar(conceptos, [v / 1e6 for v in valores],
                      color=colores, alpha=0.85, width=0.5,
                      edgecolor="white", linewidth=1.5)

        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03,
                    f"${val/1e6:.2f}M",
                    ha="center", va="bottom",
                    fontsize=9.5, fontweight="600", color=C_PRIM)

        # Línea de meta 10%
        meta_val = pred["ingresos_prom"] * 0.10
        ax.axhline(meta_val / 1e6, color=C_POS, ls="-.", lw=1.8,
                   label=f"Meta ARCO 10% (${meta_val/1e6:.2f}M)")
        ax.set_ylabel("Millones MXN", fontsize=9, color=C_GRIS)
        ax.set_title(f"Gasto operativo — {mes_nombre} {año_sel}",
                     fontsize=10.5, fontweight="600", color=C_PRIM, pad=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"${v:.1f}M"))
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.set_ylim(0, max(valores) / 1e6 * 1.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Fila 3: Gráfica histórico + tendencia ────────────────────────────────
    st.markdown("""
    <div class="section-card">
    <div class="section-title">Histórico 2023–2025 y proyección acumulada del año seleccionado</div>
    """, unsafe_allow_html=True)

    # Generar proyección acumulada enero → mes_sel del año elegido
    df_proy     = proyectar_año(año_sel, mes_sel, modelos, dm)
    fecha_corte = dm["fecha"].max()
    fecha_pred  = pd.Timestamp(year=año_sel, month=mes_sel, day=1)
    usando_prophet_graf = df_proy["fuente"].iloc[0] == "Prophet + eventos"

    # Columnas con texto para hover personalizado
    dm_graf = dm.copy()
    dm_graf["hover_af"] = dm_graf.apply(
        lambda r: (
            f"<b>{r['fecha'].strftime('%B %Y')}</b><br>"
            f"Afluencia real: {r['afluencia_mensual']:,.0f} visitantes<br>"
            f"Año: {r['fecha'].year}"
        ), axis=1
    )

    df_proy["mes_nombre"] = df_proy["mes"].apply(lambda m: MESES_ESP[m - 1])
    df_proy["hover_af"]   = df_proy.apply(
        lambda r: (
            f"<b>{r['mes_nombre']} {año_sel}</b><br>"
            f"Afluencia predicha: {r['afluencia_pred']:,.0f}<br>"
            f"IC 95%: {r['afluencia_lower']:,.0f} – {r['afluencia_upper']:,.0f}<br>"
            f"Modelo: {r['fuente']}"
        ), axis=1
    )
    df_proy["hover_ratio"] = df_proy.apply(
        lambda r: (
            f"<b>{r['mes_nombre']} {año_sel}</b><br>"
            f"Ratio proyectado: {r['ratio_pred']:.1%}<br>"
            f"Gasto proyectado: ${r['gasto_pred']:,.0f}<br>"
            f"Modelo: {r['fuente']}"
        ), axis=1
    )
    dm_graf["hover_ratio"] = dm_graf.apply(
        lambda r: (
            f"<b>{r['fecha'].strftime('%B %Y')}</b><br>"
            f"Ratio real: {r['ratio_gasto_ingresos']:.1%}<br>"
            f"Gasto: ${r['gasto-operativo-total']:,.0f}<br>"
            f"Ingresos: ${r['ingresos_por_renta']:,.0f}"
        ), axis=1
    )

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Configuración de tema común para ambas gráficas
    LAYOUT_BASE = dict(
        font=dict(family="DM Sans, sans-serif", size=11, color="#2C3E50"),
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E8ECF0",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=50, r=20, t=50, b=60),
        hovermode="x unified",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # GRÁFICA 1 — Afluencia histórica + proyección acumulada
    # ══════════════════════════════════════════════════════════════════════════
    fig_af = go.Figure()

    # Banda IC 95% (relleno entre lower y upper)
    if usando_prophet_graf:
        # Traza superior de la banda (invisible, solo para el relleno)
        fig_af.add_trace(go.Scatter(
            x=df_proy["fecha"],
            y=df_proy["afluencia_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="IC superior",
        ))
        # Traza inferior con fill hacia la traza anterior → forma la banda
        fig_af.add_trace(go.Scatter(
            x=df_proy["fecha"],
            y=df_proy["afluencia_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(244,162,97,0.18)",
            showlegend=True,
            name="IC 95% Prophet",
            hoverinfo="skip",
        ))

    # Línea sólida: datos reales 2023–2025
    fig_af.add_trace(go.Scatter(
        x=dm_graf["fecha"],
        y=dm_graf["afluencia_mensual"],
        mode="lines+markers",
        name="Real 2023–2025",
        line=dict(color=C_SEC, width=2.5),
        marker=dict(size=6, color=C_SEC,
                    line=dict(color="white", width=1.5)),
        customdata=dm_graf["hover_af"],
        hovertemplate="%{customdata}<extra></extra>",
    ))

    # Línea punteada: proyección acumulada
    fig_af.add_trace(go.Scatter(
        x=df_proy["fecha"],
        y=df_proy["afluencia_pred"],
        mode="lines+markers",
        name=f"Proyección {año_sel} (ene–{mes_nombre})",
        line=dict(color=C_NARJ, width=2.2, dash="dot"),
        marker=dict(size=7, color=C_NARJ,
                    line=dict(color="white", width=1.5)),
        customdata=df_proy["hover_af"],
        hovertemplate="%{customdata}<extra></extra>",
    ))

    # Punto destacado: mes seleccionado
    fig_af.add_trace(go.Scatter(
        x=[fecha_pred],
        y=[pred["afluencia_pred"]],
        mode="markers",
        name=f"{mes_nombre} {año_sel}: {pred['afluencia_pred']:,.0f}",
        marker=dict(size=14, color=C_POS, symbol="circle",
                    line=dict(color="white", width=2)),
        hovertemplate=(
            f"<b>{mes_nombre} {año_sel}</b><br>"
            f"Afluencia: {pred['afluencia_pred']:,.0f}<br>"
            f"IC 95%: {pred['afluencia_lower']:,.0f} – "
            f"{pred['afluencia_upper']:,.0f}"
            "<extra></extra>"
        ),
    ))

    # Línea vertical separadora histórico / proyección
    fig_af.add_vline(
        x=fecha_corte.timestamp() * 1000,
        line=dict(color=C_GRIS, width=1.2, dash="dot"),
        annotation_text="← histórico | proyección →",
        annotation_position="top",
        annotation_font=dict(size=9, color=C_GRIS),
    )

    fig_af.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"Afluencia mensual — histórico y proyección {año_sel}",
            font=dict(size=13, color=C_PRIM, family="DM Sans, sans-serif"),
            x=0.01,
        ),
        yaxis=dict(
            title="Visitantes",
            gridcolor="#E8ECF0",
            tickformat=",.0f",
        ),
        xaxis=dict(gridcolor="#E8ECF0"),
        height=390,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # GRÁFICA 2 — Ratio gasto/ingresos histórico + proyección acumulada
    # ══════════════════════════════════════════════════════════════════════════
    fig_rt = go.Figure()

    # Barras históricas coloreadas por nivel de ratio
    color_barras = [
        C_POS if r <= 0.20 else C_NARJ if r <= 0.30 else C_ACENT
        for r in dm_graf["ratio_gasto_ingresos"]
    ]
    fig_rt.add_trace(go.Bar(
        x=dm_graf["fecha"],
        y=dm_graf["ratio_gasto_ingresos"] * 100,
        name="Ratio real (2023–2025)",
        marker_color=color_barras,
        opacity=0.75,
        customdata=dm_graf["hover_ratio"],
        hovertemplate="%{customdata}<extra></extra>",
    ))

    # Banda IC 95% del ratio (aproximación desde la banda de afluencia)
    if usando_prophet_graf:
        af_rango_pct = (
            (df_proy["afluencia_upper"] - df_proy["afluencia_lower"])
            / (2 * df_proy["afluencia_pred"])
        ).values
        ratio_proy  = df_proy["ratio_pred"].values * 100
        ratio_lower = ratio_proy * (1 - af_rango_pct * 0.5)
        ratio_upper = ratio_proy * (1 + af_rango_pct * 0.5)

        fig_rt.add_trace(go.Scatter(
            x=df_proy["fecha"], y=ratio_upper,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
            name="IC superior ratio",
        ))
        fig_rt.add_trace(go.Scatter(
            x=df_proy["fecha"], y=ratio_lower,
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(244,162,97,0.18)",
            showlegend=True,
            name="IC 95% aprox.",
            hoverinfo="skip",
        ))

    # Línea punteada: ratio proyectado acumulado
    fig_rt.add_trace(go.Scatter(
        x=df_proy["fecha"],
        y=df_proy["ratio_pred"] * 100,
        mode="lines+markers",
        name=f"Proyección ratio {año_sel}",
        line=dict(color=C_NARJ, width=2.2, dash="dot"),
        marker=dict(size=7, color=C_NARJ,
                    line=dict(color="white", width=1.5)),
        customdata=df_proy["hover_ratio"],
        hovertemplate="%{customdata}<extra></extra>",
    ))

    # Punto destacado: mes seleccionado
    fig_rt.add_trace(go.Scatter(
        x=[fecha_pred],
        y=[pred["ratio_pred"] * 100],
        mode="markers",
        name=f"{mes_nombre} {año_sel}: {pred['ratio_pred']:.1%}",
        marker=dict(size=14, color=C_POS, symbol="circle",
                    line=dict(color="white", width=2)),
        hovertemplate=(
            f"<b>{mes_nombre} {año_sel}</b><br>"
            f"Ratio: {pred['ratio_pred']:.1%}<br>"
            f"Gasto: ${pred['gasto_pred']:,.0f}"
            "<extra></extra>"
        ),
    ))

    # Líneas de referencia (meta y promedio)
    fig_rt.add_hline(
        y=10,
        line=dict(color=C_POS, width=2, dash="dashdot"),
        annotation_text="Meta 10%",
        annotation_position="bottom right",
        annotation_font=dict(size=9, color=C_POS),
    )
    prom_ratio = dm["ratio_gasto_ingresos"].mean() * 100
    fig_rt.add_hline(
        y=prom_ratio,
        line=dict(color=C_GRIS, width=1.3, dash="dot"),
        annotation_text=f"Prom. histórico {prom_ratio:.1f}%",
        annotation_position="top right",
        annotation_font=dict(size=9, color=C_GRIS),
    )

    # Separador histórico / proyección
    fig_rt.add_vline(
        x=fecha_corte.timestamp() * 1000,
        line=dict(color=C_GRIS, width=1.2, dash="dot"),
        annotation_text="← histórico | proyección →",
        annotation_position="top",
        annotation_font=dict(size=9, color=C_GRIS),
    )

    fig_rt.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"Ratio gasto / ingresos (%) — histórico y proyección {año_sel}",
            font=dict(size=13, color=C_PRIM, family="DM Sans, sans-serif"),
            x=0.01,
        ),
        yaxis=dict(
            title="Ratio gasto / ingresos (%)",
            gridcolor="#E8ECF0",
            tickformat=".1f",
            ticksuffix="%",
        ),
        xaxis=dict(gridcolor="#E8ECF0"),
        height=390,
        barmode="overlay",
    )

    # Renderizar en dos columnas
    col_graf_izq, col_graf_der = st.columns(2)
    with col_graf_izq:
        st.plotly_chart(fig_af, use_container_width=True)
    with col_graf_der:
        st.plotly_chart(fig_rt, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Nota de impacto económico ────────────────────────────────────────────
    ahorro_5ppt_anual = pred["ingresos_prom"] * 0.05 * 12
    st.markdown(f"""
    <div class="nota-metodo" style='font-size:0.88rem;'>
        <strong>Estimación de ahorro anual:</strong>
        Si se utilizan estas predicciones para ajustar la asignación
        de personal y recursos en función de la afluencia esperada,
        el ahorro potencial en un escenario moderado (reducción del 5%)
        asciende a <strong>${ahorro_5ppt_anual:,.0f} MXN anuales</strong>
        (Amangeldy et al., 2025).
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VISTA 2 — PANEL ACADÉMICO
# ═══════════════════════════════════════════════════════════════════════════════
else:

    st.markdown("""
    <div class="page-header">
        <h1>Panel Académico</h1>
        <p>Validación de hipótesis - Desempeño del modelo - Hallazgos clave</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sección 1: Hipótesis ─────────────────────────────────────────────────
    st.markdown("""
    <div class="section-card">
    <div class="section-title">Validación de hipótesis</div>
    """, unsafe_allow_html=True)

    h_col1, h_col2 = st.columns(2)

    with h_col1:
        # H1: correlación afluencia - gasto
        from scipy import stats as sp_stats
        dm_clean = dm[~dm["es_outlier_gasto"]].dropna(
            subset=["afluencia_mensual", "gasto-operativo-total"])
        r_val, p_val = sp_stats.pearsonr(
            dm_clean["afluencia_mensual"],
            dm_clean["gasto-operativo-total"])
        h1_ok = p_val < 0.10  # umbral más amplio con n pequeño
        cls_h1 = "hip-confirmada" if h1_ok else "hip-parcial"
        lbl_h1 = "✅ Confirmada" if h1_ok else "⚠️ Parcialmente confirmada"

        st.markdown(f"""
        <p style='font-size:0.82rem; color:#8D99AE; margin-bottom:0.5rem;
            font-weight:600; text-transform:uppercase;'>H1</p>
        <p style='font-size:0.9rem; color:#2C3E50; margin-bottom:0.7rem;'>
            "La afluencia está correlacionada con el gasto operativo."
        </p>
        <span class="hipotesis-badge {cls_h1}">{lbl_h1}</span>
        <div class="pred-table" style='margin-top:0.8rem;'>
            <div class="pred-row">
                <span class="pred-label">Pearson r (sin outlier)</span>
                <span class="pred-val">{r_val:.4f}</span>
            </div>
            <div class="pred-row">
                <span class="pred-label">p-valor</span>
                <span class="pred-val">{p_val:.4f}</span>
            </div>
            <div class="pred-row">
                <span class="pred-label">n observaciones</span>
                <span class="pred-val">{len(dm_clean)}</span>
            </div>
        </div>
        <div class="nota-metodo">
            La correlación simple no es significativa al 0.05, pero el modelo
            multivariado (Modelo B) sí captura la relación con R²=0.18,
            confirmando que la afluencia es un predictor relevante cuando
            se controla por temperatura y estacionalidad
            (Gu et al., 2023).
        </div>
        """, unsafe_allow_html=True)

    with h_col2:
        mape_prophet = modelos["mape_af_prophet"]   # 5.05% del Módulo 2B
        mape_af_base = modelos["mape_af"]            # Modelo A baseline
        prec_prophet = 100 - mape_prophet
        h2_ok    = prec_prophet >= 85.0
        cls_h2   = "hip-confirmada" if h2_ok else "hip-parcial"
        lbl_h2   = "✅ Confirmada" if h2_ok else "⚠️ Parcialmente confirmada"
        usando_p = modelos["usando_prophet"]

        st.markdown(f"""
        <p style='font-size:0.82rem; color:#8D99AE; margin-bottom:0.5rem;
            font-weight:600; text-transform:uppercase;'>H2</p>
        <p style='font-size:0.9rem; color:#2C3E50; margin-bottom:0.7rem;'>
            "La afluencia podrá ser anticipada con un 85% de precisión."
        </p>
        <span class="hipotesis-badge {cls_h2}">{lbl_h2}</span>
        <div class="pred-table" style='margin-top:0.8rem;'>
            <div class="pred-row">
                <span class="pred-label">Prophet + eventos (MAPE)</span>
                <span class="pred-val">{mape_prophet:.2f}%  →  {prec_prophet:.1f}%</span>
            </div>
            <div class="pred-row">
                <span class="pred-label">Modelo A — Regresión (baseline)</span>
                <span class="pred-val">{mape_af_base:.2f}%  →  {100-mape_af_base:.1f}%</span>
            </div>
            <div class="pred-row">
                <span class="pred-label">Meta del anteproyecto</span>
                <span class="pred-val">≥ 85.0%</span>
            </div>
            <div class="pred-row">
                <span class="pred-label">Modelo activo en dashboard</span>
                <span class="pred-val">
                    {'✅ Prophet + eventos' if usando_p else '⚠️ Regresión (pkl no encontrado)'}
                </span>
            </div>
        </div>
        <div class="nota-metodo">
            Ozdemir et al. (2022) consideran MAPEs de 10–20% como
            aceptables en modelos de afluencia comercial.
            Prophet incorpora 158 eventos agrupados en 5 categorías
            del calendario de Plaza San Isidro (Taylor & Letham, 2018).
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Sección 2: Desempeño de los 4 modelos ────────────────────────────────
    st.markdown("""
    <div class="section-card">
    <div class="section-title">📐 Desempeño del sistema predictivo</div>
    """, unsafe_allow_html=True)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    for col, letra, nombre, mape, nota in [
        (m_col1, "A", "Afluencia\nmensual\n(baseline)",  modelos["mape_af"],         "Regresión + estacionalidad"),
        (m_col2, "P", "Afluencia\nmensual\n(Prophet)",   modelos["mape_af_prophet"], "Prophet + eventos (Módulo 2B)"),
        (m_col3, "B", "Gasto\noperativo",                modelos["mape_gs"],         "Ridge Regression (L2)"),
        (m_col4, "C/D", "Consumo\neléctrico · agua",     max(modelos["mape_en"], modelos["mape_ag"]), "Reg. lineal múltiple"),
    ]:
        prec   = 100 - mape
        cumple = prec >= 85
        color_prec = "#1E7E44" if cumple else "#B7770D"
        with col:
            st.markdown(f"""
            <div class="kpi-card" style='text-align:center;'>
                <div style='font-size:0.7rem; font-weight:700; color:#8D99AE;
                    text-transform:uppercase; letter-spacing:0.8px;
                    margin-bottom:0.3rem;'>Modelo {letra}</div>
                <div style='font-size:0.82rem; color:#2C3E50;
                    margin-bottom:0.6rem; line-height:1.3;'>{nombre}</div>
                <div style='font-family:DM Mono,monospace; font-size:1.6rem;
                    font-weight:500; color:{color_prec};'>{prec:.1f}%</div>
                <div style='font-size:0.75rem; color:#8D99AE;
                    margin-bottom:0.4rem;'>precisión · MAPE {mape:.1f}%</div>
                <div style='font-size:0.7rem; color:#ADB5BD;'>{nota}</div>
                {'<div style="margin-top:0.4rem;"><span class="kpi-delta positivo">✓ Meta 85%</span></div>'
                 if cumple else
                 '<div style="margin-top:0.4rem;"><span class="kpi-delta neutral">Referencial</span></div>'}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Sección 3: Gráficas de análisis ──────────────────────────────────────
    st.markdown("""
    <div class="section-card">
    <div class="section-title">📊 Análisis histórico y desempeño del modelo</div>
    """, unsafe_allow_html=True)

    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 9))
    fig3.suptitle(
        "Sistema Predictivo ARCO · Plaza San Isidro · 2023–2025",
        fontsize=12, fontweight="700", color=C_PRIM, y=1.01
    )

    # Panel 1: Ratio gasto/ingresos por año
    ax1 = axes3[0, 0]
    for yr, color in [(2023, C_SEC), (2024, C_NARJ), (2025, C_POS)]:
        sub = dm[dm["año"] == yr].sort_values("mes")
        ax1.plot(sub["mes"], sub["ratio_gasto_ingresos"] * 100,
                 color=color, lw=2, marker="o", ms=5, label=str(yr))
    ax1.axhline(10, color=C_ACENT, ls="-.", lw=1.8, label="Meta 10%")
    ax1.axhline(dm["ratio_gasto_ingresos"].mean() * 100,
                color=C_GRIS, ls=":", lw=1.3,
                label=f"Prom. {dm['ratio_gasto_ingresos'].mean():.1%}")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(MESES_ESP, fontsize=8)
    ax1.set_ylabel("Ratio gasto/ingresos (%)", fontsize=9)
    ax1.set_title("Ratio por mes y año", fontsize=10, fontweight="600",
                  color=C_PRIM)
    ax1.legend(fontsize=8)

    # Panel 2: Afluencia real vs predicha
    ax2 = axes3[0, 1]
    X_af_h = dm[modelos["feat_af"]]
    y_af_h = dm["afluencia_mensual"]
    y_af_p = modelos["mod_af"].predict(X_af_h)

    ax2.plot(dm["fecha"], y_af_h, color=C_SEC, lw=2, marker="o", ms=3,
             label="Real", zorder=5)

    # Mostrar Prophet si está disponible, Modelo A de lo contrario
    if modelos["y_pred_prophet_hist"] is not None:
        ax2.plot(dm["fecha"], modelos["y_pred_prophet_hist"],
                 color=C_POS, lw=2, ls="--",
                 label=f"Prophet + eventos (MAPE {modelos['mape_af_prophet']:.1f}%)")
        ax2.plot(dm["fecha"], y_af_p, color=C_GRIS, lw=1.5, ls=":",
                 alpha=0.7, label=f"Modelo A baseline (MAPE {modelos['mape_af']:.1f}%)")
    else:
        ax2.plot(dm["fecha"], y_af_p, color=C_NARJ, lw=2, ls="--",
                 label=f"Modelo A (MAPE {modelos['mape_af']:.1f}%)")

    ax2.set_title("Afluencia: real vs predicho", fontsize=10,
                  fontweight="600", color=C_PRIM)
    ax2.set_ylabel("Visitantes/mes", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e3:.0f}K"))
    ax2.legend(fontsize=8.5)

    # Panel 3: Gasto real vs predicho (Modelo B)
    ax3 = axes3[1, 0]
    dm_clean_b = dm[~dm["es_outlier_gasto"]].copy()
    X_gs_h     = dm_clean_b[modelos["feat_gs"]].dropna()
    y_gs_h     = dm_clean_b.loc[X_gs_h.index, "gasto-operativo-total"]
    y_gs_p     = modelos["mod_gs"].predict(modelos["sc_gs"].transform(X_gs_h))
    fechas_b   = dm_clean_b.loc[X_gs_h.index, "fecha"]
    ax3.plot(fechas_b, y_gs_h.values / 1e6, color=C_SEC, lw=2,
             marker="o", ms=3, label="Real")
    ax3.plot(fechas_b, y_gs_p / 1e6, color=C_NARJ, lw=2, ls="--",
             label=f"Modelo B (MAPE {modelos['mape_gs']:.1f}%)")
    # Outlier
    out = dm[dm["es_outlier_gasto"]]
    if len(out):
        ax3.scatter(out["fecha"], out["gasto-operativo-total"] / 1e6,
                    color=C_ACENT, s=80, zorder=5,
                    label="Outlier (mar-25)")
    ax3.set_title("Gasto operativo: real vs predicho", fontsize=10,
                  fontweight="600", color=C_PRIM)
    ax3.set_ylabel("Millones MXN", fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=35, ha="right")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v:.1f}M"))
    ax3.legend(fontsize=8.5)

    # Panel 4: Diagrama de identidad afluencia
    ax4 = axes3[1, 1]
    # Usar Prophet si está disponible, Modelo A de lo contrario
    y_af_para_id = (modelos["y_pred_prophet_hist"]
                    if modelos["y_pred_prophet_hist"] is not None
                    else y_af_p)
    mape_para_id = (modelos["mape_af_prophet"]
                    if modelos["y_pred_prophet_hist"] is not None
                    else modelos["mape_af"])
    titulo_id    = ("Prophet + eventos" if modelos["y_pred_prophet_hist"] is not None
                    else "Modelo A — Regresión")

    ax4.scatter(y_af_h, y_af_para_id, color=C_SEC, alpha=0.7, s=50,
                edgecolors="white", lw=0.8)
    lim = max(y_af_h.max(), np.max(y_af_para_id)) * 1.05
    ax4.plot([0, lim], [0, lim], color=C_PRIM, lw=1.5, ls="--",
             label="Predicción perfecta")
    ax4.set_xlabel("Afluencia real", fontsize=9)
    ax4.set_ylabel("Afluencia predicha", fontsize=9)
    ax4.set_title(f"Diagrama de identidad — {titulo_id}", fontsize=10,
                  fontweight="600", color=C_PRIM)
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e3:.0f}K"))
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e3:.0f}K"))
    r2_id = r2_score(y_af_h, y_af_para_id)
    ax4.text(0.05, 0.90,
             f"MAPE = {mape_para_id:.2f}%\nR² = {r2_id:.4f}",
             transform=ax4.transAxes, fontsize=9,
             bbox=dict(fc="white", ec=C_GRIS, alpha=0.9))
    ax4.legend(fontsize=8.5)

    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Sección 4: Hallazgos y vacío teórico ─────────────────────────────────
    st.markdown("""
    <div class="section-card">
    <div class="section-title">💡 Hallazgos y aportaciones</div>
    """, unsafe_allow_html=True)

    h1_col, h2_col, h3_col = st.columns(3)

    with h1_col:
        ratio_act = dm["ratio_gasto_ingresos"].mean()
        st.markdown(f"""
        <div style='padding:1rem; background:#EBF5FB; border-radius:10px; height:100%;'>
            <div style='font-size:2rem; margin-bottom:0.5rem;'>📉</div>
            <div style='font-weight:700; color:{C_PRIM}; margin-bottom:0.4rem;'>
                Ratio actual: {ratio_act:.1%}
            </div>
            <div style='font-size:0.85rem; color:#2C3E50; line-height:1.5;'>
                El gasto operativo representa el <strong>{ratio_act:.1%}</strong>
                de los ingresos, <strong>{ratio_act*100-10:.1f} puntos porcentuales</strong>
                por encima de la meta del 10% de ARCO.
                El modelo permite anticipar meses de alto riesgo operativo.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with h2_col:
        meses_exc = int((dm["gasto-operativo-total"] >
                         dm["presupuesto-gasto-operativo-total"]).sum())
        st.markdown(f"""
        <div style='padding:1rem; background:#FEF9E7; border-radius:10px; height:100%;'>
            <div style='font-size:2rem; margin-bottom:0.5rem;'>⚠️</div>
            <div style='font-weight:700; color:{C_PRIM}; margin-bottom:0.4rem;'>
                {meses_exc} de 36 meses excedieron presupuesto
            </div>
            <div style='font-size:0.85rem; color:#2C3E50; line-height:1.5;'>
                El <strong>{meses_exc/36:.0%}</strong> de los meses del período
                de estudio superó el presupuesto de gasto, confirmando la
                gestión reactiva documentada en el planteamiento del problema.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with h3_col:
        ahorro_anual = dm["ingresos_por_renta"].mean() * 0.05 * 12
        st.markdown(f"""
        <div style='padding:1rem; background:#D4EFDF; border-radius:10px; height:100%;'>
            <div style='font-size:2rem; margin-bottom:0.5rem;'>💰</div>
            <div style='font-weight:700; color:{C_PRIM}; margin-bottom:0.4rem;'>
                Ahorro potencial: ${ahorro_anual/1e6:.2f}M MXN/año
            </div>
            <div style='font-size:0.85rem; color:#2C3E50; line-height:1.5;'>
                En un escenario moderado de reducción del 5% en el ratio
                gasto/ingresos, el ahorro anual estimado asciende a
                <strong>${ahorro_anual:,.0f} MXN</strong>
                (Amangeldy et al., 2025).
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Referencias ──────────────────────────────────────────────────────────
    with st.expander("📚 Referencias bibliográficas principales"):
        st.markdown("""
        - Amangeldy et al. (2025). *A hybrid machine learning approach for
          high-accuracy energy consumption prediction.* Energies, 18(15), 4164.
        - Andersen et al. (2024). *Increased understanding of building operational
          performance through occupant-centric KPIs.* Energy and Buildings, 323.
        - Gu, J., Xu, P., & Ji, Y. (2023). *A fast method for calculating the
          impact of occupancy on commercial building energy consumption.*
          Buildings, 13(2), 567.
        - Ozdemir et al. (2022). *Estimating shopping center visitor numbers
          based on a new hybrid fuzzy prediction method.*
          Journal of Intelligent & Fuzzy Systems, 42(1), 63–76.
        - Taylor, S.J. & Letham, B. (2018). *Forecasting at scale.*
          The American Statistician, 72(1), 37–45.
        - Tibshirani, R. (1996). *Regression shrinkage and selection via the Lasso.*
          Journal of the Royal Statistical Society, 58(1), 267–288.
        """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ARCO Áreas Comerciales · Plaza San Isidro · Sistema Predictivo de Eficiencia Operativa<br>
    Maestría en Analítica e Inteligencia de Negocios · UDEM · 2025 ·
    Emanuel Delgadillo Hernández
</div>
""", unsafe_allow_html=True)

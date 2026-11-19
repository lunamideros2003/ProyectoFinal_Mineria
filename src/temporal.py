"""Análisis temporal del sentimiento (Parte 3).

Agrupa los tweets por día, semana o mes a partir de la columna ``date`` del
dataset y resume, por periodo, el volumen de tweets positivos y negativos y la
proporción de positivos. Trabaja con las etiquetas reales y, si los modelos
están entrenados, también con la predicción de cada modelo o del consenso.

Fuente de datos: ``data/processed/muestra_entrenamiento.csv`` (ya trae
``date``, ``label`` y ``split``) o, en su defecto, el CSV original de
Sentiment140. No reentrena nada.
"""

import numpy as np
import pandas as pd

from src.config import DATASET_PATH, SAMPLE_PATH

GRANULARIDADES = ("dia", "semana", "mes")

# Vistas de sentimiento disponibles. "real" siempre existe; las demás
# requieren los modelos entrenados.
VISTAS = ("real", "Consenso", "Naive Bayes", "SVM", "Random Forest")

# Para elegir el periodo "más positivo / más negativo" se exige un mínimo de
# tweets, de modo que un periodo con 1 o 2 tweets no distorsione el hallazgo.
MIN_TWEETS_EXTREMO = 10


def parsear_fecha(serie: pd.Series) -> pd.Series:
    """Convierte la columna ``date`` de Sentiment140 a datetime.

    Formato original: ``Mon Apr 06 22:19:45 PDT 2009``. La zona ``PDT``
    molesta a algunos parsers, así que se retira antes de parsear.
    """
    limpia = serie.astype(str).str.replace(" PDT", "", regex=False)
    return pd.to_datetime(limpia, format="%a %b %d %H:%M:%S %Y", errors="coerce")


def cargar_base() -> pd.DataFrame | None:
    """Devuelve fecha, etiqueta y texto limpio (si existe) o None sin datos."""
    if SAMPLE_PATH.exists():
        m = pd.read_csv(SAMPLE_PATH, usecols=lambda c: c in {"date", "label", "texto_limpio", "text"})
    elif DATASET_PATH.exists():
        m = pd.read_csv(DATASET_PATH, encoding="latin-1", header=None,
                        names=["target", "ids", "date", "flag", "user", "text"])
        m["label"] = (m["target"] == 4).astype(int)
    else:
        return None
    m["fecha"] = parsear_fecha(m["date"])
    m = m.dropna(subset=["fecha"]).copy()
    if not len(m):
        return None
    m["label"] = pd.to_numeric(m["label"], errors="coerce")
    m = m.dropna(subset=["label"])
    m["label"] = m["label"].astype(int)
    if "texto_limpio" not in m.columns:
        m["texto_limpio"] = ""
    return m[["fecha", "label", "texto_limpio"]].sort_values("fecha").reset_index(drop=True)


def predecir_columna(base: pd.DataFrame, vectorizer, modelos, vista: str) -> np.ndarray:
    """Predicción (0/1) de un modelo o del consenso sobre la muestra base."""
    from src.preprocessing import limpiar_texto

    textos = base["texto_limpio"]
    if not textos.astype(str).str.strip().any():
        return np.array([])
    X = vectorizer.transform(textos.fillna("").tolist() if isinstance(textos, pd.Series) else textos)
    if vista == "Consenso":
        nombres = [n for n in ("Naive Bayes", "SVM", "Random Forest") if n in modelos]
        votos = np.vstack([np.asarray(modelos[n].predict(X)).ravel() for n in nombres])
        return (votos.sum(axis=0) >= 2).astype(int)
    return np.asarray(modelos[vista].predict(X)).ravel().astype(int)


def _clave_periodo(fechas: pd.Series, granularidad: str) -> pd.Series:
    if granularidad == "semana":
        lunes = fechas.dt.normalize() - pd.to_timedelta(fechas.dt.weekday, unit="d")
        return lunes
    if granularidad == "mes":
        return fechas.dt.to_period("M").dt.to_timestamp()
    return fechas.dt.normalize()


def _etiqueta(ts: pd.Timestamp, granularidad: str) -> str:
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    if granularidad == "mes":
        return f"{meses[ts.month - 1]} {ts.year}"
    if granularidad == "semana":
        fin = ts + pd.Timedelta(days=6)
        return f"{ts.day} {meses[ts.month - 1]} – {fin.day} {meses[fin.month - 1]}"
    return f"{ts.day} {meses[ts.month - 1]}"


def resumir_temporal(base: pd.DataFrame, granularidad: str = "dia",
                     desde: str | None = None, hasta: str | None = None,
                     valores: np.ndarray | None = None) -> dict:
    """Agrega la base por periodo y calcula KPIs, extremos e interpretación."""
    df = base
    mascara = pd.Series(True, index=base.index)
    if desde:
        mascara &= base["fecha"] >= pd.Timestamp(desde)
    if hasta:
        mascara &= base["fecha"] < pd.Timestamp(hasta) + pd.Timedelta(days=1)
    df = base[mascara]
    if not len(df):
        return {"vacio": True, "n": 0, "periodos": []}

    # ``valores`` (predicción de un modelo) viene alineado con ``base``
    # completa: se recorta con la misma máscara del filtro de fechas.
    if valores is not None:
        y = np.asarray(valores)[mascara.to_numpy()]
    else:
        y = df["label"].to_numpy()
    df = df.assign(periodo=_clave_periodo(df["fecha"], granularidad), _y=y)
    agg = df.groupby("periodo").agg(n=("periodo", "size"), pos=("_y", "sum")).reset_index()
    agg["neg"] = agg["n"] - agg["pos"]
    agg["pct_pos"] = (agg["pos"] / agg["n"]).round(4)
    agg = agg.sort_values("periodo")
    agg["fecha"] = agg["periodo"].dt.strftime("%Y-%m-%d")
    agg["etiqueta"] = agg["periodo"].map(lambda t: _etiqueta(t, granularidad))

    n = int(agg["n"].sum())
    pos = int(agg["pos"].sum())
    pct = round(pos / n, 4) if n else 0.0

    con_minimo = agg[agg["n"] >= MIN_TWEETS_EXTREMO]
    ref = con_minimo if len(con_minimo) else agg
    pico_pos = ref.loc[ref["pct_pos"].idxmax()]
    pico_neg = ref.loc[ref["pct_pos"].idxmin()]
    pico_vol = agg.loc[agg["n"].idxmax()]

    # Tendencia: recta de regresión sobre el % positivo por periodo.
    x = np.arange(len(agg))
    pendiente = float(np.polyfit(x, agg["pct_pos"].to_numpy(), 1)[0]) if len(agg) >= 3 else 0.0
    if pendiente > 0.005:
        tendencia = "al alza"
    elif pendiente < -0.005:
        tendencia = "a la baja"
    else:
        tendencia = "estable"

    # Cambio de régimen: el dataset pasa a ser todo negativo desde mediados
    # de junio de 2009 por cómo se recolectó (ver página de Análisis).
    cola_negativa = bool(((agg["pct_pos"] == 0) & (agg["n"] >= MIN_TWEETS_EXTREMO)).any())

    inter = (
        f"Entre el {agg['fecha'].iloc[0]} y el {agg['fecha'].iloc[-1]} el sentimiento se mantiene {tendencia} "
        f"({pct * 100:.1f} % positivo en el periodo filtrado)."
    )
    if cola_negativa:
        inter += " Los últimos periodos son totalmente negativos por el cambio en la recolección del dataset, no por un cambio de opinión."

    return {
        "vacio": False,
        "n": n,
        "n_pos": pos,
        "n_neg": n - pos,
        "pct_pos": pct,
        "tendencia": tendencia,
        "pendiente": round(pendiente, 5),
        "pico_pos": {"fecha": pico_pos["fecha"], "etiqueta": pico_pos["etiqueta"],
                     "pct_pos": float(pico_pos["pct_pos"]), "n": int(pico_pos["n"])},
        "pico_neg": {"fecha": pico_neg["fecha"], "etiqueta": pico_neg["etiqueta"],
                     "pct_pos": float(pico_neg["pct_pos"]), "n": int(pico_neg["n"])},
        "pico_vol": {"fecha": pico_vol["fecha"], "etiqueta": pico_vol["etiqueta"],
                     "n": int(pico_vol["n"])},
        "cola_negativa": cola_negativa,
        "interpretacion": inter,
        "periodos": agg[["fecha", "etiqueta", "n", "pos", "neg", "pct_pos"]].to_dict(orient="records"),
    }

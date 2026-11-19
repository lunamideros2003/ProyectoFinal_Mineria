"""Análisis temporal del sentimiento (Parte 3).

Agrupa los tweets por día, semana o mes a partir de la columna ``date`` y
resume, por periodo, el volumen de tweets positivos y negativos y la
proporción de positivos. Trabaja con las etiquetas reales y, si los modelos
están entrenados, con la predicción de cada modelo o de la votación.

Fuente de datos: ``data/processed/muestra_entrenamiento.csv`` (la genera
``src.train``; trae ``date``, ``label``, ``texto_limpio`` y ``split``).

Cambio en la recolección: a partir del 16 de junio de 2009 por la mañana el
dataset solo contiene tweets negativos (ver hallazgo 01 de /analisis). Los
periodos afectados se muestran, pero no cuentan para los extremos ni para la
tendencia, porque reflejan cómo se armó el dataset y no la opinión.
"""

import numpy as np
import pandas as pd

from src.analysis import wilson
from src.config import SAMPLE_PATH

GRANULARIDADES = ("dia", "semana", "mes")

# "real" siempre existe; las demás requieren los modelos entrenados.
VISTAS = ("real", "Consenso", "Naive Bayes", "SVM", "Random Forest")

# Un periodo necesita al menos este número de tweets (y el 1 % del rango)
# para optar a "más positivo" o "más negativo".
MIN_TWEETS_EXTREMO = 100

# Por debajo de este cambio (en puntos porcentuales a lo largo del rango) la
# tendencia se considera estable.
UMBRAL_TENDENCIA = 0.02

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def parsear_fecha(serie: pd.Series) -> pd.Series:
    """``Mon Apr 06 22:19:45 PDT 2009`` -> datetime (se retira la zona PDT)."""
    limpia = serie.astype(str).str.replace(" PDT", "", regex=False)
    return pd.to_datetime(limpia, format="%a %b %d %H:%M:%S %Y", errors="coerce")


def cargar_base() -> pd.DataFrame | None:
    """Fecha, etiqueta, texto limpio y partición de la muestra, o None si no existe."""
    if not SAMPLE_PATH.exists():
        return None
    m = pd.read_csv(SAMPLE_PATH, usecols=["date", "label", "texto_limpio", "split"])
    m["fecha"] = parsear_fecha(m["date"])
    m["label"] = pd.to_numeric(m["label"], errors="coerce")
    m = m.dropna(subset=["fecha", "label"])
    if m.empty:
        return None
    m["label"] = m["label"].astype(int)
    m["texto_limpio"] = m["texto_limpio"].fillna("")
    return m[["fecha", "label", "texto_limpio", "split"]].sort_values("fecha").reset_index(drop=True)


def cambio_recoleccion(base: pd.DataFrame) -> pd.Timestamp | None:
    """Momento a partir del cual ya no hay tweets positivos en las etiquetas reales.

    Solo se considera un cambio de recolección si después quedan suficientes
    tweets (todos negativos); si no, el último positivo es simplemente el final
    de los datos.
    """
    ultimo = base.loc[base["label"] == 1, "fecha"].max()
    if pd.isna(ultimo) or (base["fecha"] > ultimo).sum() < MIN_TWEETS_EXTREMO:
        return None
    return ultimo


def evaluacion(base: pd.DataFrame) -> pd.DataFrame:
    """Solo los tweets de evaluación: los modelos no los vieron al entrenar."""
    return base[base["split"] == "test"].reset_index(drop=True)


def predecir(base: pd.DataFrame, vectorizer, modelos, vista: str) -> np.ndarray:
    """Predicción 0/1 de un modelo, o de la votación de los tres, alineada con ``base``."""
    X = vectorizer.transform(base["texto_limpio"].tolist())
    if vista == "Consenso":
        votos = np.vstack([np.asarray(m.predict(X)).ravel() for m in modelos.values()])
        return (votos.sum(axis=0) * 2 > len(modelos)).astype(int)
    return np.asarray(modelos[vista].predict(X)).ravel().astype(int)


def _inicio_periodo(fechas: pd.Series, granularidad: str) -> pd.Series:
    if granularidad == "semana":
        return fechas.dt.normalize() - pd.to_timedelta(fechas.dt.weekday, unit="D")
    if granularidad == "mes":
        return fechas.dt.to_period("M").dt.to_timestamp()
    return fechas.dt.normalize()


def _todos_los_periodos(inicio: pd.Timestamp, fin: pd.Timestamp, granularidad: str) -> pd.DatetimeIndex:
    """Todos los periodos del rango, también los que no tienen tweets."""
    paso = {"dia": "D", "semana": "7D", "mes": "MS"}[granularidad]
    return pd.date_range(inicio, fin, freq=paso)


def _fin_periodo(ts: pd.Timestamp, granularidad: str) -> pd.Timestamp:
    if granularidad == "semana":
        return ts + pd.Timedelta(days=7)
    if granularidad == "mes":
        return ts + pd.offsets.MonthBegin(1)
    return ts + pd.Timedelta(days=1)


def _etiqueta(ts: pd.Timestamp, granularidad: str) -> str:
    if granularidad == "mes":
        return f"{MESES[ts.month - 1]} {ts.year}"
    if granularidad == "semana":
        fin = ts + pd.Timedelta(days=6)
        if fin.month == ts.month:
            return f"{ts.day} al {fin.day} {MESES[ts.month - 1]}"
        return f"{ts.day} {MESES[ts.month - 1]} al {fin.day} {MESES[fin.month - 1]}"
    return f"{ts.day} {MESES[ts.month - 1]}"


def _tendencia(validos: pd.DataFrame) -> tuple[str, float | None]:
    """Recta ponderada por volumen sobre el % positivo; devuelve el cambio total en el rango."""
    if len(validos) < 3:
        return "sin datos", None
    dias = (validos["periodo"] - validos["periodo"].iloc[0]).dt.days.to_numpy(dtype=float)
    pendiente = np.polyfit(dias, validos["pct_pos"].to_numpy(), 1, w=np.sqrt(validos["n"].to_numpy()))[0]
    cambio = float(pendiente * (dias[-1] - dias[0]))
    if cambio > UMBRAL_TENDENCIA:
        return "al alza", cambio
    if cambio < -UMBRAL_TENDENCIA:
        return "a la baja", cambio
    return "estable", cambio


def _extremo(periodo: pd.Series) -> dict:
    return {"fecha": periodo["fecha"], "etiqueta": periodo["etiqueta"],
            "pct_pos": float(periodo["pct_pos"]), "n": int(periodo["n"])}


def resumir_temporal(base: pd.DataFrame, granularidad: str = "dia", desde: str | None = None,
                     hasta: str | None = None, prediccion: np.ndarray | None = None,
                     corte: pd.Timestamp | None = None) -> dict:
    """Agrega por periodo y calcula volumen, % positivo, extremos y tendencia.

    ``prediccion`` (opcional) va alineada con ``base``; si se da, se resume la
    predicción y se conserva el % positivo real de cada periodo para comparar.
    ``corte`` es el cambio de recolección: los periodos que lo alcanzan quedan
    fuera de los extremos y de la tendencia.
    """
    mascara = pd.Series(True, index=base.index)
    if desde:
        mascara &= base["fecha"] >= pd.Timestamp(desde)
    if hasta:
        mascara &= base["fecha"] < pd.Timestamp(hasta) + pd.Timedelta(days=1)
    df = base.loc[mascara, ["fecha", "label"]]
    if df.empty:
        return {"vacio": True}
    y = df["label"].to_numpy() if prediccion is None else np.asarray(prediccion)[mascara.to_numpy()]
    df = df.assign(periodo=_inicio_periodo(df["fecha"], granularidad), y=y)

    agg = df.groupby("periodo").agg(n=("y", "size"), pos=("y", "sum"), pos_real=("label", "sum"))
    todos = _todos_los_periodos(agg.index.min(), agg.index.max(), granularidad)
    agg = agg.reindex(todos, fill_value=0).rename_axis("periodo").reset_index()
    agg["neg"] = agg["n"] - agg["pos"]
    con_datos = agg["n"] > 0
    agg["pct_pos"] = (agg["pos"] / agg["n"]).where(con_datos).round(4)
    agg["pct_real"] = (agg["pos_real"] / agg["n"]).where(con_datos).round(4)
    agg["fecha"] = agg["periodo"].dt.strftime("%Y-%m-%d")
    agg["etiqueta"] = agg["periodo"].map(lambda t: _etiqueta(t, granularidad))
    fin = agg["periodo"].map(lambda t: _fin_periodo(t, granularidad))
    agg["afectado"] = (fin > corte) if corte is not None else False

    n, pos = int(agg["n"].sum()), int(agg["pos"].sum())
    validos = agg[con_datos & ~agg["afectado"]]
    minimo = max(MIN_TWEETS_EXTREMO, int(0.01 * validos["n"].sum()))
    candidatos = validos[validos["n"] >= minimo]
    pico_pos = pico_neg = None
    if len(candidatos) >= 2:
        # Se ordena por el intervalo de Wilson: un periodo pequeño no gana por azar.
        inf = candidatos.apply(lambda p: wilson(int(p["pos"]), int(p["n"]))[0], axis=1)
        sup = candidatos.apply(lambda p: wilson(int(p["pos"]), int(p["n"]))[1], axis=1)
        pico_pos, pico_neg = _extremo(candidatos.loc[inf.idxmax()]), _extremo(candidatos.loc[sup.idxmin()])
    tendencia, cambio = _tendencia(validos)
    # Comparación antes / después del cambio, por tweet (no por periodo): un día
    # partido por el corte no mezcla etiquetas de los dos tramos.
    despues = df["fecha"] > corte if corte is not None else pd.Series(False, index=df.index)

    def tramo(sel: pd.Series) -> dict:
        if not sel.any():
            return {"n": 0, "pct_pos": None, "pct_real": None}
        return {"n": int(sel.sum()), "pct_pos": round(float(df.loc[sel, "y"].mean()), 4),
                "pct_real": round(float(df.loc[sel, "label"].mean()), 4)}

    return {
        "vacio": False,
        "n": n, "n_pos": pos, "n_neg": n - pos,
        "pct_pos": round(pos / n, 4),
        "pct_real": round(float(df["label"].mean()), 4),
        "desde": df["fecha"].min().strftime("%Y-%m-%d"),
        "hasta": df["fecha"].max().strftime("%Y-%m-%d"),
        "n_periodos": int(con_datos.sum()),
        "n_vacios": int((~con_datos).sum()),
        "minimo_extremo": minimo,
        "pico_pos": pico_pos,
        "pico_neg": pico_neg,
        "pico_vol": _extremo(agg.loc[agg["n"].idxmax()]),
        "tendencia": tendencia,
        "cambio": None if cambio is None else round(cambio, 4),
        "validos": tramo(~despues),
        "afectados": tramo(despues),
        "periodos": [{**p, "pct_pos": None if pd.isna(p["pct_pos"]) else p["pct_pos"],
                      "pct_real": None if pd.isna(p["pct_real"]) else p["pct_real"]}
                     for p in agg[["fecha", "etiqueta", "n", "pos", "neg", "pct_pos", "pct_real", "afectado"]]
                     .astype({"n": int, "pos": int, "neg": int, "afectado": bool}).to_dict(orient="records")],
    }

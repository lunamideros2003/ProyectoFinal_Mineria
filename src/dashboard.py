"""Resumen de sentimiento para el dashboard.

`resumir_lote` recibe cualquier conjunto de textos (el split de evaluación, el test
de marcas o los datos que aporte el usuario) y devuelve todo lo que muestra
el panel: KPIs, distribución, rendimiento por modelo, acuerdo, confianza y términos.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.config import COLUMNS, RAW_DIR, SAMPLE_PATH
from src.preprocessing import STOP_WORDS, limpiar_texto
from src.train import metricas_modelo

ORDEN_MODELOS = ["Naive Bayes", "SVM", "Random Forest"]
CONSENSO = "Consenso"


def _metricas(y, pred) -> dict:
    return metricas_modelo(y, pred)


def _terminos(limpios: pd.Series, pred: np.ndarray, top: int = 12) -> dict:
    """Términos más característicos de cada clase predicha (log-ratio suavizado)."""
    min_df = 3 if len(limpios) >= 500 else 1
    try:
        cv = CountVectorizer(stop_words=STOP_WORDS, min_df=min_df, token_pattern=r"(?u)\b[a-z][a-z0-9]{2,}\b", ngram_range=(1, 2), max_features=20_000)
        X = cv.fit_transform(limpios)
    except ValueError:  # vocabulario vacío
        return {"positivo": [], "negativo": []}
    vocab = np.array(cv.get_feature_names_out())
    pos = np.asarray(X[pred == 1].sum(0)).ravel()
    neg = np.asarray(X[pred == 0].sum(0)).ravel()
    ratio = np.log((pos + 1) / (pos.sum() + len(vocab))) - np.log((neg + 1) / (neg.sum() + len(vocab)))
    soporte = pos + neg
    umbral = max(2, np.percentile(soporte, 90)) if len(vocab) > 50 else 1
    idx = np.where(soporte >= umbral)[0]
    orden = idx[np.argsort(ratio[idx])]
    fmt = lambda ids, cuenta: [{"termino": vocab[i], "peso": round(float(abs(ratio[i])), 3), "n": int(cuenta[i])} for i in ids]
    return {"positivo": fmt(orden[::-1][:top], pos), "negativo": fmt(orden[:top], neg)}


def predecir_lote(textos, vectorizer, modelos) -> dict:
    """Predicción de los tres modelos y la votación para una lista de textos."""
    textos = pd.Series(list(textos), dtype="object").fillna("")
    limpios = textos.map(limpiar_texto)
    X = vectorizer.transform(limpios)
    nombres = [n for n in ORDEN_MODELOS if n in modelos]
    preds = {n: modelos[n].predict(X).astype(int) for n in nombres}
    suma = np.vstack([preds[n] for n in nombres]).sum(0)
    preds[CONSENSO] = (suma >= 2).astype(int)
    proba = modelos["Naive Bayes"].predict_proba(X)[:, 1] if "Naive Bayes" in modelos else None
    return {"textos": textos, "limpios": limpios, "nombres": nombres, "preds": preds, "suma": suma, "proba": proba}


def tabla_resultados(p: dict, etiquetas=None, grupos=None, fechas=None) -> pd.DataFrame:
    """Una fila por texto: predicción de cada modelo, votación y, si existen, etiqueta real y grupo."""
    sent = lambda v: np.where(v == 1, "positivo", "negativo")
    t = pd.DataFrame({"texto": p["textos"].to_numpy()})
    if fechas is not None:
        t["fecha"] = list(fechas)
    if grupos is not None:
        t["grupo"] = list(grupos)
    for n in p["nombres"]:
        t[n.lower().replace(" ", "_")] = sent(p["preds"][n])
    t["votacion"] = sent(p["preds"][CONSENSO])
    t["acuerdo"] = np.where((p["suma"] == 0) | (p["suma"] == len(p["nombres"])), "unanime", "dividido")
    if p["proba"] is not None:
        t["prob_positivo_nb"] = p["proba"].round(3)
    if etiquetas is not None:
        y = pd.Series(list(etiquetas), dtype="float")
        t["real"] = np.where(y.isna(), "", np.where(y == 1, "positivo", "negativo"))
        t["acierto_votacion"] = np.where(y.isna(), "", np.where(y.to_numpy() == p["preds"][CONSENSO], "si", "no"))
    return t


def resumir_lote(textos, vectorizer, modelos, etiquetas=None, grupos=None, prediccion=None) -> dict:
    p = prediccion or predecir_lote(textos, vectorizer, modelos)
    textos, limpios, nombres, preds, suma = p["textos"], p["limpios"], p["nombres"], p["preds"], p["suma"]

    n = len(textos)
    r = {
        "n": n,
        "modelos": nombres + [CONSENSO],
        "pct_positivo_pred": {k: round(float(p.mean()), 4) for k, p in preds.items()},
        "acuerdo": {
            "unanime_positivo": int((suma == len(nombres)).sum()),
            "unanime_negativo": int((suma == 0).sum()),
            "dividido": int(((suma > 0) & (suma < len(nombres))).sum()),
        },
        "terminos": _terminos(limpios, preds[CONSENSO]),
    }
    r["acuerdo"]["pct_unanime"] = round((r["acuerdo"]["unanime_positivo"] + r["acuerdo"]["unanime_negativo"]) / n, 4)

    # Distribución de la probabilidad de "positivo" según Naive Bayes (el único con probabilidades calibradas)
    if p["proba"] is not None:
        proba = p["proba"]
        hist, _ = np.histogram(proba, bins=10, range=(0, 1))
        r["confianza"] = {"bins": [f"{i * 10}-{(i + 1) * 10}%" for i in range(10)], "conteo": hist.tolist(),
                          "pct_seguro": round(float(((proba < .2) | (proba > .8)).mean()), 4)}

    if etiquetas is not None:
        # Las filas sin etiqueta válida (neutrales o vacías) se excluyen solo de las métricas.
        y_todas = np.asarray(etiquetas, dtype=float)
        con = ~np.isnan(y_todas)
        y = y_todas[con].astype(int)
        r["n_etiquetados"] = int(con.sum())
        r["pct_positivo_real"] = round(float(y.mean()), 4)
        r["metricas"] = {k: _metricas(y, v[con]) for k, v in preds.items()}
        r["mejor_modelo"] = max(nombres, key=lambda k: r["metricas"][k]["f1"])

    if grupos is not None:
        g = pd.DataFrame({"grupo": list(grupos), "pred": preds[CONSENSO]})
        if etiquetas is not None:
            g["real"] = np.asarray(etiquetas, dtype=float)
        agg = g.groupby("grupo").agg(n=("pred", "size"), pct_pred=("pred", "mean"),
                                     **({"pct_real": ("real", "mean")} if etiquetas is not None else {}))
        agg = agg.sort_values("n", ascending=False).head(15).sort_values("pct_pred", ascending=False)
        r["grupos"] = [{"grupo": k, **{c: (int(v) if c == "n" else (None if pd.isna(v) else round(float(v), 4)))
                                       for c, v in fila.items()}}
                       for k, fila in agg.iterrows()]
    return r


# --- Fuentes de datos predefinidas -----------------------------------------
def fuente_evaluacion():
    """Tweets del split de prueba: los modelos no los vieron durante el entrenamiento."""
    m = pd.read_csv(SAMPLE_PATH)
    test = m[m["split"] == "test"]
    return test["text"], test["label"], None


def fuente_marcas():
    """Tweets etiquetados por personas y agrupados por marca o tema (sin neutrales)."""
    ruta = RAW_DIR / "testdata.manual.2009.06.14.csv"
    if not ruta.exists():
        return None
    m = pd.read_csv(ruta, encoding="latin-1", header=None, names=COLUMNS)
    m = m[m["target"] != 2]
    return m["text"], (m["target"] == 4).astype(int), m["flag"].str.strip('"')


FUENTE_PROPIA = "propios"

FUENTES = {
    "evaluacion": {"nombre": "Conjunto de evaluación", "cargar": fuente_evaluacion,
                   "descripcion": "Tweets reservados para evaluar. Los modelos no los vieron durante el entrenamiento."},
    "marcas": {"nombre": "Marcas y temas", "cargar": fuente_marcas,
               "descripcion": "Tweets sobre marcas y temas concretos, etiquetados manualmente por personas."},
    FUENTE_PROPIA: {"nombre": "Tus datos", "cargar": None,
                    "descripcion": "Analiza tus propios textos: pégalos o sube un archivo CSV."},
}

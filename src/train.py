import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    MAX_FEATURES,
    METRICS_PATH,
    MIN_DF,
    SUBLINEAR_TF,
    RESULTS_DIR,
    SAMPLE_PATH,
    MODEL_FILES,
    MODELS_DIR,
    NGRAM_RANGE,
    RANDOM_STATE,
    TEST_SIZE,
    VECTORIZER_PATH,
)
from src.dataset import preparar_muestra


def crear_vectorizador(stop_words=None) -> TfidfVectorizer:
    """TF-IDF del proyecto. No se eliminan stop words: con suficiente vocabulario,
    palabras como "not", "but" o "so" aportan señal de sentimiento."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words=stop_words,
        min_df=MIN_DF,
        sublinear_tf=SUBLINEAR_TF,
    )


def construir_modelos() -> dict:
    """Hiperparámetros elegidos por validación sobre el conjunto de entrenamiento
    (src/tuning.py), nunca mirando el test."""
    return {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "SVM": LinearSVC(C=0.25, max_iter=4000, random_state=RANDOM_STATE),
        # Random Forest no maneja bien cientos de miles de columnas dispersas:
        # se preseleccionan los 20 000 términos más asociados al sentimiento (χ²,
        # ajustado solo con train) y se dejan crecer los árboles sin podar, como
        # en la formulación original de Breiman; el bagging controla la varianza.
        "Random Forest": Pipeline([
            ("seleccion", SelectKBest(chi2, k=20_000)),
            ("bosque", RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def metricas_por_clase(cm) -> dict:
    """Precisión, recall y F1 de cada clase a partir de la matriz [[VN, FP], [FN, VP]].

    Con el test balanceado, los promedios ponderados casi coinciden con la
    accuracy (el recall ponderado es igual a la accuracy por definición), así
    que lo informativo es ver cada clase por separado.
    """
    (vn, fp), (fn, vp) = cm

    def clase(aciertos, falsos_alarma, perdidos):
        p = aciertos / (aciertos + falsos_alarma) if aciertos + falsos_alarma else 0.0
        r = aciertos / (aciertos + perdidos) if aciertos + perdidos else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    pos, neg = clase(vp, fp, fn), clase(vn, fn, fp)
    return {"positivo": pos, "negativo": neg, "f1_macro": round((pos["f1"] + neg["f1"]) / 2, 4)}


def puntuacion_positiva(modelo, X) -> np.ndarray:
    """Puntuación continua de la clase positiva: probabilidad (Naive Bayes, Random
    Forest) o distancia al hiperplano (SVM). La ROC solo necesita que ordene bien."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def metricas_roc(y_true, puntuacion, puntos: int = 101) -> dict:
    """AUC y curva ROC resumida en una rejilla fija de tasas de falsos positivos."""
    fpr, tpr, _ = roc_curve(y_true, puntuacion)
    rejilla = np.linspace(0, 1, puntos)
    return {
        "auc": round(float(roc_auc_score(y_true, puntuacion)), 4),
        "fpr": [round(float(v), 4) for v in rejilla],
        "tpr": [round(float(v), 4) for v in np.interp(rejilla, fpr, tpr)],
    }


def actualizar_roc() -> dict:
    """Añade la ROC a metrics.json con los modelos ya entrenados (sin reentrenar)."""
    from src.predict import cargar_artefactos

    vectorizer, modelos, resultados = cargar_artefactos()
    test = pd.read_csv(SAMPLE_PATH, usecols=["texto_limpio", "label", "split"])
    test = test[test["split"] == "test"]
    X = vectorizer.transform(test["texto_limpio"].fillna(""))
    for nombre, modelo in modelos.items():
        resultados["modelos"][nombre]["roc"] = metricas_roc(test["label"], puntuacion_positiva(modelo, X))
        print(f"  {nombre}: AUC={resultados['modelos'][nombre]['roc']['auc']}")
    METRICS_PATH.write_text(json.dumps(resultados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resultados


def metricas_modelo(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "matriz_confusion": cm.tolist(),
        "por_clase": metricas_por_clase(cm.tolist()),
    }


def entrenar() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Cargando y preprocesando muestra del dataset...")
    muestra = preparar_muestra()

    idx_train, idx_test = train_test_split(
        muestra.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=muestra["label"],
    )
    # Se marca cada fila para que dashboard / análisis temporal puedan
    # evaluar solo sobre datos que los modelos no vieron al entrenar.
    muestra["split"] = "train"
    muestra.loc[idx_test, "split"] = "test"
    muestra.to_csv(SAMPLE_PATH, index=False)

    X_train, y_train = muestra.loc[idx_train, "texto_limpio"], muestra.loc[idx_train, "label"]
    X_test, y_test = muestra.loc[idx_test, "texto_limpio"], muestra.loc[idx_test, "label"]

    print("Vectorizando con TF-IDF...")
    vectorizer = crear_vectorizador()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    resultados = {
        "n_muestra": int(len(muestra)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "modelos": {},
    }

    modelos = construir_modelos()
    for nombre, modelo in modelos.items():
        print(f"Entrenando {nombre}...")
        modelo.fit(X_train_vec, y_train)
        pred = modelo.predict(X_test_vec)
        resultados["modelos"][nombre] = metricas_modelo(y_test, pred)
        resultados["modelos"][nombre]["roc"] = metricas_roc(y_test, puntuacion_positiva(modelo, X_test_vec))
        joblib.dump(modelo, MODEL_FILES[nombre], compress=3)
        print(f"  accuracy={resultados['modelos'][nombre]['accuracy']}  f1={resultados['modelos'][nombre]['f1']}")

    METRICS_PATH.write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    print(f"Métricas guardadas en {METRICS_PATH}")

    from src.analysis import generar as generar_analisis  # import tardío: analysis usa predict → train
    generar_analisis()
    return resultados


def modelos_listos() -> bool:
    archivos = [VECTORIZER_PATH, METRICS_PATH, *MODEL_FILES.values()]
    return all(Path(p).exists() for p in archivos)


if __name__ == "__main__":
    if "--roc" in sys.argv:  # python -m src.train --roc
        actualizar_roc()
    else:
        entrenar()

# Sentinel Blue — Análisis de sentimiento en tweets

Proyecto final en equipo (3 personas) sobre el dataset **Sentiment140** (`training.1600000.processed.noemoticon.csv`): 1.6 millones de tweets etiquetados como negativo (`target = 0`) o positivo (`target = 4`).

Estilo visual **editorial claro**: fondo crema `#f7f3ec`, tinta `#1b1815`, positivo `#2a7f6f`, negativo `#d9573b`.
Tipografías: **Fraunces** (títulos), **Inter** (texto), **JetBrains Mono** (números). Todo está en
`static/css/style.css` como variables CSS; los gráficos usan **Apache ECharts** con el tema de `static/js/charts.js`.

Las fórmulas se renderizan con **KaTeX** (`\( ... \)` en las plantillas).
Las fuentes, ECharts y KaTeX están **alojados en el propio proyecto** (`static/fonts`, `static/vendor`). No se carga nada
de Google Fonts ni de CDNs, así que no hay rastreo de terceros y la app funciona sin internet (útil para la presentación).

---

## Cómo está dividido el trabajo

| Parte | Quién | Entregable |
| --- | --- | --- |
| **1** | Tú (lista en el código) | Preprocesamiento + Naive Bayes + SVM + Random Forest + Accuracy / F1 / matriz de confusión + app web base |
| **2** | Compañero 2 | Aplicación web de análisis de texto (completa) + dashboard de sentimiento |
| **3** | Compañero 3 | Análisis temporal (gráficos por fecha) |

---

## Parte 1 — hecha

### Qué incluye

- Limpieza de texto: entidades HTML, minúsculas, quitar URLs, menciones `@usuario`, signos y espacios de más. Los hashtags se convierten en la palabra (`#love` → `love`).
- Negaciones: se expanden las contracciones (`don't` → `do not`) y `not`/`no`/`never`... se conservan. No se eliminan stop words: palabras como `not`, `but` o `so` aportan señal de sentimiento (ver /analisis).
- Vectorización TF-IDF con unigramas y bigramas, sin tope de vocabulario (`min_df=2`) y `sublinear_tf`.
- Hiperparámetros elegidos por validación sobre el conjunto de entrenamiento (`python -m src.tuning`): SVM `C=0.25`, Naive Bayes `alpha=1`, Random Forest con 100 árboles sin podar sobre los 20 000 términos más informativos (χ²).
- Tres modelos de clasificación:
  - **Naive Bayes** (`MultinomialNB`)
  - **SVM** lineal (`LinearSVC`, adecuado para texto)
  - **Random Forest**
- Métricas en conjunto de prueba: **Accuracy**, **F1**, precision, recall y **matriz de confusión**.
- App Flask con:
  - Inicio (resumen de las 3 partes)
  - Página de métricas (con fórmulas e IC 95 %)
  - Página de análisis: hallazgos con su evidencia (McNemar, χ², kappa, Spearman, IC de Wilson)
  - Analizar un texto (clasificación con los 3 modelos y un consenso)
  - Rutas `/dashboard` y `/temporal` como marcadores para las partes 2 y 3

### Cómo ejecutarlo

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m src.download   # baja Sentiment140 (~80 MB) a data/raw/ — sin cuenta de Kaggle
python -m src.train      # si falta el dataset, lo descarga solo
python app.py
```

Para el notebook de análisis (gráficos de EDA, matrices de confusión, percepción por marca):

```bash
pip install -r requirements-notebook.txt
jupyter lab notebooks/01_eda_y_evaluacion.ipynb
```

El dataset de Kaggle es un espejo del zip original de Stanford; `src/download.py` baja ese zip directamente, así que no hacen falta credenciales.

### Estructura de carpetas

```
data/
  raw/         dataset original descargado (no se sube a git)
  processed/   muestra_entrenamiento.csv: texto limpio + etiqueta + fecha + split (train/test)
  results/     metrics.json y salidas de la app (predicciones, exports)
models/        modelos .joblib entrenados (no se suben a git)
src/analysis.py   hallazgos + pruebas estadísticas → data/results/analisis.json (página /analisis)
src/dashboard.py  resumen de un lote de textos (KPIs, métricas, acuerdo, términos) para /dashboard
src/lote.py       lectura de textos pegados o CSV del usuario (detección de columnas)
samples/       CSV de ejemplo para la pestaña «Tus datos» (ver samples/README.md)
notebooks/     01_eda_y_evaluacion.ipynb: análisis exploratorio + evaluación (para informe/presentación)
src/           pipeline: config, download, preprocessing, dataset, train, predict
templates/, static/   frontend Flask
```

Abre `http://127.0.0.1:5000`.

El entrenamiento usa una **muestra balanceada de 200 000 tweets** (80/20: 160 000 para entrenar y 40 000 para evaluar) y tarda unos 3 minutos, casi todo en Random Forest. Con más datos los modelos lineales mejoran poco (≈ 82 % con 640 000) y Random Forest crece a varios GB. El tamaño se cambia con `SAMPLE_SIZE` en `src/config.py`.

Al entrenar se generan:

- `data/processed/muestra_entrenamiento.csv` — muestra con texto original, texto limpio, etiqueta, **fecha** (la Parte 3 la necesita) y `split` (`train`/`test`: usar `test` para evaluar sin fuga de datos)
- `models/tfidf_vectorizer.joblib`
- `models/naive_bayes.joblib`, `models/svm.joblib`, `models/random_forest.joblib`
- `data/results/metrics.json`

### Archivos de la Parte 1

```
app.py
src/config.py
src/download.py
src/preprocessing.py
src/dataset.py
src/train.py
src/predict.py
templates/
static/css/style.css
```

---

## Parte 2 — completada

**`/dashboard`** con tres fuentes de datos:

- **Conjunto de evaluación** (40 000 tweets que los modelos no vieron) y **Marcas y temas** (test manual, etiquetado por personas).
- **Tus datos**: pegar textos (uno por línea) o subir un CSV de hasta 50 000 filas / 10 MB.
  Las columnas se reconocen por nombre (`src/lote.py`): texto (obligatoria), sentimiento, grupo y fecha (opcionales).
  También acepta el CSV original de Sentiment140. En «Ver datos de ejemplo» se pueden previsualizar, descargar y
  analizar los CSV de `samples/`, que cubren cada formato admitido.

Paneles: KPIs, distribución del sentimiento, rendimiento por modelo y matriz de confusión (si hay sentimiento real),
sentimiento por grupo, acuerdo entre modelos, % positivo por modelo, seguridad de la clasificación y términos
por sentimiento. Con datos propios, además: tabla por texto con filtros y búsqueda, y descarga de resultados en CSV
(predicción de cada modelo, votación, probabilidad, acierto y la fecha si venía en el archivo).

**`/analizar`**: clasificación de un texto individual con los tres modelos.

La lógica está en `src/dashboard.py` (`predecir_lote`, `resumir_lote`, `tabla_resultados`). Los análisis de datos
propios se guardan en memoria (los últimos 8) y se pierden al reiniciar la app.

---

## Parte 3 — completada

**`/temporal`**: **análisis temporal** usando la columna `date` del dataset (`Mon Apr 06 22:19:45 PDT 2009`,
parseada retirando la zona `PDT` antes de convertir a fecha).

### Qué incluye

- Granularidad elegible: **día, semana o mes**.
- Línea de tiempo: cantidad de tweets **positivos y negativos** por periodo (barras apiladas).
- **Ratio de positividad** por periodo (`positivos / total`), con la media del rango y la tendencia (al alza / a la baja / estable).
- **Filtro de rango de fechas** (el dataset es sobre todo abril–junio 2009).
- Fuente de datos: `data/processed/muestra_entrenamiento.csv` (ya trae `date`, `label`, `text`); sin modelos
  entrenados muestra las etiquetas reales, y con modelos permite ver la **predicción de cada modelo o del consenso**
  frente a la etiqueta real.
- Hallazgos: periodo **más positivo y más negativo** (con mínimo de 10 tweets), periodo de mayor volumen y un
  **recuadro de interpretación** (1–2 frases) para la presentación. Avisa de la cola 100 % negativa de junio,
  que se debe al cambio en la recolección del dataset (ver `/analisis`), no a un cambio de opinión.

La lógica está en `src/temporal.py` (`parsear_fecha`, `cargar_base`, `predecir_columna`, `resumir_temporal`)
y la vista en `templates/temporal.html` (ECharts con la identidad visual del proyecto).

---

## Columnas del CSV original

El archivo no trae cabecera. Las columnas son:

| Columna | Significado |
| --- | --- |
| `target` | 0 = negativo, 4 = positivo |
| `ids` | id del tweet |
| `date` | fecha y hora |
| `flag` | query (casi siempre `NO_QUERY`) |
| `user` | usuario |
| `text` | texto del tweet |

La Parte 1 guarda `label` como 0/1 (`1` = positivo).

---

## Requisitos cubiertos vs pendientes

| Requerimiento | Estado |
| --- | --- |
| Preprocesamiento de texto | Parte 1 lista |
| Naive Bayes, SVM, Random Forest | Parte 1 lista |
| Accuracy, F1, matriz de confusión | Parte 1 lista |
| Aplicación web de análisis de texto | Parte 2 lista (texto individual y lotes / CSV) |
| Dashboard de sentimiento | Parte 2 lista |
| Análisis temporal | Parte 3 lista |

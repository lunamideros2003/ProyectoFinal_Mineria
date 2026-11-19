import json
import uuid
from collections import OrderedDict
from datetime import date, timedelta

from flask import Flask, Response, abort, redirect, render_template, request, send_from_directory, url_for

from src.analysis import ANALISIS_PATH
from src.curva import CURVA_PATH
from src.curva import resumir as resumir_curva
from src.dashboard import FUENTE_PROPIA, FUENTES, predecir_lote, resumir_lote, tabla_resultados
from src.temporal import GRANULARIDADES, VISTAS, cambio_recoleccion, cargar_base, evaluacion, predecir, resumir_temporal
from src.config import SAMPLES_DIR
from src.lote import (EJEMPLOS, MAX_BYTES, MAX_FILAS, ErrorDatos, leer_csv, leer_ejemplo,
                      leer_texto_pegado, listar_ejemplos, preparar)
from src.predict import cargar_artefactos, predecir_texto
from src.train import metricas_por_clase, modelos_listos

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 1024 * 1024

# Análisis de datos propios en memoria: se conservan los últimos MAX_LOTES.
MAX_LOTES = 8
FILAS_TABLA = 5_000  # filas que se envían a la tabla del navegador; la descarga incluye todas
_lotes: OrderedDict = OrderedDict()

_cache = {"vectorizer": None, "modelos": None, "metricas": None, "dashboard": {}}

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def obtener_artefactos():
    if not modelos_listos():
        return None, None, None
    if _cache["vectorizer"] is None:
        _cache["vectorizer"], _cache["modelos"], _cache["metricas"] = cargar_artefactos()
        for m in (_cache["metricas"] or {}).get("modelos", {}).values():  # metrics.json anteriores
            m.setdefault("por_clase", metricas_por_clase(m["matriz_confusion"]))
    return _cache["vectorizer"], _cache["modelos"], _cache["metricas"]


def cargar_analisis():
    if not ANALISIS_PATH.exists():
        return None
    return json.loads(ANALISIS_PATH.read_text(encoding="utf-8"))


# --- Filtros de formato para las plantillas --------------------------------
@app.template_filter("pct")
def fmt_pct(valor, decimales=1):
    return f"{valor * 100:.{decimales}f}\u00a0%"


@app.template_filter("miles")
def fmt_miles(valor):
    return f"{valor:,}".replace(",", "\u00a0")  # espacio no separable: "160 000" no se parte


@app.template_filter("fecha")
def fmt_fecha(iso, desplazamiento_dias=0, con_anio=True):
    """'2009-06-17' -> '17 de junio de 2009' (desplazamiento opcional en días)."""
    f = date.fromisoformat(iso) + timedelta(days=desplazamiento_dias)
    texto = f"{f.day} de {MESES[f.month - 1]}"
    return f"{texto} de {f.year}" if con_anio else texto


@app.template_filter("pval")
def fmt_pval(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


@app.context_processor
def contexto_global():
    return {"listos": modelos_listos(), "ruta": request.endpoint}


# --- Rutas -----------------------------------------------------------------
@app.route("/")
def inicio():
    _, _, datos = obtener_artefactos()
    return render_template("index.html", datos=datos, analisis=cargar_analisis())


@app.route("/metricas")
def metricas():
    _, _, datos = obtener_artefactos()
    return render_template("metricas.html", datos=datos, analisis=cargar_analisis())


@app.route("/analisis")
def analisis():
    a = cargar_analisis()
    if a is not None and CURVA_PATH.exists():
        a["curva"] = json.loads(CURVA_PATH.read_text(encoding="utf-8"))
        a["curva"]["resumen"] = resumir_curva(a["curva"])
    return render_template("analisis.html", a=a)


@app.route("/analizar", methods=["GET", "POST"])
def analizar():
    resultado = None
    error = None
    texto = ""
    if request.method == "POST":
        texto = (request.form.get("texto") or "").strip()
        if not modelos_listos():
            error = "Los modelos todavía no están disponibles."
        elif not texto:
            error = "Escribe un texto para analizar."
        else:
            vectorizer, modelos, _ = obtener_artefactos()
            resultado = predecir_texto(texto, vectorizer, modelos)
    return render_template("analizar.html", resultado=resultado, error=error, texto=texto)


@app.route("/dashboard")
def dashboard():
    fuente = request.args.get("fuente", "evaluacion")
    if fuente not in FUENTES:
        abort(404)
    if fuente == FUENTE_PROPIA:
        lote = _lotes.get(request.args.get("lote", ""))
        if lote is None:
            return render_template("dashboard.html", r=None, fuente=fuente, fuentes=FUENTES,
                                   max_filas=MAX_FILAS, max_mb=MAX_BYTES // 2**20, ejemplos=listar_ejemplos())
        _lotes.move_to_end(request.args["lote"])
        filas = lote["tabla"].head(FILAS_TABLA).fillna("")
        return render_template("dashboard.html", r=lote["resumen"], fuente=fuente, fuentes=FUENTES,
                               lote=lote, lote_id=request.args["lote"],
                               filas=filas.to_dict(orient="records"), columnas=list(filas.columns))
    resumen = None
    if modelos_listos():
        if fuente not in _cache["dashboard"]:
            datos = FUENTES[fuente]["cargar"]()
            if datos is not None:
                textos, etiquetas, grupos = datos
                vectorizer, modelos, _ = obtener_artefactos()
                _cache["dashboard"][fuente] = resumir_lote(textos, vectorizer, modelos, etiquetas, grupos)
        resumen = _cache["dashboard"].get(fuente)
    return render_template("dashboard.html", r=resumen, fuente=fuente, fuentes=FUENTES)


@app.route("/dashboard/propios", methods=["POST"])
def dashboard_propios():
    def error(mensaje, codigo=400):
        return render_template("dashboard.html", r=None, fuente=FUENTE_PROPIA, fuentes=FUENTES, error=mensaje,
                               texto=request.form.get("texto", ""), max_filas=MAX_FILAS,
                               max_mb=MAX_BYTES // 2**20, ejemplos=listar_ejemplos()), codigo

    if not modelos_listos():
        return error("Los modelos todavía no están disponibles.", 503)
    archivo = request.files.get("archivo")
    texto = (request.form.get("texto") or "").strip()
    ejemplo = request.form.get("ejemplo")
    try:
        if ejemplo:
            datos, info = leer_csv(leer_ejemplo(ejemplo))
            info["nombre"] = ejemplo
        elif archivo and archivo.filename:
            datos, info = leer_csv(archivo.read())
            info["nombre"] = archivo.filename
        elif texto:
            datos, info = leer_texto_pegado(texto)
            info["nombre"] = "Texto pegado"
        else:
            return error("Pega algunos textos o selecciona un archivo CSV.")
        datos, info = preparar(datos, info)
    except ErrorDatos as e:
        return error(str(e))

    vectorizer, modelos, _ = obtener_artefactos()
    etiquetas = datos["etiqueta"] if "etiqueta" in datos else None
    grupos = datos["grupo"] if "grupo" in datos else None
    pred = predecir_lote(datos["texto"], vectorizer, modelos)
    lote_id = uuid.uuid4().hex[:12]
    _lotes[lote_id] = {
        "info": info,
        "resumen": resumir_lote(None, vectorizer, modelos, etiquetas, grupos, prediccion=pred),
        "tabla": tabla_resultados(pred, etiquetas, grupos, datos["fecha"] if "fecha" in datos else None),
    }
    while len(_lotes) > MAX_LOTES:
        _lotes.popitem(last=False)
    return redirect(url_for("dashboard", fuente=FUENTE_PROPIA, lote=lote_id))


@app.route("/dashboard/propios/<lote_id>.csv")
def descargar_resultados(lote_id):
    lote = _lotes.get(lote_id)
    if lote is None:
        abort(404)
    return Response(lote["tabla"].to_csv(index=False).encode("utf-8-sig"), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sentimiento_resultados.csv"})


@app.route("/dashboard/ejemplos/<archivo>")
def descargar_ejemplo(archivo):
    if archivo not in EJEMPLOS:
        abort(404)
    return send_from_directory(SAMPLES_DIR, archivo, as_attachment=True, mimetype="text/csv")


@app.errorhandler(413)
def demasiado_grande(_):
    return render_template("dashboard.html", r=None, fuente=FUENTE_PROPIA, fuentes=FUENTES,
                           error=f"El archivo supera el máximo de {MAX_BYTES // 2**20} MB.",
                           max_filas=MAX_FILAS, max_mb=MAX_BYTES // 2**20, ejemplos=listar_ejemplos()), 413


def _fecha_valida(valor: str | None) -> str | None:
    """'2009-05-01' si es una fecha ISO válida; None en cualquier otro caso."""
    try:
        return date.fromisoformat((valor or "").strip()).isoformat()
    except ValueError:
        return None


@app.route("/temporal")
def temporal():
    gran = request.args.get("granularidad", "dia")
    gran = gran if gran in GRANULARIDADES else "dia"
    vista = request.args.get("vista", "real")
    vista = vista if vista in VISTAS else "real"
    desde, hasta = _fecha_valida(request.args.get("desde")), _fecha_valida(request.args.get("hasta"))
    if desde and hasta and desde > hasta:
        desde, hasta = hasta, desde

    if "temporal" not in _cache:
        base = cargar_base()
        _cache["temporal"] = {"base": base, "corte": None if base is None else cambio_recoleccion(base),
                              "evaluacion": None if base is None else evaluacion(base), "pred": {}}
    t = _cache["temporal"]
    contexto = {"gran": gran, "desde": desde, "hasta": hasta, "aviso": None}
    if t["base"] is None:
        return render_template("temporal.html", hay_datos=False, r=None, vista="real", vistas=["real"], **contexto)

    vistas = list(VISTAS) if modelos_listos() else ["real"]
    if vista not in vistas:
        contexto["aviso"] = "Los modelos todavía no están disponibles. Se muestran las etiquetas reales."
        vista = "real"
    if vista == "real":
        base, prediccion = t["base"], None
    else:
        # Solo tweets de evaluación: predecir sobre los de entrenamiento inflaría el acuerdo con la etiqueta real.
        base = t["evaluacion"]
        if vista not in t["pred"]:
            vectorizer, modelos, _ = obtener_artefactos()
            t["pred"][vista] = predecir(base, vectorizer, modelos, vista)
        prediccion = t["pred"][vista]
    r = resumir_temporal(base, gran, desde, hasta, prediccion, t["corte"])
    return render_template("temporal.html", hay_datos=True, r=r, vista=vista, vistas=vistas,
                           corte=None if t["corte"] is None else t["corte"].strftime("%Y-%m-%d"),
                           corte_hora=None if t["corte"] is None else t["corte"].strftime("%H:%M"),
                           min_fecha=t["base"]["fecha"].min().strftime("%Y-%m-%d"),
                           max_fecha=t["base"]["fecha"].max().strftime("%Y-%m-%d"),
                           n_total=len(t["base"]), n_evaluacion=len(t["evaluacion"]), **contexto)


if __name__ == "__main__":
    app.run(debug=True)

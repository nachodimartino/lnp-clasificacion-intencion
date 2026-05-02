"""
SUITE DE TESTS — Servicio de Clasificación NLP (Modelo A)
Correr con: python test_nlp_service.py
La API debe estar en localhost:8000 (o cambiar BASE_URL).
"""
 
import requests
import sys
 
BASE_URL = "http://localhost:8000"
PASS = 0
FAIL = 0
 
CLASES_VALIDAS = {
    "consulta_precio", "consulta_propiedad", "desistimiento",
    "intencion_compra_alquiler", "interes_avanzar",
    "saludo_consulta", "solicitar_visita"
}
 
PESOS = {
    "saludo_consulta": 0.10, "consulta_propiedad": 0.30,
    "consulta_precio": 0.45, "solicitar_visita": 0.65,
    "interes_avanzar": 0.75, "intencion_compra_alquiler": 0.90,
    "desistimiento": -0.20,
}
 
 
def check(nombre, condicion, detalle=""):
    global PASS, FAIL
    if condicion:
        print(f"  ✓ {nombre}")
        PASS += 1
    else:
        print(f"  ✗ {nombre}  ← {detalle}")
        FAIL += 1
 
 
def post(texto):
    r = requests.post(f"{BASE_URL}/clasificar", json={"texto": texto})
    return r.status_code, r.json() if r.status_code in (200, 400, 422, 500) else {}
 
 
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1 — Health check y estructura del servicio")
print("=" * 60)
r = requests.get(f"{BASE_URL}/health")
check("GET /health responde 200", r.status_code == 200)
data = r.json()
check("Tiene campo 'clases'",   "clases"   in data)
check("Tiene campo 'n_clases'", "n_clases" in data)
check("n_clases == 7",          data.get("n_clases") == 7,
      f"n_clases={data.get('n_clases')}")
check("Las 7 clases están presentes",
      set(data.get("clases", [])) == CLASES_VALIDAS,
      f"clases={data.get('clases')}")
 
r2 = requests.get(f"{BASE_URL}/clases")
check("GET /clases responde 200", r2.status_code == 200)
 
 
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 2 — Estructura de la respuesta")
print("=" * 60)
status, resp = post("hola buenas, me pasas info del depa?")
check("HTTP 200",                      status == 200)
check("Tiene 'clase_predicha'",        "clase_predicha"   in resp)
check("Tiene 'confianza'",             "confianza"        in resp)
check("Tiene 'intencion_compra'",      "intencion_compra" in resp)
check("Tiene 'probabilidades'",        "probabilidades"   in resp)
check("Tiene 'advertencia'",           "advertencia"      in resp)  # puede ser null
check("clase_predicha es clase válida",
      resp.get("clase_predicha") in CLASES_VALIDAS,
      f"clase={resp.get('clase_predicha')}")
check("confianza en [0, 1]",
      0.0 <= resp.get("confianza", -1) <= 1.0,
      f"confianza={resp.get('confianza')}")
check("intencion_compra en [0, 1]",
      0.0 <= resp.get("intencion_compra", -1) <= 1.0,
      f"intencion={resp.get('intencion_compra')}")
check("probabilidades tiene 7 clases",
      len(resp.get("probabilidades", {})) == 7,
      f"n={len(resp.get('probabilidades', {}))}")
check("probabilidades suman ≈ 1.0",
      abs(sum(resp.get("probabilidades", {}).values()) - 1.0) < 0.01,
      f"suma={sum(resp.get('probabilidades', {}).values()):.4f}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 3 — Clasificación semántica por clase")
print("=" * 60)
# Mensajes claros para cada clase — deben clasificarse correctamente
casos = [
    ("saludo_consulta",           "hola buenas tardes, tienen propiedades?"),
    ("saludo_consulta",           "buen dia me pueden dar info?"),
    ("consulta_propiedad",        "tiene cochera ese departamento?"),
    ("consulta_propiedad",        "cuantos ambientes son?"),
    ("consulta_propiedad",        "acepta mascotas el edificio?"),
    ("consulta_precio",           "cuanto sale ese depa?"),
    ("consulta_precio",           "cuanto son las expensas?"),
    ("consulta_precio",           "el precio es negociable?"),
    ("solicitar_visita",          "puedo ir a verlo el sabado?"),
    ("solicitar_visita",          "me dan un turno para visitar la casa?"),
    ("interes_avanzar",           "me interesa pero tengo que hablarlo con mi señora"),
    ("interes_avanzar",           "me convence pero dependo del credito"),
    ("intencion_compra_alquiler", "quiero señar ese depa, como hago?"),
    ("intencion_compra_alquiler", "quiero alquilar esa propiedad, que necesito?"),
    ("desistimiento",             "ya no me interesa, consegui algo por otro lado"),
    ("desistimiento",             "lo dejamos, no nos convence"),
]
 
aciertos = 0
for clase_esperada, mensaje in casos:
    _, r = post(mensaje)
    clase_pred = r.get("clase_predicha", "")
    ok = clase_pred == clase_esperada
    if ok:
        aciertos += 1
    check(
        f"'{mensaje[:45]}...' → {clase_esperada}",
        ok,
        f"predijo: {clase_pred}"
    )
 
pct = aciertos / len(casos) * 100
print(f"\n  Precisión en casos claros: {aciertos}/{len(casos)} ({pct:.0f}%)")
check(f"Precisión ≥ 75% en casos claros", pct >= 75,
      f"precisión={pct:.0f}%")
 
 
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 4 — Jerarquía de intención de compra")
print("=" * 60)
# La intención de compra debe crecer según el funnel de ventas
mensajes_por_etapa = [
    ("saludo_consulta",           "hola, tienen propiedades disponibles?"),
    ("consulta_propiedad",        "tiene pileta el edificio?"),
    ("consulta_precio",           "cuanto sale ese departamento?"),
    ("solicitar_visita",          "cuando puedo ir a verlo?"),
    ("interes_avanzar",           "me interesa, como sigo?"),
    ("intencion_compra_alquiler", "quiero señar, que necesito?"),
]
 
intentiones = []
print(f"  {'Clase':<30} {'Intencion':>10} {'Clase pred':>25}")
print("  " + "-"*68)
for clase_esperada, msg in mensajes_por_etapa:
    _, r = post(msg)
    intent = r.get("intencion_compra", 0)
    intentiones.append(intent)
    print(f"  {clase_esperada:<30} {intent:>10.4f} {r.get('clase_predicha',''):>25}")
 
# Verificar tendencia general creciente (con tolerancia para ambigüedad)
# No exigimos monotonicidad perfecta, pero sí que saludo < consulta < intención
check("saludo_consulta < intencion_compra_alquiler",
      intentiones[0] < intentiones[-1],
      f"{intentiones[0]:.4f} >= {intentiones[-1]:.4f}")
check("promedio primeras 3 < promedio últimas 3",
      sum(intentiones[:3])/3 < sum(intentiones[3:])/3,
      f"{sum(intentiones[:3])/3:.4f} >= {sum(intentiones[3:])/3:.4f}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 5 — Desistimiento baja la intención")
print("=" * 60)
_, r_neutral  = post("hola, tienen propiedades?")
_, r_interesa = post("me interesa mucho ese depa")
_, r_desiste  = post("ya no me interesa, me fui a otra inmobiliaria")
 
intent_neutral  = r_neutral.get("intencion_compra", 0)
intent_interesa = r_interesa.get("intencion_compra", 0)
intent_desiste  = r_desiste.get("intencion_compra", 0)
 
print(f"  saludo neutral:  {intent_neutral:.4f}")
print(f"  me interesa:     {intent_interesa:.4f}")
print(f"  desistimiento:   {intent_desiste:.4f}")
 
check("desistimiento < saludo neutral",
      intent_desiste < intent_neutral,
      f"{intent_desiste:.4f} >= {intent_neutral:.4f}")
check("me interesa > saludo neutral",
      intent_interesa > intent_neutral,
      f"{intent_interesa:.4f} <= {intent_neutral:.4f}")
check("intencion_compra desistimiento ≥ 0 (está clipeado)",
      intent_desiste >= 0.0,
      f"intencion={intent_desiste:.4f}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 6 — Registro informal (como habla un lead real)")
print("=" * 60)
# El modelo debe funcionar con vocabulario rioplatense informal
casos_informales = [
    ("consulta_precio",           "cuanto sale eso?"),
    ("consulta_precio",           "q precio tiene ese depa"),
    ("consulta_propiedad",        "tiene cochera ese dpto?"),
    ("solicitar_visita",          "puedo ir a verlo?"),
    ("intencion_compra_alquiler", "kiero señar ese depa"),
    ("desistimiento",             "ya consegui algo xq gracias igual"),
]
aciertos_inf = 0
for clase_esperada, mensaje in casos_informales:
    _, r = post(mensaje)
    clase_pred = r.get("clase_predicha", "")
    ok = clase_pred == clase_esperada
    if ok:
        aciertos_inf += 1
    check(f"'{mensaje}' → {clase_esperada}", ok, f"predijo: {clase_pred}")
 
pct_inf = aciertos_inf / len(casos_informales) * 100
check(f"Precisión ≥ 60% en registro informal", pct_inf >= 60,
      f"precisión={pct_inf:.0f}%")
 
 
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 7 — Validación de inputs inválidos")
print("=" * 60)
invalidos = [
    ("texto vacío",           ""),
    ("solo espacios",         "   "),
    ("un solo carácter",      "a"),
    ("texto de 501 chars",    "x" * 501),
    ("texto de 1000 chars",   "hola quiero info " * 60),
]
for nombre, texto in invalidos:
    r = requests.post(f"{BASE_URL}/clasificar", json={"texto": texto})
    check(f"'{nombre}' devuelve 4xx",
          r.status_code in (400, 422),
          f"status={r.status_code}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 8 — Robustez con inputs válidos pero difíciles")
print("=" * 60)
dificiles = [
    "ok",
    "si",
    "bueno",
    "dale",
    "perfecto gracias",
    "...",
    "123",
    "jajajaja",
    "hola?? 👋",
    "me pasas info del 3B que vi en zonaprop",
]
for texto in dificiles:
    s, r = post(texto)
    check(f"'{texto}' no da error 500",
          s == 200,
          f"status={s} detail={r.get('detail','')}")
    if s == 200:
        check(f"  → intencion en [0,1]",
              0.0 <= r.get("intencion_compra", -1) <= 1.0,
              f"intencion={r.get('intencion_compra')}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 9 — Confianza baja genera advertencia")
print("=" * 60)
# Mensajes muy cortos o ambiguos deben tener confianza baja y activar advertencia
casos_ambiguos = ["ok", "si", "dale", "bueno"]
for texto in casos_ambiguos:
    s, r = post(texto)
    if s == 200:
        confianza = r.get("confianza", 1.0)
        advertencia = r.get("advertencia")
        if confianza < 0.40:
            check(f"'{texto}' con confianza baja tiene advertencia",
                  advertencia is not None,
                  f"confianza={confianza:.2f} advertencia={advertencia}")
        else:
            print(f"  → '{texto}' confianza={confianza:.2f} (alta, no requiere advertencia)")
 
 
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 10 — Estabilidad (mismo input → mismo output)")
print("=" * 60)
texto_test = "quiero señar ese departamento cuanto necesito de seña?"
resultados = []
for _ in range(3):
    _, r = post(texto_test)
    resultados.append((r.get("clase_predicha"), r.get("intencion_compra")))
 
check("3 llamadas iguales dan la misma clase",
      len(set(x[0] for x in resultados)) == 1,
      f"clases={[x[0] for x in resultados]}")
check("3 llamadas iguales dan la misma intención",
      len(set(x[1] for x in resultados)) == 1,
      f"intenciones={[x[1] for x in resultados]}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 11 — Consistencia con el pipeline de scoring")
print("=" * 60)
# Los valores que devuelve este servicio van a alimentar el Modelo B.
# Verificamos que los rangos son compatibles con lo que espera el scoring service.
mensajes_pipeline = [
    "hola me pasas info",
    "cuanto sale el depa de 3 ambientes?",
    "puedo ir a verlo mañana?",
    "quiero señar ya, como hago?",
]
for msg in mensajes_pipeline:
    _, r = post(msg)
    intencion = r.get("intencion_compra", -1)
    confianza = r.get("confianza", -1)
    probs = r.get("probabilidades", {})
 
    check(f"intencion ∈ [0,1] para '{msg[:30]}'",
          0.0 <= intencion <= 1.0,
          f"intencion={intencion}")
    check(f"confianza ∈ [0,1] para '{msg[:30]}'",
          0.0 <= confianza <= 1.0,
          f"confianza={confianza}")
    check(f"probabilidades suma ≈ 1 para '{msg[:30]}'",
          abs(sum(probs.values()) - 1.0) < 0.01 if probs else False,
          f"suma={sum(probs.values()):.4f}" if probs else "vacío")
 
 
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
total = PASS + FAIL
print(f"  Pasaron: {PASS}/{total}")
print(f"  Fallaron: {FAIL}/{total}")
 
if FAIL == 0:
    print("  ✓ LISTO PARA PRODUCCIÓN")
elif FAIL <= 3:
    print("  ⚠ Casi listo — revisar los tests que fallaron")
    print("    (pueden ser casos borderline del modelo NLP, no bugs del servicio)")
else:
    print("  ✗ HAY PROBLEMAS — revisar antes de desplegar")
 
print()
print("NOTA sobre TEST 3 y 6:")
print("  Si fallan clasificaciones semánticas, el problema es el modelo NLP")
print("  (dataset de entrenamiento), no el servicio. El servicio es correcto")
print("  si los rangos y la estructura de respuesta son válidos.")
 
sys.exit(0 if FAIL == 0 else 1)
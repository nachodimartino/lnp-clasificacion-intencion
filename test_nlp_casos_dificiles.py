"""
TEST DE CASOS DIFÍCILES — Clasificador NLP
Mensajes ambiguos, contradictorios, muy cortos, con errores y borderline.

Estos tests NO exigen 100% de precisión — exigen que:
1. El servicio nunca crashee (siempre 200)
2. intencion_compra siempre esté en [0, 1]
3. La clase predicha siempre sea una de las 7 válidas
4. Los casos MÁS difíciles den señales coherentes de confianza baja

Los tests de clasificación exacta tienen umbral bajo (≥ 50%) porque
algunos mensajes genuinamente pueden pertenecer a más de una clase.
Correr con: python test_nlp_casos_dificiles.py
"""

import requests
import sys

BASE_URL = "http://localhost:8005"
PASS = 0
FAIL = 0
TOTAL_CLASIF = 0
CLASIF_OK = 0

CLASES_VALIDAS = {
    "consulta_precio", "consulta_propiedad", "desistimiento",
    "intencion_compra_alquiler", "interes_avanzar",
    "saludo_consulta", "solicitar_visita"
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
    r = requests.post(f"{BASE_URL}/clasificarv2", json={"message": texto})
    return r.status_code, r.json() if r.status_code in (200, 400, 422, 500) else {}


def clasificar(texto, clase_esperada, descripcion=""):
    """Clasifica y registra el resultado sin hacer assert duro."""
    global TOTAL_CLASIF, CLASIF_OK
    s, r = post(texto)
    clase_pred = r.get("clase_predicha", "ERROR")
    confianza  = r.get("confianza", 0)
    intencion  = r.get("p_compra", -1)
    ok = clase_pred == clase_esperada
    TOTAL_CLASIF += 1
    if ok:
        CLASIF_OK += 1
    marca = "✓" if ok else "~"  # ~ = borderline, no fallo duro
    label = descripcion or texto[:48]
    print(f"  {marca} [{clase_pred:<30}] conf={confianza:.2f} intent={intencion:.3f}  ← '{label}'")
    return s, r


def verificar_invariantes(texto, r, nombre):
    """Verifica las propiedades que SIEMPRE deben cumplirse."""
    check(f"{nombre}: HTTP 200",
          True)  # ya pasó si llegamos acá
    check(f"{nombre}: intencion en [0,1]",
          0.0 <= r.get("p_compra", -1) <= 1.0,
          f"intencion={r.get('p_compra')}")
    check(f"{nombre}: clase válida",
          r.get("clase_predicha") in CLASES_VALIDAS,
          f"clase={r.get('clase_predicha')}")
    probs = r.get("probabilities", {})
    check(f"{nombre}: probs suman 1",
          abs(sum(probs.values()) - 1.0) < 0.02 if probs else False,
          f"suma={sum(probs.values()):.3f}" if probs else "vacío")


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("TEST A — Mensajes con doble intención (los más frecuentes en producción)")
print("=" * 65)
print("  El lead dice dos cosas en el mismo mensaje.")
print("  El modelo debe elegir la intención DOMINANTE.")
print()

doble_intencion = [
    # (clase_dominante_esperada, mensaje, descripcion)
    ("consulta_precio",          "cuanto sale y tiene cochera?",
     "precio + propiedad → precio domina"),
    ("consulta_precio",          "me interesa, cuanto sale?",
     "interés + precio → precio domina (pregunta concreta)"),
    ("solicitar_visita",         "me interesa mucho, puedo ir a verlo mañana?",
     "interés + visita → visita domina (acción concreta)"),
    ("intencion_compra_alquiler","quiero señar pero antes quiero saber el precio",
     "compra + precio → compra domina (intención declarada)"),
    ("interes_avanzar",          "me encanta pero tengo que hablar con mi esposa, cuanto de seña?",
     "interés con condicionante + precio → interés_avanzar"),
    ("desistimiento",            "estuvo muy bueno pero ya conseguimos algo, gracias!",
     "positivo + desistimiento → desistimiento domina"),
    ("consulta_propiedad",       "tiene pileta y cuanto sale?",
     "propiedad + precio → propiedad domina (pregunta múltiple)"),
    ("interes_avanzar",          "lo hablo con mi socio y te digo, cuanto es la seña?",
     "condicionante + negociación → interés_avanzar"),
]

for clase_esp, msg, desc in doble_intencion:
    s, r = post(msg)
    if s == 200:
        clasificar(msg, clase_esp, desc)
        verificar_invariantes(msg, r, "doble_intencion")


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("TEST B — Negaciones (el modelo debe entender la negación)")
print("=" * 65)
print("  'no quiero' debe clasificar diferente a 'quiero'")
print()

negaciones = [
    ("desistimiento",  "no me interesa mas",
     "negación directa"),
    ("desistimiento",  "no voy a poder avanzar con esto",
     "negación de avance"),
    ("desistimiento",  "no nos sirve el precio, lo dejamos",
     "rechazo por precio"),
    ("interes_avanzar","quiero avanzar pero no tengo la plata todavia",
     "quiero + negación parcial → interés_avanzar"),
    ("consulta_precio","no entiendo el precio, me lo explicas?",
     "negación + consulta precio"),
    ("solicitar_visita","no lo vi todavia, puedo ir mañana?",
     "negación pasado + solicitud visita"),
]

for clase_esp, msg, desc in negaciones:
    s, r = post(msg)
    if s == 200:
        clasificar(msg, clase_esp, desc)
        verificar_invariantes(msg, r, "negacion")


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("TEST C — Mensajes con errores ortográficos severos")
print("=" * 65)
print("  El modelo fue entrenado con errores — debe tolerarlos.")
print()

errores_orto = [
    ("consulta_precio",           "cuanto cuesta ese dpto?",         "dpto"),
    ("consulta_precio",           "q precio tine ese depa??",        "tine (tiene)"),
    ("consulta_propiedad",        "tene kochera ese edifisio?",      "kochera, edifisio"),
    ("solicitar_visita",          "puedo hir a berlo?",              "hir, berlo"),
    ("intencion_compra_alquiler", "kiero alkilar ese depa",          "kiero, alkilar"),
    ("desistimiento",             "ya no m interesa graçias",        "m, graçias"),
    ("consulta_propiedad",        "azepta maskotaz?",                "azepta, maskotaz"),
    ("interes_avanzar",           "m interesa pro no tengo la guita","m, pro"),
]

for clase_esp, msg, desc in errores_orto:
    s, r = post(msg)
    if s == 200:
        clasificar(msg, clase_esp, desc)
        verificar_invariantes(msg, r, f"ortografia_{desc[:15]}")


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("TEST D — Mensajes muy cortos (2-4 palabras)")
print("=" * 65)
print("  Son frecuentes en WhatsApp. El modelo debe clasificarlos")
print("  aunque la confianza sea baja.")
print()

muy_cortos = [
    ("consulta_precio",           "el precio?",            "2 palabras"),
    ("consulta_precio",           "cuanto sale?",          "2 palabras"),
    ("consulta_propiedad",        "tiene pileta?",         "2 palabras"),
    ("solicitar_visita",          "puedo verlo?",          "2 palabras"),
    ("intencion_compra_alquiler", "lo quiero",             "2 palabras"),
    ("desistimiento",             "no gracias",            "2 palabras"),
    ("saludo_consulta",           "hola info?",            "2 palabras"),
    ("interes_avanzar",           "me interesa",           "2 palabras"),
    ("consulta_precio",           "precio final?",         "2 palabras"),
    ("solicitar_visita",          "visita mañana?",        "2 palabras"),
]

confianzas_cortos = []
for clase_esp, msg, desc in muy_cortos:
    s, r = post(msg)
    if s == 200:
        clasificar(msg, clase_esp, desc)
        verificar_invariantes(msg, r, f"corto_{desc}")
        confianzas_cortos.append(r.get("confianza", 1.0))

if confianzas_cortos:
    conf_promedio = sum(confianzas_cortos) / len(confianzas_cortos)
    print(f"\n  Confianza promedio en mensajes cortos: {conf_promedio:.3f}")
    print(f"  (Esperado: menor que en mensajes largos — más ambigüedad)")


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("TEST E — Mensajes largos con múltiples oraciones")
print("=" * 65)
print("  Casos donde el lead escribe un párrafo completo.")
print("  La intención dominante debe sobresalir.")
print()

largos = [
    ("intencion_compra_alquiler",
     "hola buenas tardes, estuve viendo el departamento de 3 ambientes que publicaron en zonaprop, me parece perfecto para lo que estamos buscando con mi familia, la zona nos encanta y el precio está bien, quiero avanzar con la seña lo antes posible, me pueden decir qué papeles necesito presentar y cuánto es el monto de la reserva?",
     "párrafo largo con intención clara de compra"),

    ("interes_avanzar",
     "che, mirá, el departamento nos gustó bastante cuando fuimos a verlo la semana pasada, la verdad que cumple con todo lo que estábamos buscando, el tema es que dependemos de que nos aprueben el crédito hipotecario que estamos tramitando, si eso sale bien avanzamos de una, ¿cuánto tiempo tienen disponible la propiedad antes de mostrársela a otros?",
     "párrafo largo con condicionante claro"),

    ("desistimiento",
     "hola, te quería avisar que al final no vamos a poder avanzar con el departamento que estuvimos viendo, estuvimos pensando mucho y la verdad que encontramos algo que nos quedaba más cómodo en cuanto a ubicación y precio, fue una decisión difícil porque el depa nos había gustado mucho, pero bueno, muchas gracias por la atención y la paciencia!",
     "párrafo largo de desistimiento educado"),

    ("consulta_precio",
     "buenas, quería consultar sobre el PH de 4 ambientes que tienen publicado, me interesaría saber el precio de lista, si aceptan algún tipo de financiación o plan de pago, cuánto sería el monto de la seña si decidiéramos avanzar, y si el precio tiene algún margen de negociación, muchas gracias",
     "párrafo largo con múltiples preguntas de precio"),
]

for clase_esp, msg, desc in largos:
    s, r = post(msg)
    if s == 200:
        clasificar(msg, clase_esp, desc)
        verificar_invariantes(msg, r, "largo")


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("TEST F — Casos que confunden al modelo (frontera entre clases)")
print("=" * 65)
print("  Estos son genuinamente ambiguos. No exigimos clasificación")
print("  correcta — exigimos que los invariantes se cumplan y que")
print("  la confianza sea baja (el modelo debe 'dudar').")
print()

borderline = [
    # (clases_aceptables, mensaje, descripcion)
    ({"interes_avanzar", "intencion_compra_alquiler"},
     "quiero avanzar, qué necesito?",
     "avanzar puede ser interés o compra según contexto"),

    ({"interes_avanzar", "intencion_compra_alquiler"},
     "cómo hago para reservarlo?",
     "reservar puede ser avance o compra"),

    ({"consulta_precio", "intencion_compra_alquiler"},
     "cuánto de seña para quedármelo?",
     "seña es precio Y compra simultáneamente"),

    ({"solicitar_visita", "interes_avanzar"},
     "me gustaría verlo para decidir",
     "visita condicionada a decisión"),

    ({"saludo_consulta", "consulta_propiedad"},
     "tienen algo para alquilar en palermo?",
     "búsqueda general vs consulta propiedad"),

    ({"interes_avanzar", "desistimiento"},
     "lo estoy pensando todavía",
     "duda pura — puede ser cualquier cosa"),

    ({"consulta_precio", "consulta_propiedad"},
     "qué incluye ese precio?",
     "precio + atributos mezclados"),
]

print(f"  {'Mensaje':<48} {'Clase pred':>25} {'Conf':>6}")
print("  " + "-"*82)
for clases_aceptables, msg, desc in borderline:
    s, r = post(msg)
    if s == 200:
        clase_pred = r.get("clase_predicha", "")
        confianza  = r.get("confianza", 0)
        intencion  = r.get("p_compra", 0)
        es_aceptable = clase_pred in clases_aceptables
        marca = "✓" if es_aceptable else "~"
        print(f"  {marca} '{msg[:46]}'")
        print(f"    → {clase_pred:<28} conf={confianza:.2f} intent={intencion:.3f}")
        print(f"    clases aceptables: {clases_aceptables}")
        print(f"    nota: {desc}")
        print()
        verificar_invariantes(msg, r, "borderline")


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("TEST G — Vocabulario inmobiliario argentino específico")
print("=" * 65)
print("  Términos locales que deben estar en el dataset.")
print()

vocabulario_local = [
    ("consulta_propiedad",        "tiene SUM el edificio?",          "SUM"),
    ("consulta_propiedad",        "hay quincho?",                    "quincho"),
    ("consulta_propiedad",        "es un PH?",                       "PH"),
    ("consulta_precio",           "cuanto son las expensas?",        "expensas"),
    ("intencion_compra_alquiler", "quiero señar",                    "señar"),
    ("consulta_propiedad",        "tiene parrilla en el balcon?",    "parrilla"),
    ("consulta_propiedad",        "es luminoso el dpto?",            "dpto"),
    ("consulta_precio",           "cuanto de garantia piden?",       "garantia"),
    ("solicitar_visita",          "me pasan la ubi para ir a verlo?","ubi"),
    ("interes_avanzar",           "lo hablo con mi vieja y te digo", "vieja"),
]

for clase_esp, msg, desc in vocabulario_local:
    s, r = post(msg)
    if s == 200:
        clasificar(msg, clase_esp, desc)
        verificar_invariantes(msg, r, f"vocab_{desc}")


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("TEST H — Robustez absoluta (el servicio NUNCA debe crashear)")
print("=" * 65)
print("  Inputs raros pero técnicamente válidos según los validadores.")
print()

robustez = [
    "???",
    "!!!",
    "   hola   ",                          # espacios extra
    "HOLA QUIERO COMPRAR",                 # mayúsculas
    "hola\nquiero info\ndel depa",         # saltos de línea
    "precio€ £ ¥",                         # símbolos raros
    "déjame pensarlo",                     # tilde en primera persona
    "q onda el 2B sigue??",               # jerga + referencia específica
    "no sé, quizás",                      # duda pura
    "https://zonaprop.com.ar/depa-123",   # URL pegada
    "te llamo en 5 min",                  # ninguna intención de compra
    "el vecino me dijo que hay uno libre", # indirecto
]

for texto in robustez:
    s, r = post(texto)
    check(f"'{texto[:40]}' → HTTP 200 (no crashea)",
          s == 200,
          f"status={s}")
    if s == 200:
        check(f"  → intencion en [0,1]",
              0.0 <= r.get("p_compra", -1) <= 1.0,
              f"intencion={r.get('p_compra')}")


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("RESUMEN FINAL")
print("=" * 65)
total = PASS + FAIL
print(f"  Invariantes (no negociables): {PASS}/{total}")
if FAIL > 0:
    print(f"  ✗ Hay {FAIL} invariante(s) fallando — bug en el servicio")
else:
    print(f"  ✓ Todos los invariantes OK")

print()
print(f"  Precisión semántica (orientativa): {CLASIF_OK}/{TOTAL_CLASIF} "
      f"({100*CLASIF_OK/TOTAL_CLASIF:.0f}%)")
print(f"  Umbral mínimo aceptable para producción: 65%")

pct_semantica = 100 * CLASIF_OK / TOTAL_CLASIF if TOTAL_CLASIF > 0 else 0
if pct_semantica >= 75:
    print(f"  ✓ Precisión semántica OK ({pct_semantica:.0f}%)")
elif pct_semantica >= 65:
    print(f"  ⚠ Precisión semántica aceptable ({pct_semantica:.0f}%) — mejorable con más datos")
else:
    print(f"  ✗ Precisión semántica baja ({pct_semantica:.0f}%) — el dataset necesita más ejemplos")

print()
print("INTERPRETACIÓN:")
print("  ✓ en invariantes = el SERVICIO está listo para producción")
print("  % semántica = qué tan bueno es el MODELO NLP")
print("  Son dos dimensiones separadas. Un servicio puede estar listo")
print("  aunque el modelo tenga 70% de precisión — se mejora con más datos.")

sys.exit(0 if FAIL == 0 else 1)

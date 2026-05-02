from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import json

app = FastAPI(
    title="InmoScore API — Clasificador de Intención",
    description="Clasifica mensajes de leads inmobiliarios en 7 categorías de intención de compra.",
    version="2.0.0"
)

# ─── Constantes ───────────────────────────────────────────────────────────────
MAX_CHARS = 500    # límite de un mensaje WhatsApp real (~4 líneas)
MIN_CHARS = 2      # mínimo para que SBERT produzca embedding útil

CLASES_ESPERADAS = {
    "consulta_precio", "consulta_propiedad", "desistimiento",
    "intencion_compra_alquiler", "interes_avanzar",
    "saludo_consulta", "solicitar_visita"
}

# ─── Carga de modelos (una sola vez al arranque) ──────────────────────────────
print("Cargando Sentence-BERT...")
sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("✅ SBERT cargado")

with open("modelo_clasificador.pkl", "rb") as f:
    modelo_final = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("pesos_intencion.json") as f:
    PESOS_INTENCION = json.load(f)

# ─── PROBLEMA 5 FIX: Verificar consistencia de clases al arranque ────────────
clases_modelo = set(encoder.classes_)
clases_faltantes_en_pesos = clases_modelo - set(PESOS_INTENCION.keys())
clases_extra_en_pesos     = set(PESOS_INTENCION.keys()) - clases_modelo

if clases_faltantes_en_pesos:
    raise RuntimeError(
        f"CONFIGURACIÓN INVÁLIDA: estas clases del encoder no tienen peso definido: "
        f"{clases_faltantes_en_pesos}. Agregalas a pesos_intencion.json."
    )
if clases_extra_en_pesos:
    print(f"⚠ Advertencia: pesos_intencion.json tiene clases que no están en el encoder: "
          f"{clases_extra_en_pesos}. Se ignorarán.")

print(f"✅ Modelos cargados | Clases: {sorted(encoder.classes_)}")


# ─── Schemas ──────────────────────────────────────────────────────────────────
class MensajeInput(BaseModel):
    message: str

    # PROBLEMA 1 FIX: Validar longitud del texto
    @field_validator("message")
    @classmethod
    def validar_texto(cls, v):
        v = v.strip()
        if len(v) < MIN_CHARS:
            raise ValueError(
                f"El mensaje es demasiado corto (mínimo {MIN_CHARS} caracteres). "
                "Enviá el texto del mensaje del lead."
            )
        if len(v) > MAX_CHARS:
            raise ValueError(
                f"El mensaje supera el límite de {MAX_CHARS} caracteres "
                f"(recibido: {len(v)}). SBERT trunca silenciosamente textos largos, "
                "lo que produce embeddings incorrectos."
            )
        return v

    model_config = {"json_schema_extra": {"examples": [
        {"message": "hola buenas, me pasas info del depa de 3 ambientes?"},
        {"message": "quiero señar ese departamento, cómo hago?"},
        {"message": "ya no me interesa, conseguí algo por otro lado"},
    ]}}


class ClasificacionOutput(BaseModel):
    clase_predicha:  str
    confianza:       float
    p_compra: float   # [0, 1] — clipeado, nunca negativo
    probabilities:  dict
    advertencia:     str | None = None  # presente si la confianza es baja


# ─── Endpoint principal ───────────────────────────────────────────────────────
@app.post("/clasificarv2", response_model=ClasificacionOutput)
async def clasificar_mensaje(input_data: MensajeInput):
    texto = input_data.message  # ya viene validado y con strip()

    # 1. Embedding normalizado
    emb = sbert.encode([texto], normalize_embeddings=True)

    # FIX NaN: textos con solo símbolos (???, !!!, €£¥) pueden producir
    # embeddings con NaN. El NaN se propaga a predict_proba → a la suma
    # ponderada → Pydantic lo serializa como null → el cliente recibe None.
    # np.nan_to_num reemplaza NaN por 0.0 antes de que entre al modelo.
    emb = np.nan_to_num(emb, nan=0.0)

    # 2. Probabilidades por clase
    probs = modelo_final.predict_proba(emb)[0]

    # 3. Mapear a nombres de clase
    probs_dict = {
        encoder.classes_[i]: round(float(probs[i]), 4)
        for i in range(len(encoder.classes_))
    }

    # 4. Clase ganadora y confianza
    idx_max      = int(probs.argmax())
    clase_predicha = encoder.classes_[idx_max]
    confianza    = float(probs[idx_max])

    # 5. Suma ponderada de intención
    # PROBLEMA 3 FIX: clipear a [0, 1]
    # La intención puede ser negativa si el modelo predice desistimiento con
    # alta confianza (peso -0.20). En el pipeline TypeScript se espera [0, 1].
    intencion_raw = sum(
        probs_dict.get(clase, 0.0) * PESOS_INTENCION.get(clase, 0.0)
        for clase in encoder.classes_
    )
    # Segunda línea de defensa: si probs_dict tiene NaN residual, lo neutralizamos
    intencion_raw = 0.0 if (intencion_raw != intencion_raw) else intencion_raw  # NaN check
    intencion = float(np.clip(intencion_raw, 0.0, 1.0))

    # 6. Advertencia si la confianza es baja (el modelo está inseguro)
    advertencia = None
    if confianza < 0.40:
        advertencia = (
            f"Confianza baja ({confianza:.2f}). El mensaje es ambiguo o "
            "contiene vocabulario poco representado en el entrenamiento. "
            "Considerar revisión manual."
        )

    return {
        "clase_predicha":  clase_predicha,
        "confianza":       round(confianza, 4),
        "p_compra": round(intencion, 4),
        "probabilities":  probs_dict,
        "advertencia":     advertencia,
    }


# ─── PROBLEMA 2 FIX: Health check ────────────────────────────────────────────
@app.get("/health")
def health():
    """
    Verifica que el servicio está operativo y los modelos están cargados.
    Usado por Docker/k8s para readiness probes.
    """
    return {
        "status":      "ok",
        "clases":      sorted(encoder.classes_.tolist()),
        "n_clases":    len(encoder.classes_),
        "max_chars":   MAX_CHARS,
        "sbert_model": "paraphrase-multilingual-MiniLM-L12-v2",
    }


@app.get("/clases")
def listar_clases():
    """Devuelve las 7 clases con su peso de intención."""
    return {
        "clases": [
            {
                "nombre": c,
                "peso_intencion": PESOS_INTENCION.get(c, 0),
                "descripcion": _descripcion_clase(c)
            }
            for c in sorted(encoder.classes_)
        ]
    }


def _descripcion_clase(clase: str) -> str:
    desc = {
        "saludo_consulta":           "Primer contacto vago, sin intención clara",
        "consulta_propiedad":        "Preguntas sobre características del inmueble",
        "consulta_precio":           "Preguntas sobre precio, expensas, condiciones económicas",
        "solicitar_visita":          "Quiere ver la propiedad",
        "interes_avanzar":           "Quiere avanzar pero con dudas o condicionantes",
        "intencion_compra_alquiler": "Quiere concretar: señar, reservar, firmar",
        "desistimiento":             "Se retira o cancela",
    }
    return desc.get(clase, "")

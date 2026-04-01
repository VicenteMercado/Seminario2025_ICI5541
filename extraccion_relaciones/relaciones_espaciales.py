# relaciones_espaciales.py
from dotenv import load_dotenv
import json, time, unicodedata, getpass, PyPDF2, os
from pathlib import Path
import re
from openai import OpenAI

load_dotenv() # Carga variables de entorno desde .env 
_ROOT = Path(__file__).resolve().parent.parent
_JSON_DIR = _ROOT / "json"
# ============================================================
# A) Extracción desde PDF + LLM + normalización
# ============================================================
pdf_path = _ROOT / "textos" / "El Imperio Final Ed revisada - Brandon Sanderson.pdf"

# 1) Leer PDF completo
text = ""
with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
print("Texto cargado, longitud:", len(text))


# 2) Separar capítulos completos
def split_text_by_chapters(text):
    """
    Divide el texto en chunks por capítulos (empieza en Capítulo 1).
    Si hay Prólogo u otro material antes del Cap. 1, se DESCARTA.
    """
    text = text.replace("\r\n", "\n").strip()

    # Encuentra el primer capítulo numérico y corta todo lo de antes
    m_first = re.search(r"\n\s*1\s*\n", text)
    if m_first:
        text = text[m_first.start():]

    # Divide por números de capítulo en línea
    chapter_splits = re.split(r"\n\s*(\d+)\s*\n", text)

    chunks = []
    if chapter_splits:
        # chapter_splits = ["", "1", cap1_text, "2", cap2_text, ...]
        for i in range(1, len(chapter_splits), 2):
            chapter_text = chapter_splits[i + 1].strip() if i + 1 < len(chapter_splits) else ""
            if chapter_text:
                chunks.append(chapter_text)
    else:
        chunks = [text]
    return chunks


chunks = split_text_by_chapters(text)
print("Número de chunks (capítulos completos):", len(chunks))

# 3) Filtrar fragmentos relevantes (heurística rápida)
keywords = [
    "norte", "sur", "este", "oeste", "cerca", "lejos", "entre", "millas",
    "kilometros", "kilómetros", "derecha", "izquierda", "camino", "puente",
    "valle", "bosque", "río", "rio", "ciudad", "pueblo", "castillo", "reino",
    "calle", "plaza", "canal", "fortaleza", "torreón", "torreon"
]


def is_relevant(chunk):
    c = unicodedata.normalize("NFKD", chunk.lower()).encode("ascii", "ignore").decode("ascii")
    return any(k in c for k in keywords)


relevant_chunks = [c for c in chunks if is_relevant(c)]
print("Fragmentos relevantes:", len(relevant_chunks))

# Limitar la cantidad de fragmentos a procesar
MAX_CHUNKS = 200
relevant_chunks = relevant_chunks[:MAX_CHUNKS]
print(f"Procesando solo los primeros {len(relevant_chunks)} fragmentos relevantes.")

# 4) Configuración de cliente OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "No se encontró la variable de entorno OPENAI_API_KEY. "
        "Defínela antes de ejecutar: $env:OPENAI_API_KEY = 'tu_api_key_aquí'"
    )

client = OpenAI(api_key=api_key)

# 5) Prompt estructurado
system_prompt = """
Eres un extractor de relaciones espaciales en texto narrativo.

TAREA:
Extraer lugares y relaciones espaciales dentro de la ciudad de Luthadel.

========================
ÁMBITO
========================

- Prioriza lugares dentro de Luthadel.
- Si un lugar probablemente pertenece a la ciudad por el contexto, puedes incluirlo.

========================
LUGARES
========================

- Incluye lugares con nombre propio (ej: "Kredik Shaw", "Plaza Ahlstrom").
- Incluye estructuras urbanas relevantes (calles, plazas, torreones, canales).
- NO incluyas interiores (salas, habitaciones, etc.).

========================
RELACIONES
========================

Extrae relaciones espaciales explícitas:
- NORTE_DE, SUR_DE, ESTE_DE, OESTE_DE, CERCA_DE

Además:

- Si varios lugares aparecen en la misma zona o contexto cercano, puedes agregar relaciones CERCA_DE adicionales.
- Prioriza capturar múltiples relaciones entre lugares para reflejar su posición relativa.

========================
IMPORTANTE
========================

- Usa solo el texto como base
- No inventes lugares
- Puedes incluir relaciones aproximadas si el texto sugiere proximidad

========================
FORMATO JSON
========================

{
  "lugares": ["..."],
  "relaciones": [
    {"origen":"X","tipo":"NORTE_DE","destino":"Y"}
  ]
}
"""

# 6) Llamadas al modelo
results = []
pivot_raw = []  # lugares_clave informados por el modelo

for i, chunk in enumerate(relevant_chunks):
    print(f"Procesando {i+1}/{len(relevant_chunks)}...")
    try:
        resp = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ],
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("Respuesta vacía del modelo")
        data = json.loads(content)
        results.append(data)
        pivot_raw.extend(data.get("lugares_clave", []))
    except Exception as e:
        print("Error:", e)
    time.sleep(0.6)  # pequeña pausa para evitar rate limits

# ============================================================
# B) Normalización y filtros
# ============================================================
ARTS = {"el", "la", "los", "las", "del", "de", "al", "lo"}


def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def norm_place(s: str) -> str:
    t = strip_accents(s).strip().lower()
    parts = [p for p in t.split() if p]
    while parts and parts[0] in ARTS:
        parts = parts[1:]
    return " ".join(parts)


# Tipos genéricos (para detectar patrones <tipo> de <Nombre>)
GENERIC_TYPES = {
    "ciudad", "pueblo", "aldea", "villa", "reino", "imperio", "region", "provincia", "distrito",
    "condado", "capital", "fortaleza", "castillo", "torre", "templo", "posada", "campamento",
    "bosque", "rio", "río", "lago", "montana", "montanas", "montañas", "cordillera", "desierto", "mar", "oceano",
    "océano", "valle", "puente", "camino", "puerto", "muralla", "mina", "campo", "colina",
    "tierra", "tierras", "zona", "zonas", "territorio", "nacion", "nación", "paraje", "comarca",
    "salon", "salón", "almacenes", "almacen", "edificio", "plantacion", "plantación", "montes", "cavernas", "pozos"
}

# Artículos compuestos (para detectar inicios reales)
ARTS_MULTI = {"el", "la", "los", "las", "del", "de la", "de los", "de las", "al"}


def strip_leading_articles_phrase(t: str) -> str:
    """Quita artículos simples/compuestos al inicio de la frase normalizada."""
    t = t.strip()
    for a in sorted(ARTS_MULTI, key=len, reverse=True):
        if t.startswith(a + " "):
            return t[len(a) + 1:].strip()
    return t


def tokens_core(label_norm: str):
    """Núcleo sin artículos ni genéricos ni 'de'."""
    toks = [t for t in label_norm.split() if t and t not in ARTS and t not in GENERIC_TYPES and t != "de"]
    return toks


def core_key(label_human: str) -> str:
    """Clave canónica del lugar (para fusionar variantes)."""
    cn = norm_place(label_human)
    core = tokens_core(cn)
    return " ".join(core)


def is_named_place(s: str) -> bool:
    """
    Acepta solo:
      (i) Nombre propio (al menos una palabra con mayúscula inicial no genérica), o
      (ii) <genérico> de <Nombre Propio>
    pero excluye cosas demasiado interiores o genéricas.
    """
    if not s:
        return False
    s2 = re.sub(r"\s+", " ", s.strip())
    tokens = s2.split()
    # quita artículos iniciales
    while tokens and tokens[0].lower() in ARTS:
        tokens.pop(0)
    if not tokens:
        return False

    # caso 1: empieza con genérico → necesita 'de + Nombre Propio'
    if tokens[0].lower() in GENERIC_TYPES:
        return bool(re.search(r"\bde(l| la| las| los)?\s+[A-ZÁÉÍÓÚÑ]", s2))

    # caso 2: nombre propio solo: al menos una palabra con mayúscula inicial no genérica
    for t in tokens:
        if t[0].isupper() and t.lower() not in GENERIC_TYPES and t.lower() not in ARTS:
            return True
    return False


# Palabras que indican interiores (para excluirlos siempre)
def es_interior(lugar: str) -> bool:
    exclude_keywords = {
        "ventana", "pasillo", "comedor", "escalera", "sala", "guardarropa",
        "ropero", "fondo", "habitacion", "cuarto", "piso", "puerta", "cocina",
        "bano", "baño", "chimenea", "mesa", "silla", "rincon", "armario", "armarios",
        "salon", "salón", "almacen", "almacenes", "edificio"
    }
    t = norm_place(lugar)
    return any(kw in t for kw in exclude_keywords)


# Alias específicos adaptados a este libro/mapa
ALIASES_NORM = {
    # Kredik Shaw / palacio del Lord Legislador
    "palacio del lord legislador": "Kredik Shaw",
    "palacio del legislador": "Kredik Shaw",

    # Guarniciones
    "guarnicion": "Guarnición de Luthadel",
    "guarnicion de la ciudad": "Guarnición de Luthadel",

    # Casas vs torreones/fortalezas (se unifican en el torreón)
    "casa venture": "Torreón de Venture",
    "fortaleza venture": "Torreón de Venture",
    "casa hasting": "Torreón de Hasting",
    "fortaleza hasting": "Torreón de Hasting",
    "casa lekal": "Torreón de Lekal",
    "fortaleza lekal": "Torreón de Lekal",
    "casa erikell": "Torreón de Erikeller",
    "fortaleza erikeller": "Torreón de Erikeller",

    # Cantones (incluyen variantes tipo "sede de...")
    "sede del canton de la ortodoxia": "Cantón de la Ortodoxia",
    "sede del canton de las finanzas": "Cantón de las Finanzas",

    # Variantes de pozos de Hathsin unificadas en un nodo
    "pozo de la ascension": "Pozos de Hathsin",
    "los pozos de hathsin": "Pozos de Hathsin",

    "barrios skaa de luthadel": "suburbios skaa de Luthadel",
    "cantón de las finanzas": "Cantón de las finanzas"

}


def apply_alias(label: str) -> str:
    np = norm_place(label)
    return ALIASES_NORM.get(np, label)


# Tipos de relación permitidos
VALID_TIPOS = {"NORTE_DE", "SUR_DE", "ESTE_DE", "OESTE_DE", "CERCA_DE"}

# 8) Consolidar aplicando filtros principales
all_places, all_relations = set(), []

for chunk_idx, r in enumerate(results):
    # lugares
    for p in r.get("lugares", []):
        p = apply_alias(p)
        if is_named_place(p) and not es_interior(p):
            all_places.add(p)

    # relaciones
    for rel in r.get("relaciones", []):
        o = (rel.get("origen") or "").strip()
        d = (rel.get("destino") or "").strip()
        t = (rel.get("tipo") or rel.get("relacion") or "").strip().upper()

        o = apply_alias(o)
        d = apply_alias(d)

        if t in VALID_TIPOS and is_named_place(o) and is_named_place(d) \
                and o != d and not es_interior(o) and not es_interior(d):
            all_relations.append({"origen": o, "tipo": t, "destino": d, "chunk_idx": chunk_idx})

def detectar_contradicciones(relations):
    """
    Para cada par de nodos, detecta relaciones direccionales contradictorias.
    Ejemplo: 'A NORTE_DE B' y 'B NORTE_DE A' son imposibles al mismo tiempo.
    Retorna (relaciones_limpias, relaciones_contradictorias, reporte_agrupado).
    El reporte agrupa cada conflicto por par e incluye el chunk de origen de cada relación.
    """
    from collections import defaultdict

    NS_TIPOS = {"NORTE_DE", "SUR_DE"}
    EO_TIPOS = {"ESTE_DE", "OESTE_DE"}

    ns_claims = defaultdict(set)
    eo_claims = defaultdict(set)
    ns_rels   = defaultdict(list)
    eo_rels   = defaultdict(list)

    for r in relations:
        o, d, t = r["origen"], r["destino"], r["tipo"].upper()
        pair = frozenset({o, d})
        if t == "NORTE_DE":
            ns_claims[pair].add(o)
            ns_rels[pair].append(r)
        elif t == "SUR_DE":
            ns_claims[pair].add(d)
            ns_rels[pair].append(r)
        elif t == "ESTE_DE":
            eo_claims[pair].add(o)
            eo_rels[pair].append(r)
        elif t == "OESTE_DE":
            eo_claims[pair].add(d)
            eo_rels[pair].append(r)

    pares_ns_conflict = {p for p, c in ns_claims.items() if len(c) > 1}
    pares_eo_conflict = {p for p, c in eo_claims.items() if len(c) > 1}

    # Construir reporte agrupado e imprimir en consola
    reporte = []
    for pair in pares_ns_conflict:
        rels = sorted(ns_rels[pair], key=lambda x: x.get("chunk_idx") or 0)
        nombres = sorted(pair)
        reporte.append({"par": nombres, "eje": "NS", "relaciones": rels})
        print(f"\n⚠️  Contradicción N/S: \"{nombres[0]}\" vs \"{nombres[1]}\"")
        for r in rels:
            cap = (r.get("chunk_idx") or 0) + 1
            print(f"   Cap.{cap:>3} → {r['origen']} {r['tipo']} {r['destino']}")

    for pair in pares_eo_conflict:
        rels = sorted(eo_rels[pair], key=lambda x: x.get("chunk_idx") or 0)
        nombres = sorted(pair)
        reporte.append({"par": nombres, "eje": "EO", "relaciones": rels})
        print(f"\n⚠️  Contradicción E/O: \"{nombres[0]}\" vs \"{nombres[1]}\"")
        for r in rels:
            cap = (r.get("chunk_idx") or 0) + 1
            print(f"   Cap.{cap:>3} → {r['origen']} {r['tipo']} {r['destino']}")

    clean, contradictorias = [], []
    for r in relations:
        o, d, t = r["origen"], r["destino"], r["tipo"].upper()
        pair = frozenset({o, d})
        if (t in NS_TIPOS and pair in pares_ns_conflict) or \
           (t in EO_TIPOS and pair in pares_eo_conflict):
            contradictorias.append(r)
        else:
            clean.append(r)

    return clean, contradictorias, reporte

# 9) Deduplicar y canonizar lugares
canon2label = {}
cleaned_places = []
for p in all_places:
    c = norm_place(p)
    if c and c not in canon2label:
        canon2label[c] = p  # se conserva la primera etiqueta humana encontrada
        cleaned_places.append(p)


def remap(name: str) -> str:
    return canon2label.get(norm_place(name), name)


# Normalizar relaciones y deduplicar
tmp_rel = []
for r in all_relations:
    o, d = remap(r["origen"]), remap(r["destino"])
    if o in cleaned_places and d in cleaned_places and o != d:
        tmp_rel.append({"origen": o, "tipo": r["tipo"], "destino": d, "chunk_idx": r.get("chunk_idx")})

rel_set = set()
clean_relations = []
for r in tmp_rel:
    key = (r["origen"], r["tipo"], r["destino"])
    if key not in rel_set:
        rel_set.add(key)
        clean_relations.append(r)

# Eliminar relaciones contradictorias
clean_relations, contradicciones, reporte_contradicciones = detectar_contradicciones(clean_relations)

print(f"\nTotal: {len(contradicciones)} relaciones contradictorias eliminadas ({len(reporte_contradicciones)} pares en conflicto).")

with open("contradicciones.json", "w", encoding="utf-8") as f:
    json.dump(reporte_contradicciones, f, indent=2, ensure_ascii=False)
print("Reporte de contradicciones guardado en contradicciones.json")

# ============================================================
# C) Meta-info, pivotes y filtro Luthadel
# ============================================================

# Conteo de menciones en el texto completo
text_norm_full = strip_accents(text).lower()


def count_mentions(label: str) -> int:
    base = re.escape(norm_place(label))
    if not base:
        return 0
    return len(re.findall(rf"\b{base}\b", text_norm_full))


mentions = {p: count_mentions(p) for p in cleaned_places}

# Grado y conexiones por CERCA_DE
deg = {p: 0 for p in cleaned_places}
near_conn = {p: 0 for p in cleaned_places}

for rel in clean_relations:
    o, d, t = rel["origen"], rel["destino"], rel["tipo"].upper()
    deg[o] += 1
    deg[d] += 1
    if t == "CERCA_DE":
        near_conn[o] += 1
        near_conn[d] += 1

# Meta por lugar
lugares_meta = {}
for p in cleaned_places:
    score = (2 * mentions[p]) + deg[p] + near_conn[p]
    lugares_meta[p] = {
        "mentions": int(mentions[p]),
        "deg": int(deg[p]),
        "near_conn": int(near_conn[p]),
        "pivot_score": int(score),
    }

# Pivote: solo Kredik Shaw si aparece
SPECIAL_PIVOTS_NORM = {"kredik shaw"}
pivotes = []
for p in cleaned_places:
    if norm_place(p) in SPECIAL_PIVOTS_NORM and p not in pivotes:
        pivotes.append(p)

print("\n=== Pivotes seleccionados (extractor) ===")
if not pivotes:
    print("(ninguno; no se detectó 'Kredik Shaw' en los lugares finales)")
else:
    for i, p in enumerate(pivotes, 1):
        m = lugares_meta[p]
        print(f"{i}. {p}  (score={m['pivot_score']}, menciones={m['mentions']}, grado={m['deg']})")

# ============================================================
# D) Heurística de pertenencia a Luthadel + filtro final
# ============================================================

text_plain = strip_accents(text).lower()


def appears_with_luthadel(label: str, window: int = 250) -> bool:
    base = strip_accents(label).lower().strip()
    if not base or len(base) < 4:
        return False
    pattern = re.escape(base)
    for m in re.finditer(pattern, text_plain):
        start = max(0, m.start() - window)
        end = min(len(text_plain), m.end() + window)
        if "luthadel" in text_plain[start:end]:
            return True
    return False


# Algunos lugares se fuerzan como "dentro de Luthadel" aunque no siempre aparezcan cerca de la palabra "Luthadel"
INSIDE_FORCE_RAW = {
    "kredik shaw",
    "plaza de la fuente",
    "guarnicion de luthadel",
    "cantón de las finanzas",
    "cantón de la ortodoxia",
    "cantón de la inquisicion",
    "cantón de la inquisición",
    "calle kenton",
    "taller de clubs",
    "plaza ahlstrom",
    "colina de las mil torres",
    "torreon de venture",
    "torreon de hasting",
    "torreon de lekal",
    "torreon de erikeller",
    "fortaleza venture",
    "fortaleza hasting",
    "fortaleza lekal",
    "fortaleza erikeller",
    "suburbios skaa de luthadel",
    "mercado ska",
}

INSIDE_FORCE_NORM = {norm_place(s) for s in INSIDE_FORCE_RAW}

inside_luthadel = {}
for p in cleaned_places:
    np = norm_place(p)
    if np in INSIDE_FORCE_NORM:
        inside_luthadel[p] = True
    elif "luthadel" in np:
        # cualquier "X de Luthadel"
        inside_luthadel[p] = True
    else:
        inside_luthadel[p] = appears_with_luthadel(p)

# Lugares fuera de Luthadel que se descartan siempre
OUTSIDE_LUTHADEL_RAW = {
    "fellise",
    "holstep",
    "valtroux",
    "dominio central",
    "dominio extremo",
    "montes de ceniza",
    "plantación de lord tresting",
    "cavernas arguois",
    "pozos de hathsin",
    "los pozos de hathsin",
    "guarnición de holstep",
    "guarnición de valtroux",
}

OUTSIDE_LUTHADEL_NORM = {norm_place(s) for s in OUTSIDE_LUTHADEL_RAW}


# Lugares conceptuales que no se quieren como nodo
DROP_NORM = {
    "grandes casas",
    "grandes casas de luthadel"
}

TOP_LEVEL_CITY_NORM = "luthadel"
SPECIAL_ALWAYS_KEEP = {"kredik shaw", "plaza de la fuente"}

INCLUDE_ISOLATES_POLICY = "all"  # se mantienen todos los lugares de Luthadel, incluso aislados

# Lugares principales del mapa de Luthadel que conviene conservar siempre
OFFICIAL_PLACES = [
    "Plaza de la Fuente",
    "Kredik Shaw",
    "Cantón de la Ortodoxia",
    "Cantón de las Finanzas",
    "Guarnición de Luthadel",
    "Torreón de Venture",
    "Torreón de Hasting",
    "Torreón de Lekal",
    "Torreón de Erikeller",
    "taller de Clubs",
    "guarida de Camon",
    "Calle de la Antigua Muralla",
    "calle Kenton",
    "Plaza Ahlstrom",
    "Encrucijada Quince",
    "Calle del Canal",
    "Mercado Ska",
]

ALLOWLIST_NORM = {norm_place(n) for n in OFFICIAL_PLACES}

# Lugares que no se quieren nunca en el mapa de la ciudad (claramente fuera)
BLOCKLIST_PLACES = [
    "Holstep", "ciudad de Holstep", "Guarnición de Holstep",
    "Valtroux", "ciudad de Valtroux", "Guarnición de Valtroux",
    "Dominio Central", "Dominio Extremo",
    "Mansión Renoux", "mansión de Renoux", "almacenes de Renoux",
    "plantación de lord Tresting", "plantación de Tresting",
    "Montes de Ceniza", "Colina de las Mil Torres"
    "Fellise","Casa de vecinos", "Casa de Clubs", "guarida de Vin",
]
BLOCKLIST_NORM = {norm_place(n) for n in BLOCKLIST_PLACES}

# Coincide con la allowlist, pero se marca explícitamente
SPECIAL_ALWAYS_KEEP = {
    norm_place("Kredik Shaw"),
    norm_place("Plaza de la Fuente"),
}

# Grado por nodo (recalculado por si clean_relations ha cambiado)
deg = {p: 0 for p in cleaned_places}
for rel in clean_relations:
    o, d = rel["origen"], rel["destino"]
    if o in deg: deg[o] += 1
    if d in deg: deg[d] += 1

def keep_place(p: str) -> bool:
    np = norm_place(p)

    # 0) Blocklist fuerte → fuera siempre
    if np in BLOCKLIST_NORM:
        return False
    if np in OUTSIDE_LUTHADEL_NORM or np in DROP_NORM:
        return False

    # 1) No se incluye el nodo "Luthadel" (es la ciudad completa)
    if np == "luthadel":
        return False

    # 2) Interiores (salones, habitaciones, etc.) → fuera
    if es_interior(p):
        return False

    # 3) Allowlist y pivotes importantes → siempre dentro
    if np in ALLOWLIST_NORM:
        return True
    if np in SPECIAL_ALWAYS_KEEP:
        return True

    # 4) Si tiene al menos una relación, se mantiene
    if deg.get(p, 0) > 0:
        return True

    # 5) Nodos muy mencionados pero sin relaciones explícitas
    meta_p = lugares_meta.get(p, {})
    if meta_p.get("mentions", 0) >= 10:
        return True

    # 6) El resto se considera ruido (exteriores, genéricos, menciones puntuales)
    return False

# Aplicar filtro final
filtered_places = [p for p in cleaned_places if keep_place(p)]
kept = set(filtered_places)
filtered_relations = [
    r for r in clean_relations
    if r["origen"] in kept and r["destino"] in kept
]
excluidos = [p for p in cleaned_places if p not in kept]

print("\n=== Lugares finales incluidos en el grafo (después de filtros) ===")
for i, p in enumerate(filtered_places, 1):
    print(f"{i}. {p}")

print(f"\nFiltrado final: mantuve {len(filtered_places)} lugares, descarté {len(excluidos)}.")
if excluidos:
    print("Descartados (muestra):", ", ".join(excluidos[:10]), ("..." if len(excluidos) > 10 else ""))


print("\nResumen:")
print(f"- Lugares totales detectados (antes de filtros): {len(cleaned_places)}")
print(f"- Relaciones totales detectadas (antes de filtros): {len(clean_relations)}")
print(f"- Lugares finales guardados: {len(filtered_places)}")
_map_out = _JSON_DIR / "map_relations.json"
print(f"- Relaciones finales guardadas: {len(filtered_relations)} en {_map_out}")

# Guardar JSON final
_JSON_DIR.mkdir(parents=True, exist_ok=True)
with open(_map_out, "w", encoding="utf-8") as f:
    json.dump(
        {
            "lugares": filtered_places,
            "relaciones": filtered_relations,
            "lugares_meta": lugares_meta,
            "pivotes": pivotes,
            "excluidos": excluidos
        },
        f,
        indent=2,
        ensure_ascii=False
    )

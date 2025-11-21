# relaciones_espaciales.py
import json, time, unicodedata, getpass, PyPDF2
import re
from openai import OpenAI

# ============================================================
# A) EXTRACCIÓN DESDE PDF + LLM + NORMALIZACIÓN
# ============================================================
pdf_path = r"textos\El Imperio Final Ed revisada - Brandon Sanderson.pdf"

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

# === Limitar la cantidad de fragmentos a procesar ===
MAX_CHUNKS = 200
relevant_chunks = relevant_chunks[:MAX_CHUNKS]
print(f"Procesando solo los primeros {len(relevant_chunks)} fragmentos relevantes.")

# 4) OpenAI
api_key = getpass.getpass("Introduce tu API key: ")
client = OpenAI(api_key=api_key)

# 5) Prompt estructurado
system_prompt = """
Eres un extractor de relaciones espaciales entre lugares en un texto narrativo.
Ahora debes ser MENOS estricto que antes: es preferible capturar más lugares y más relaciones,
aunque algunas sean aproximadas, siempre que:

- Estén claramente dentro o en la ciudad de Luthadel, o formen parte directa de ella.
- Tengan UN NOMBRE CLARO y no sean simplemente objetos o habitaciones internas.

========================
ÁMBITO DEL MAPA
========================
Nos interesa SOLO la ciudad de Luthadel, capital del Imperio Final, y sus componentes urbanos.
Incluye:
- Fortalezas, casas nobles, torreones y lugares urbanos importantes dentro de Luthadel.
- Plazas, calles, canales, guarniciones, cantones, mercados y zonas urbanas claramente dentro de la ciudad.
- Periferias urbanas, barrios o zonas como "suburbios skaa de Luthadel", si se describen como parte de la ciudad.

EXCLUYE (NO los incluyas, aunque tengan nombre propio):
- Otras ciudades, dominios, plantaciones, montes, cavernas lejanas, etc.
- Ejemplos concretos que DEBES excluir:
  "Montes de Ceniza", "plantación de lord Tresting",
  "Dominio Central", "Dominio Extremo", "Fellise",
  "Holstep", "Valtroux", "Guarnición de Holstep",
  "Pozos de Hathsin", "Pozo de Hathsin", "Pozos de Hathsin",
  "cavernas Arguois" (y cualquier lugar claramente lejano de Luthadel).

========================
INTERIORES (MUY IMPORTANTE)
========================
NO incluyas salas internas, habitaciones ni sub-espacios interiores de edificios:
- Ejemplos:
  "salón de baile Venture", "salón principal Venture",
  "salón de caballeros del Torreón de Lekal",
  "almacenes de Renoux", "edificio de los Quiebros".
Estos NO deben aparecer como lugares en la salida. Solo importa el edificio o complejo principal:
por ejemplo, "Casa Venture", "Torreón de Lekal", "Casa Elariel", etc.

========================
INSTRUCCIONES PRINCIPALES
========================

1) Lugares válidos
Incluye lugares solo si:
  A. Tienen nombre propio o topónimo claro (ej.: "Plaza Ahlstrom", "calle Kenton").
  B. Son combinaciones de genérico + nombre propio ("Torreón de Hasting", "canal de Luth-Davn").
  C. Son entidades urbanas significativas dentro de Luthadel: plazas, calles, fuertes, canales, guarniciones,
     cantones, fortalezas, torreones, casas nobles, plazas importantes, mercados.

Excluye:
  - Términos completamente genéricos sin nombre específico ("la plaza", "la calle", "el canal").
  - Interiores y salas internas (salones, almacenes, edificios internos, despachos, habitaciones).
  - Objetos pequeños (mesas, pasillos, habitaciones).
  - Regiones lejanas, dominios o ciudades externas.

2) Relaciones espaciales
Extrae TODAS las relaciones espaciales explícitas que encuentres entre lugares válidos:
  - NORTE_DE, SUR_DE, ESTE_DE, OESTE_DE
  - CERCA_DE

"Explícitas" significa que el texto indica claramente una relación espacial,
aunque sea aproximada (por ejemplo, "cerca de", "junto a", "al norte de", etc.).

Está permitido usar frases como:
  - "X se encontraba cerca de Y" → {"tipo":"CERCA_DE"}
  - "X quedaba al norte de Y" → {"tipo":"NORTE_DE"}

Si la relación es muy ambigua o puramente narrativa sin referencia espacial, no la uses.

3) Luthadel y macro-lugares
No incluyas "Luthadel" como nodo/lugar final: úsalo solo mentalmente como contexto.
No incluyas "Grandes Casas" ni "Grandes Casas de Luthadel" como lugar independiente:
es una categoría social, no una ubicación puntual del mapa.

4) Identificación de pivotes
Si aparece "Kredik Shaw", añádelo siempre como lugar y considéralo como pivote central.

========================
LISTA NEGRA DE GENÉRICOS (si aparecen sin nombre propio → EXCLUIR)
========================
plaza, calle, avenida, puente, canal, muralla, puerta, barrio, distrito, mercado,
palacio, templo, fortaleza, torre, castillo, taberna, posada, campamento, edificio, casa,
salon, salón, almacenes, plantación, montes, montañas, cavernas, pozos

(Esta lista se suma a cualquier filtro interno de genéricos que uses.)

========================
SALIDA JSON
========================
Devuelve SIEMPRE un JSON estricto:

{
  "lugares_clave": ["..."],     // ej. ["Kredik Shaw"]
  "lugares": ["..."],          // lista de lugares válidos dentro de Luthadel
  "relaciones": [
    {"origen":"X","tipo":"NORTE_DE","destino":"Y"},
    {"origen":"A","tipo":"SUR_DE","destino":"B"},
    {"origen":"C","tipo":"ESTE_DE","destino":"D"},
    {"origen":"E","tipo":"OESTE_DE","destino":"F"},
    {"origen":"G","tipo":"CERCA_DE","destino":"H"}
  ]
}

Si en el fragmento no hay lugares válidos dentro de Luthadel:
{"lugares_clave": [], "lugares": [], "relaciones": []}

========================
REGLAS ADICIONALES
========================
- Es mejor incluir una relación dudosa pero plausible que omitir demasiadas.
- NO uses conocimiento externo al fragmento proporcionado.
- No incluyas lugares fuera de Luthadel (ni dominios, ni montes, ni otras ciudades).
- NO incluyas interiores ni salas internas.
"""

# 6) Llamadas al LLM
results = []
pivot_raw = []  # lugares_clave reportados por el LLM

for i, chunk in enumerate(relevant_chunks):
    print(f"Procesando {i+1}/{len(relevant_chunks)}...")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        results.append(data)
        pivot_raw.extend(data.get("lugares_clave", []))
    except Exception as e:
        print("Error:", e)
    time.sleep(0.6)  # anti rate-limit

# ============================================================
# B) NORMALIZACIÓN Y FILTROS
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

# Artículos compuestos (para chequear inicios reales)
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


# interiores a excluir pase lo que pase
def es_interior(lugar: str) -> bool:
    exclude_keywords = {
        "ventana", "pasillo", "comedor", "escalera", "sala", "guardarropa",
        "ropero", "fondo", "habitacion", "cuarto", "piso", "puerta", "cocina",
        "bano", "baño", "chimenea", "mesa", "silla", "rincon", "armario", "armarios",
        "salon", "salón", "almacen", "almacenes", "edificio"
    }
    t = norm_place(lugar)
    return any(kw in t for kw in exclude_keywords)


# ===== Alias específicos (para este libro/mapa) =====
ALIASES_NORM = {
    # Kredik Shaw / palacio del Lord Legislador
    "palacio del lord legislador": "Kredik Shaw",
    "palacio del legislador": "Kredik Shaw",

    # Guarniciones
    "guarnicion": "Guarnición de Luthadel",
    "guarnicion de la ciudad": "Guarnición de Luthadel",

    # Casas vs torreones/fortalezas (unificamos en el torreón)
    "casa venture": "Torreón de Venture",
    "fortaleza venture": "Torreón de Venture",
    "casa hasting": "Torreón de Hasting",
    "fortaleza hasting": "Torreón de Hasting",
    "casa lekal": "Torreón de Lekal",
    "fortaleza lekal": "Torreón de Lekal",
    "casa erikell": "Torreón de Erikeller",
    "fortaleza erikeller": "Torreón de Erikeller",

    # Cantones (por si vienen con "sede de...")
    "sede del canton de la ortodoxia": "Cantón de la Ortodoxia",
    "sede del canton de las finanzas": "Cantón de las Finanzas",

    # Variantes de pozos de Hathsin (si quieres tratarlos como un solo nodo)
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

# 8) Consolidar aplicando filtros duros
all_places, all_relations = set(), []

for r in results:
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
            all_relations.append({"origen": o, "tipo": t, "destino": d})

# 9) Deduplicar y canonizar lugares
canon2label = {}
cleaned_places = []
for p in all_places:
    c = norm_place(p)
    if c and c not in canon2label:
        canon2label[c] = p  # conservar primera etiqueta humana
        cleaned_places.append(p)


def remap(name: str) -> str:
    return canon2label.get(norm_place(name), name)


# normalizar relaciones y deduplicar
tmp_rel = []
for r in all_relations:
    o, d = remap(r["origen"]), remap(r["destino"])
    if o in cleaned_places and d in cleaned_places and o != d:
        tmp_rel.append({"origen": o, "tipo": r["tipo"], "destino": d})

rel_set = set()
clean_relations = []
for r in tmp_rel:
    key = (r["origen"], r["tipo"], r["destino"])
    if key not in rel_set:
        rel_set.add(key)
        clean_relations.append(r)

# ============================================================
# C) META-INFO, PIVOTES Y FILTRO LUTH ADEL
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
# D) HEURÍSTICA DE PERTENENCIA A LUTHADEL + FILTRO FINAL
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


# Forzamos como "dentro de Luthadel" algunos lugares clave,
# aunque el contexto no los mencione siempre cerca de la palabra "Luthadel".
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

# Lugares FUERA de Luthadel que queremos descartar siempre
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


# Lugares "conceptuales" que no queremos como nodo
DROP_NORM = {
    "grandes casas",
    "grandes casas de luthadel"
}

TOP_LEVEL_CITY_NORM = "luthadel"
SPECIAL_ALWAYS_KEEP = {"kredik shaw", "plaza de la fuente"}

INCLUDE_ISOLATES_POLICY = "all"  # mantenemos todos los lugares de Luthadel, incluso aislados

# === Filtro de lugares (post-procesado específico de Luthadel) ===

# Lugares "oficiales" del mapa de Luthadel que quieres conservar sí o sí.
OFFICIAL_PLACES = [
    # lista que me diste + algunos nombres que salen tal cual en el texto
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

# Lugares que NO quieres nunca en el mapa de la ciudad (claramente fuera).
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

# Aunque en la allowlist ya están, por claridad:
SPECIAL_ALWAYS_KEEP = {
    norm_place("Kredik Shaw"),
    norm_place("Plaza de la Fuente"),
}

# Grado por nodo (vuelve a calcularse aquí por si ha cambiado algo en clean_relations)
deg = {p: 0 for p in cleaned_places}
for rel in clean_relations:
    o, d = rel["origen"], rel["destino"]
    if o in deg: deg[o] += 1
    if d in deg: deg[d] += 1

def keep_place(p: str) -> bool:
    np = norm_place(p)

    # 0) Blocklist dura → fuera siempre
    if np in BLOCKLIST_NORM:
        return False
    if np in OUTSIDE_LUTHADEL_NORM or np in DROP_NORM:
        return False

    # 1) No queremos el nodo "Luthadel" en este mapa (es la ciudad entera)
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

    # 4) Si está conectado por lo menos a algo, lo mantenemos
    if deg.get(p, 0) > 0:
        return True

    # 5) Nodos muy mencionados pero sin relaciones explícitas:
    meta_p = lugares_meta.get(p, {})
    if meta_p.get("mentions", 0) >= 10:
        return True

    # 6) El resto, fuera (ruido: exteriores, genéricos, menciones puntuales)
    return False

# Aplicar filtro
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
print(f"- Relaciones finales guardadas: {len(filtered_relations)} en map_relations.json")

# Guardado JSON final
with open("map_relations.json", "w", encoding="utf-8") as f:
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

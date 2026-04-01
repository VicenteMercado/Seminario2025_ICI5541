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
pdf_path = _ROOT / "textos" / "Historia de Roma Libro 1 al 10 - Tito Livio.pdf"

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
    Divide el texto en chunks por capítulos. Detecta automáticamente el formato:
      - Marcadores [libro,cap] como [1,1], [2,3], etc. (ej.: Tito Livio)
      - Números solos en línea como separadores (ej.: El Imperio Final)
    Si hay prólogo u otro material antes del primer marcador, se descarta.
    """
    text = text.replace("\r\n", "\n").strip()

    # Formato 1: marcadores [libro,cap] — ej. [1,1], [2,3], [10,40]
    markers = list(re.finditer(r"\[\d+,\d+\]", text))
    if len(markers) >= 3:
        chunks = []
        for i, m in enumerate(markers):
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    # Formato 2: números solos en línea — ej. \n 1 \n ... \n 2 \n
    m_first = re.search(r"\n\s*1\s*\n", text)
    if m_first:
        text = text[m_first.start():]

    chapter_splits = re.split(r"\n\s*(\d+)\s*\n", text)

    chunks = []
    if chapter_splits:
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

# 5) Prompt estructurado (genérico para cualquier texto narrativo)
system_prompt = """
Eres un extractor de relaciones espaciales entre lugares en un texto narrativo.
Tu objetivo es identificar TODOS los lugares geográficos con nombre propio y las
relaciones espaciales explícitas entre ellos.

Debes capturar la mayor cantidad posible de lugares y relaciones, aunque algunas
sean aproximadas, siempre que:
- Los lugares tengan UN NOMBRE PROPIO o topónimo claro.
- Las relaciones espaciales estén indicadas explícitamente en el texto.

========================
LUGARES VÁLIDOS
========================
Incluye lugares solo si:
  A. Tienen nombre propio o topónimo claro
     (ej.: "Roma", "río Tíber", "Plaza Ahlstrom", "Monte Aventino").
  B. Son combinaciones de genérico + nombre propio
     (ej.: "Templo de Júpiter", "Torreón de Hasting", "Puerta de Capena").
  C. Son entidades geográficas significativas: ciudades, ríos, montes, regiones,
     colinas, templos, plazas, fortalezas, puentes, calles, lagos, puertos,
     bosques, valles, islas, mares, caminos, etc.

Excluye:
  - Términos completamente genéricos SIN nombre específico
    ("la plaza", "la calle", "el río", "el monte", "la ciudad", "el templo").
  - Interiores y sub-espacios de edificios: salones, habitaciones, almacenes,
    despachos, pasillos, cocinas. Solo importa el edificio o complejo principal.
  - Objetos pequeños (mesas, sillas, puertas internas).
  - Personas, dioses o conceptos abstractos usados como si fueran lugares.
  - Gentilicios o nombres de pueblos como grupo social (ej.: "los romanos",
    "los etruscos", "los skaa"), a menos que nombren un territorio concreto.

========================
RELACIONES ESPACIALES
========================
Extrae TODAS las relaciones espaciales explícitas entre lugares válidos:
  - NORTE_DE, SUR_DE, ESTE_DE, OESTE_DE
  - CERCA_DE

"Explícitas" significa que el texto indica claramente una relación espacial,
aunque sea aproximada. Frases válidas incluyen:
  - "X se encontraba cerca de Y" → {"tipo":"CERCA_DE"}
  - "X quedaba al norte de Y" → {"tipo":"NORTE_DE"}
  - "X estaba junto a Y" / "X, vecina de Y" → {"tipo":"CERCA_DE"}
  - "al sur de X se hallaba Y" → {"tipo":"SUR_DE"} (origen=Y, destino=X)
  - "entre X e Y" → {"tipo":"CERCA_DE"} para ambos pares si aplica
  - "a orillas de X estaba Y" → {"tipo":"CERCA_DE"}
  - "cruzando el río X se llegaba a Y" → {"tipo":"CERCA_DE"}

Si la relación es muy ambigua o puramente narrativa sin referencia espacial, no la uses.

========================
DISTANCIAS (OPCIONAL)
========================
Si el texto menciona una distancia numérica explícita entre dos lugares, inclúyela
en la relación con los campos "distancia" (valor numérico) y "unidad" (la unidad
tal como aparece en el texto).

Ejemplos:
  - "Roma estaba a 12 millas de Veyes"
    → {"origen":"Roma","tipo":"CERCA_DE","destino":"Veyes","distancia":12,"unidad":"millas"}
  - "a 300 estadios al norte de Capua"
    → {"origen":"...","tipo":"NORTE_DE","destino":"Capua","distancia":300,"unidad":"estadios"}
  - "a dos jornadas de camino de Antium"
    → {"origen":"...","tipo":"CERCA_DE","destino":"Antium","distancia":2,"unidad":"jornadas"}

Si NO hay distancia numérica explícita, simplemente omite los campos "distancia" y "unidad".

========================
LUGARES CLAVE (PIVOTES)
========================
En "lugares_clave", incluye los lugares que aparecen como referencia central o
punto de anclaje geográfico en el fragmento (el lugar más mencionado o que sirve
de referencia para ubicar a los demás). Puede estar vacío si no hay uno claro.

========================
LISTA NEGRA DE GENÉRICOS (si aparecen sin nombre propio → EXCLUIR)
========================
plaza, calle, avenida, puente, canal, muralla, puerta, barrio, distrito, mercado,
palacio, templo, fortaleza, torre, castillo, taberna, posada, campamento, edificio,
casa, salón, almacenes, plantación, montes, montañas, cavernas, pozos, río, monte,
colina, lago, puerto, campo, valle, bosque, región, territorio, isla, mar, camino

========================
SALIDA JSON
========================
Devuelve SIEMPRE un JSON estricto con esta estructura:

{
  "lugares_clave": ["..."],
  "lugares": ["..."],
  "relaciones": [
    {"origen":"X","tipo":"NORTE_DE","destino":"Y"},
    {"origen":"A","tipo":"CERCA_DE","destino":"B","distancia":12,"unidad":"millas"},
    {"origen":"C","tipo":"ESTE_DE","destino":"D"},
    {"origen":"G","tipo":"CERCA_DE","destino":"H"}
  ]
}

Si en el fragmento no hay lugares válidos:
{"lugares_clave": [], "lugares": [], "relaciones": []}

========================
REGLAS ADICIONALES
========================
- Es mejor incluir una relación dudosa pero plausible que omitir demasiadas.
- NO uses conocimiento externo al fragmento proporcionado.
- NO incluyas interiores ni salas internas de edificios.
- Usa siempre el nombre tal como aparece en el texto (no traduzcas ni normalices).
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
# B-0) Diccionario de conversión de unidades a metros
# ============================================================
UNIDADES_A_METROS = {
    # Métricas
    "m": 1, "metro": 1, "metros": 1,
    "km": 1000, "kilometro": 1000, "kilómetro": 1000,
    "kilometros": 1000, "kilómetros": 1000,

    # Romanas / antiguas
    "milla": 1480, "millas": 1480,                       # milla romana (mille passus)
    "milla romana": 1480, "millas romanas": 1480,
    "estadio": 185, "estadios": 185,                      # stadion griego/romano
    "paso": 1.48, "pasos": 1.48,                          # passus romano
    "pie": 0.296, "pies": 0.296,                          # pes romano

    # Imperiales / anglosajonas
    "milla inglesa": 1609, "millas inglesas": 1609,
    "mile": 1609, "miles": 1609,
    "yarda": 0.9144, "yardas": 0.9144,
    "yard": 0.9144, "yards": 0.9144,
    "pie ingles": 0.3048, "pies ingleses": 0.3048,
    "foot": 0.3048, "feet": 0.3048,

    # Medievales / hispanas
    "legua": 5572, "leguas": 5572,                        # legua castellana
    "league": 5556, "leagues": 5556,

    # Aproximaciones de tiempo como distancia
    "jornada": 30000, "jornadas": 30000,                  # ~30 km/día a pie
    "dia de camino": 30000, "dias de camino": 30000,
    "día de camino": 30000, "días de camino": 30000,
}


def convertir_a_metros(distancia, unidad):
    """Convierte una distancia con unidad textual a metros. Retorna None si no se reconoce."""
    if distancia is None or unidad is None:
        return None
    clave = strip_accents(str(unidad).strip().lower())
    factor = UNIDADES_A_METROS.get(clave)
    if factor is None:
        print(f"  ⚠️ Unidad no reconocida: '{unidad}' (distancia={distancia})")
        return None
    return round(float(distancia) * factor, 1)


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
            entry = {"origen": o, "tipo": t, "destino": d, "chunk_idx": chunk_idx}
            # Convertir distancia a metros si el LLM la extrajo
            dist_raw = rel.get("distancia")
            unit_raw = rel.get("unidad")
            if dist_raw is not None and unit_raw is not None:
                dist_m = convertir_a_metros(dist_raw, unit_raw)
                if dist_m is not None:
                    entry["distancia_m"] = dist_m
                    entry["distancia_orig"] = f"{dist_raw} {unit_raw}"
            all_relations.append(entry)

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
        entry = {"origen": o, "tipo": r["tipo"], "destino": d, "chunk_idx": r.get("chunk_idx")}
        if "distancia_m" in r:
            entry["distancia_m"] = r["distancia_m"]
            entry["distancia_orig"] = r.get("distancia_orig")
        tmp_rel.append(entry)

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

# Pivotes: los lugares_clave más reportados por el LLM
from collections import Counter
MAX_PIVOTES = 5
pivot_counts = Counter(apply_alias(p) for p in pivot_raw)
pivotes = [p for p, _ in pivot_counts.most_common() if p in cleaned_places][:MAX_PIVOTES]

print("\n=== Pivotes seleccionados (extractor) ===")
if not pivotes:
    print("(ninguno; el LLM no reportó lugares_clave en los fragmentos)")
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


# Resumen de distancias extraídas
rels_con_dist = [r for r in filtered_relations if "distancia_m" in r]

print("\nResumen:")
print(f"- Lugares totales detectados (antes de filtros): {len(cleaned_places)}")
print(f"- Relaciones totales detectadas (antes de filtros): {len(clean_relations)}")
print(f"- Lugares finales guardados: {len(filtered_places)}")
_map_out = _JSON_DIR / "map_relations.json"
print(f"- Relaciones finales guardadas: {len(filtered_relations)} en {_map_out}")
print(f"- Relaciones con distancia concreta: {len(rels_con_dist)}")
if rels_con_dist:
    print("\n=== Distancias extraídas ===")
    for r in rels_con_dist:
        print(f"  {r['origen']} ↔ {r['destino']}: {r['distancia_m']}m ({r.get('distancia_orig', '?')})")

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

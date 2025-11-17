# relaciones_espaciales.py
import json, time, unicodedata, getpass, PyPDF2
import PyPDF2
import re
from openai import OpenAI

# ============================================================
# A) EXTRACCIÓN DESDE PDF + LLM (con 'tipo') + NORMALIZACIÓN
# ============================================================
pdf_path = r"textos\El Imperio Final Ed revisada - Brandon Sanderson.pdf"

# 1) Leer PDF
text = ""
with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
print("Texto cargado, longitud:", len(text))

# 2) Separar capítulos y chunking adaptativo
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
            # chapter_number = chapter_splits[i]  # no usamos de momento
            chapter_text = chapter_splits[i + 1].strip() if i + 1 < len(chapter_splits) else ""
            if chapter_text:
                chunks.append(chapter_text)
    else:
        chunks = [text]
    return chunks

chunks = split_text_by_chapters(text)
print("Número de chunks (capítulos completos):", len(chunks))

# 3) Filtrar fragmentos relevantes (heurística rápida)
keywords = ["norte","sur","este","oeste","cerca","lejos","entre","millas",
            "kilometros","kilómetros","derecha","izquierda","camino","puente",
            "valle","bosque","río","rio","ciudad","pueblo","castillo","reino"]

def is_relevant(chunk):
    c = unicodedata.normalize("NFKD", chunk.lower()).encode("ascii","ignore").decode("ascii")
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

# 5) Prompt estructurado (usa campo 'tipo')
system_prompt = """
Eres un extractor extremadamente estricto de relaciones espaciales entre lugares mencionados en textos narrativos (fantasía, ciencia ficción o realismo). 
Tu objetivo es identificar ÚNICAMENTE lugares geográficos o espaciales claros y sus relaciones espaciales explícitas.

========================
INSTRUCCIONES PRINCIPALES
========================

1) Identificación de lugares clave/pivote
- Identifica los lugares más importantes o recurrentes que sirvan como pivote.
- Excluye nombres genéricos o personajes.
- Devuelve estos lugares como "lugares_clave", asociando el tipo si aplica (“ciudad de Luthadel”).
- Ten en cuenta de que algunos lugares detectados pueden tener diferente nombre pero son el mismo lugar. Asegurate de saber si hay lugares repetidos. Si detectas algun lugar así, sobreescribre el nombre.

2) Extracción de lugares
- Solo incluye lugares válidos:
  A. Nombre propio o topónimo.
  B. Término geográfico genérico + nombre propio (“río Argento”, “bosque de Eldar”).
  C. Entidad geográfica o territorial con nombre específico (reino, región, ciudad, isla, lago, desierto, cordillera, fortaleza, torre, etc.).
- Excluye términos genéricos sin nombre propio (“bosques”, “pozos”, “montañas”, “ciudad”, “colina”, “campo”, etc.).
- Si un nombre propio corresponde a un tipo genérico implícito, combina: "<tipo> de <nombre>" (p.ej., “ciudad de Luthadel”).
- Mantén ortografía y capitalización original.

3) Extracción de relaciones
- Tipos válidos: NORTE_DE, SUR_DE, ESTE_DE, OESTE_DE, CERCA_DE, CONECTA, DENTRO_DE.
- Solo relaciones explícitas en el texto. No infieras ni inventes.

4) Chunking
- Analiza por fragmentos/capítulos y relaciona con los lugares clave.

========================
LISTA NEGRA DE GENÉRICOS (si aparecen sin nombre propio → EXCLUIR)
========================
ciudad, pueblo, aldea, villa, reino, imperio, región, provincia, distrito, condado,
capital, fortaleza, castillo, torre, templo, posada, campamento, bosque, río, lago,
montaña, cordillera, desierto, mar, océano, valle, puente, camino, puerto, muralla, mina, campo.

EJEMPLOS (permitido vs. prohibido):
- "ciudad de Luthadel" ✅
- "Luthadel" ✅
- "ciudad" ❌
- "bosque" ❌
- Si dudas, EXCLUYE.

========================
SALIDA JSON
========================
Devuelve SIEMPRE un JSON estricto:

{
  "lugares_clave": ["..."],
  "lugares": ["..."],
  "relaciones": [
    {"origen":"X","tipo":"NORTE_DE","destino":"Y"},
    {"origen":"A","tipo":"CERCA_DE","destino":"B"},
    {"origen":"C","tipo":"DENTRO_DE","destino":"D"},
    {"origen":"E","tipo":"CONECTA","destino":"F"}
  ]
}

Si no hay lugares válidos:
{"lugares_clave": [], "lugares": [], "relaciones": []}

========================
REGLAS ADICIONALES
========================
- Sé extremadamente estricto.
- Antes de devolver, revisa: ¿es un espacio físico? ¿tiene nombre propio o tipo+nombre?
- Nunca separes el tipo genérico del nombre propio.
- No uses conocimiento externo.
- Analiza desde el Capítulo 1 en adelante.
"""

# 6) Llamadas al LLM
results = []
pivot_raw = []  # lugares_clave reportados por el LLM (crudos)

for i, chunk in enumerate(relevant_chunks):
    print(f"Procesando {i+1}/{len(relevant_chunks)}...")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":chunk}],
            temperature=0.1,
            response_format={"type":"json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        results.append(data)
        pivot_raw.extend(data.get("lugares_clave", []))
    except Exception as e:
        print("Error:", e)
    time.sleep(0.6)  # anti rate-limit

# 7) Normalización y utilidades
ARTS = {"el","la","los","las","del","de","al","lo"}

def strip_accents(s:str)->str:
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")

def norm_place(s: str) -> str:
    t = strip_accents(s).strip().lower()
    parts = [p for p in t.split() if p]
    while parts and parts[0] in ARTS:
        parts = parts[1:]
    return " ".join(parts)

# ===== Filtro duro de genéricos sin nombre (ampliado) =====
GENERIC_TYPES = {
    "ciudad","pueblo","aldea","villa","reino","imperio","region","provincia","distrito",
    "condado","capital","fortaleza","castillo","torre","templo","posada","campamento",
    "bosque","rio","río","lago","montana","montañas","cordillera","desierto","mar","oceano",
    "océano","valle","puente","camino","puerto","muralla","mina","campo","colina",
    # evitar frases vagas como "tierra cercana a ..."
    "tierra","tierras","zona","zonas","territorio","nacion","nación","paraje","comarca"
}

# Artículos compuestos (para chequear inicios reales)
ARTS_MULTI = {"el","la","los","las","del","de la","de los","de las","al"}

def strip_leading_articles_phrase(t: str) -> str:
    """Quita artículos simples/compuestos al inicio de la frase normalizada."""
    t = t.strip()
    for a in sorted(ARTS_MULTI, key=len, reverse=True):
        if t.startswith(a + " "):
            return t[len(a)+1:].strip()
    return t

def starts_with_type_label(label: str, type_word: str) -> bool:
    """
    True si (tras quitar artículos) la etiqueta empieza por el 'type_word' exacto.
    Evita falsos positivos como 'tierra cercana a la península ...'.
    """
    ln = norm_place(label)
    ln = strip_leading_articles_phrase(ln)
    return re.match(rf"^{re.escape(type_word)}\b", ln) is not None

def tokens_core(label_norm: str):
    """Núcleo sin artículos ni genéricos ni 'de'."""
    toks = [t for t in label_norm.split() if t and t not in ARTS and t not in GENERIC_TYPES and t != "de"]
    return toks

def core_key(label_human: str) -> str:
    """Clave canónica del lugar (para fusionar 'ciudad de Luthadel' y 'Luthadel')."""
    cn = norm_place(label_human)
    core = tokens_core(cn)
    return " ".join(core)

def is_named_place(s: str) -> bool:
    """Acepta solo: (i) Nombre propio; (ii) <genérico> de <Nombre Propio>."""
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
def es_interior(lugar:str)->bool:
    exclude_keywords = {
        "ventana","pasillo","comedor","escalera","sala","guardarropa",
        "ropero","fondo","habitacion","cuarto","piso","puerta","cocina","bano",
        "baño","chimenea","mesa","silla","rincon","armario","armarios"
    }
    t = norm_place(lugar)
    return any(kw in t for kw in exclude_keywords)

VALID_TIPOS = {"NORTE_DE","SUR_DE","ESTE_DE","OESTE_DE","CERCA_DE","CONECTA","DENTRO_DE"}

# 8) Consolidar aplicando filtros duros
all_places, all_relations = set(), []

for r in results:
    # filtra lugares de la fuente
    for p in r.get("lugares", []):
        if is_named_place(p) and not es_interior(p):
            all_places.add(p)

    # filtra relaciones de la fuente (ambos extremos válidos)
    for rel in r.get("relaciones", []):
        o = (rel.get("origen") or "").strip()
        d = (rel.get("destino") or "").strip()
        t = (rel.get("tipo") or rel.get("relacion") or "").strip().upper()
        if t in VALID_TIPOS and is_named_place(o) and is_named_place(d) and o != d and not es_interior(o) and not es_interior(d):
            all_relations.append({"origen": o, "tipo": t, "destino": d})

# 9) Deduplicar y canonizar etiquetas humanas
canon2label = {}
cleaned_places = []
for p in all_places:
    c = norm_place(p)
    if c and c not in canon2label:
        canon2label[c] = p  # conservar primera etiqueta humana
        cleaned_places.append(p)

def remap(name: str) -> str:
    return canon2label.get(norm_place(name), name)

# normaliza relaciones ya filtradas
clean_relations = []
for r in all_relations:
    o, d = remap(r["origen"]), remap(r["destino"])
    if o in cleaned_places and d in cleaned_places and o != d:
        clean_relations.append({"origen": o, "tipo": r["tipo"], "destino": d})

# ====== Decidir regiones y pivotes en el EXTRACTOR ======
REGION_TYPES = {
    "reino","imperio","region","provincia","distrito","condado",
    "valle","bosque","lago","desierto","mar","océano","oceano",
    "cordillera","campo","isla","peninsula","península"
}
URBAN_TYPES = {"ciudad","pueblo","villa","capital","castillo","fortaleza","templo","puerto"}

def has_type_label(s: str, types:set) -> str|None:
    """Devuelve el tipo SOLO si la etiqueta COMIENZA con ese tipo (tras artículos)."""
    for k in types:
        if starts_with_type_label(s, k):
            return k
    return None

# Umbrales para considerar región
MIN_REGION_MENTIONS = 3            # región macro necesita >=3 menciones o contención
URBAN_REGION_MIN_MENTIONS = 15     # ciudad como 'región' si muy central/nombrada

# 1) Conteo de menciones en el texto completo (normalizado)
text_norm_full = unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("ascii").lower()
def count_mentions(label:str)->int:
    base = re.escape(norm_place(label))
    return len(re.findall(rf"\b{base}\b", text_norm_full))

mentions = {p: count_mentions(p) for p in cleaned_places}

# 2) Señales estructurales (desde relaciones)
deg = {p:0 for p in cleaned_places}
inside_as_region = {p:0 for p in cleaned_places}
near_conn = {p:0 for p in cleaned_places}
for rel in clean_relations:
    o, d, t = rel["origen"], rel["destino"], rel["tipo"].upper()
    deg[o]+=1; deg[d]+=1
    if t == "DENTRO_DE":
        inside_as_region[d]+=1
    if t in {"CERCA_DE","CONECTA"}:
        near_conn[o]+=1; near_conn[d]+=1

# 3) Etiquetas de tipo + score de pivote
def radius_hint(kind:str, n:int)->int:
    base = max(600, 20*n)
    if kind in {"imperio","reino","mar","océano","oceano"}: return max(24, base//5)
    if kind in {"region","provincia","valle","desierto","cordillera"}: return max(20, base//7)
    if kind in {"bosque","campo"}: return max(18, base//8)
    if kind in {"lago","isla","peninsula","península"}: return max(16, base//9)
    if kind in URBAN_TYPES: return max(16, base//10)  # ciudades pivote
    return max(16, base//10)

lugares_meta = {}
for p in cleaned_places:
    kind = has_type_label(p, REGION_TYPES) or has_type_label(p, URBAN_TYPES) or "otro"
    # score: pondera DENTRO_DE > cercanías/conexiones > menciones > grado
    score = (3*inside_as_region[p]) + (2*near_conn[p]) + (2*mentions[p]) + (1*deg[p])

    # decisión de región (estricta)
    is_region_flag = False
    if kind in REGION_TYPES:
        if (mentions[p] >= MIN_REGION_MENTIONS) or (inside_as_region[p] >= 1):
            is_region_flag = True
    elif kind in URBAN_TYPES:
        if (inside_as_region[p] >= 2) or (mentions[p] >= URBAN_REGION_MIN_MENTIONS):
            is_region_flag = True

    rh = radius_hint(kind, len(cleaned_places)) if is_region_flag else 0

    lugares_meta[p] = {
        "kind": kind,
        "is_region": bool(is_region_flag),
        "mentions": int(mentions[p]),
        "deg": int(deg[p]),
        "inside_in": int(inside_as_region[p]),
        "near_conn": int(near_conn[p]),
        "pivot_score": int(score),
        "radius_hint": int(rh)
    }

# 4) Seleccionar pivotes explícitos (fusionando variantes por 'core_key')
K = 6

# Agrupa labels por núcleo
groups = {}
for p in cleaned_places:
    k = core_key(p)
    if not k:
        continue
    groups.setdefault(k, []).append(p)

def pick_repr(labels: list[str]) -> str:
    """Elige una etiqueta representante: preferimos la que no inicia con genérico; si no, la más corta."""
    def starts_with_generic(lbl: str) -> bool:
        t = norm_place(lbl).split()
        return bool(t) and t[0] in GENERIC_TYPES
    candidates = [l for l in labels if not starts_with_generic(l)]
    if not candidates:
        candidates = labels
    return min(candidates, key=len)

# Métricas agregadas por grupo
group_scores = []
for k, labels in groups.items():
    rep = pick_repr(labels)
    ment = max(mentions.get(l, 0) for l in labels)      # máximo menciones
    ins  = sum(inside_as_region.get(l, 0) for l in labels)
    deg_ = sum(deg.get(l, 0) for l in labels)
    near = sum(near_conn.get(l, 0) for l in labels)
    score = (3*ins) + (2*near) + (2*ment) + (1*deg_)
    group_scores.append((score, ins, ment, rep))

group_scores.sort(reverse=True)
pivotes = []
seen_cores = set()
for score, ins, ment, rep in group_scores:
    core = core_key(rep)
    if core in seen_cores:
        continue
    if (ins >= 1) or (ment >= 8):     # umbral conservador
        pivotes.append(rep)
        seen_cores.add(core)
    if len(pivotes) >= K:
        break

# ========================
# Impresiones de control
# ========================
print("\n=== Pivotes seleccionados (extractor) ===")
for i,p in enumerate(pivotes,1):
    m = lugares_meta[p]
    print(f"{i}. {p}  (score={m['pivot_score']}, menciones={m['mentions']}, dentro_de={m['inside_in']}, grado={m['deg']})")

# Regiones detectadas (macro vs urbanas elevadas)
regiones_macro, regiones_urbanas = [], []
for p, meta in lugares_meta.items():
    if meta["is_region"]:
        if meta["kind"] in REGION_TYPES:
            regiones_macro.append(p)
        else:
            regiones_urbanas.append(p)

def _orden(p):
    m = lugares_meta[p]
    return (m["inside_in"], m["mentions"], m["pivot_score"], p.lower())

regiones_macro.sort(key=_orden, reverse=True)
regiones_urbanas.sort(key=_orden, reverse=True)

print("\n=== Regiones detectadas ===")
if not regiones_macro and not regiones_urbanas:
    print("(ninguna)")
else:
    if regiones_macro:
        print("\n[Macro-regiones]")
        for i, p in enumerate(regiones_macro, 1):
            m = lugares_meta[p]
            print(f"{i}. {p}  - tipo={m['kind']}  menciones={m['mentions']}  contiene={m['inside_in']}  radio_hint={m['radius_hint']}")
    if regiones_urbanas:
        print("\n[Regiones urbanas (elevadas por pivote)]")
        for i, p in enumerate(regiones_urbanas, 1):
            m = lugares_meta[p]
            print(f"{i}. {p}  - tipo={m['kind']}  menciones={m['mentions']}  contiene={m['inside_in']}  radio_hint={m['radius_hint']}")

# ========================
# Guardado ÚNICO del JSON
# ========================
# --- Filtro de lugares aislados (sin relaciones) ---
INCLUDE_ISOLATES_POLICY = "smart"   # "none" (elimina todos), "all" (no filtra), "smart" (recomendado)
MENTIONS_MIN_KEEP = 8               # si aparece ≥8 veces, se mantiene aunque no tenga relaciones
REGION_MIN_MENTIONS = 1             # regiones aisladas necesitan ≥1 mención para quedarse

# Grado por nodo
deg = {p: 0 for p in cleaned_places}
for rel in clean_relations:
    o, d = rel["origen"], rel["destino"]
    if o in deg: deg[o] += 1
    if d in deg: deg[d] += 1

def keep_place(p: str) -> bool:
    if INCLUDE_ISOLATES_POLICY == "all":
        return True
    if INCLUDE_ISOLATES_POLICY == "none":
        return deg.get(p, 0) > 0

    # "smart"
    if deg.get(p, 0) > 0:
        return True  # tiene alguna relación
    if p in pivotes:
        return True  # pivote siempre entra
    meta_p = lugares_meta.get(p, {})
    if meta_p.get("is_region") and meta_p.get("mentions", 0) >= REGION_MIN_MENTIONS:
        return True  # región con mínima presencia
    if meta_p.get("mentions", 0) >= MENTIONS_MIN_KEEP:
        return True  # muy mencionado, aunque aún sin vínculos
    return False

filtered_places = [p for p in cleaned_places if keep_place(p)]
kept = set(filtered_places)
filtered_relations = [r for r in clean_relations if r["origen"] in kept and r["destino"] in kept]
excluidos = [p for p in cleaned_places if p not in kept]

print(f"\nFiltrado de aislados: mantuve {len(filtered_places)} lugares, descarté {len(excluidos)}.")
if excluidos:
    print("Descartados (muestra):", ", ".join(excluidos[:10]), ("..." if len(excluidos) > 10 else ""))

# >>> Al guardar, usa las listas filtradas <<<
with open("map_relations.json","w",encoding="utf-8") as f:
    json.dump({
        "lugares": filtered_places,
        "relaciones": filtered_relations,
        "lugares_meta": lugares_meta,
        "pivotes": pivotes,
        # opcional: por trazabilidad
        "excluidos": excluidos
    }, f, indent=2, ensure_ascii=False)


print(f"\nLugares: {len(cleaned_places)} | Relaciones: {len(clean_relations)} guardadas en map_relations.json")

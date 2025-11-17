# solver_grafos.py
import json
import random
import re
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from z3 import *

# ------------------------------------------------------------
# 1. Cargar datos desde JSON (con meta y pivotes si existen)
# ------------------------------------------------------------
with open("map_relations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

clean_places = data.get("lugares", [])
clean_relations = data.get("relaciones", [])
meta = data.get("lugares_meta", {})        # dict opcional por lugar
pivotes_list = data.get("pivotes", [])     # lista opcional
regiones_from_extractor = data.get("regiones", {})  # opcional: {"macro":[...], "urbanas":[...]}

print(f"Cargados {len(clean_places)} lugares y {len(clean_relations)} relaciones.")
if not meta:
    print("⚠️ Aviso: 'lugares_meta' no viene en el JSON. Usaré defaults.")
if not pivotes_list:
    print("⚠️ Aviso: 'pivotes' no viene en el JSON. Anclas de pivotes deshabilitadas.")

# ------------------------------------------------------------
# 2. Normalización auxiliar
# ------------------------------------------------------------
def norm_place(s: str) -> str:
    return (s or "").strip().lower()

# Índices robustos (por si hubiera variaciones menores)
meta_by_norm = {norm_place(k): v for k, v in meta.items()}
name_by_norm = {norm_place(p): p for p in clean_places}
pivots_norm = {norm_place(p) for p in pivotes_list}

# ------------------------------------------------------------
# 3. API que respeta lo definido por el EXTRACTOR
# ------------------------------------------------------------
def is_region(name: str) -> bool:
    """Región si y solo si el extractor lo marcó como tal."""
    m = meta_by_norm.get(norm_place(name), {})
    return bool(m.get("is_region", False))

def radio_hint(name: str, side: int) -> int:
    """Tamaño de región sugerido por el extractor; fallback moderado."""
    m = meta_by_norm.get(norm_place(name), {})
    val = int(m.get("radius_hint", 0))
    if val > 0:
        return val
    return max(16, side // 10)

def pivot_score(name: str) -> int:
    m = meta_by_norm.get(norm_place(name), {})
    return int(m.get("pivot_score", 0))

def is_pivot(name: str) -> bool:
    return norm_place(name) in pivots_norm

# ------------------------------------------------------------
# 4. Parámetros del lienzo y distancias
# ------------------------------------------------------------
N = max(1, len(clean_places))
SIDE = max(600, 20 * N)   # lienzo amplio
WIDTH, HEIGHT = SIDE, SIDE

MARGIN_DIR   = max(8, SIDE // 28)   # N/S/E/O
DIST_CLOSE   = max(14, SIDE // 9)   # CERCA_DE
DIST_CONNECT = max(16, SIDE // 8)   # CONECTA
MIN_SEP      = max(8, SIDE // 22)   # separación mínima nodos

REGION_CLEARANCE = max(6, SIDE // 40)  # margen extra entre regiones
SOLVE_TIMEOUT_MS = 2500
REL_LIMIT = 900

# ------------------------------------------------------------
# 5. Utilidades de restricciones Z3
# ------------------------------------------------------------
def add_abs_le(s, expr, bound):
    s.add(expr <= bound, expr >= -bound)

def add_min_sep(s, dx, dy, d):
    s.add(Or(dx >= d, dx <= -d, dy >= d, dy <= -d))

def add_nonoverlap_var(s, dx, dy, bound_expr):
    """|dx| >= bound_expr o |dy| >= bound_expr (separación L∞ variable)."""
    s.add(Or(dx >= bound_expr, dx <= -bound_expr, dy >= bound_expr, dy <= -bound_expr))

def rel_to_constraints(A, B, tipo, x, y, R):
    dx, dy = x[A] - x[B], y[A] - y[B]
    cons = []
    if   tipo == "NORTE_DE": cons.append(dy >= MARGIN_DIR)
    elif tipo == "SUR_DE":   cons.append(dy <= -MARGIN_DIR)
    elif tipo == "ESTE_DE":  cons.append(dx >= MARGIN_DIR)
    elif tipo == "OESTE_DE": cons.append(dx <= -MARGIN_DIR)
    elif tipo == "CERCA_DE":
        cons += [("abs_le", dx, DIST_CLOSE), ("abs_le", dy, DIST_CLOSE)]
    elif tipo == "CONECTA":
        cons += [("abs_le", dx, DIST_CONNECT), ("abs_le", dy, DIST_CONNECT)]
    elif tipo == "DENTRO_DE":
        if is_region(B):
            cons += [("abs_le_var", dx, R[B]), ("abs_le_var", dy, R[B])]
    return cons

def priority(rel):
    t = rel["tipo"].upper()
    if t in {"NORTE_DE", "SUR_DE", "ESTE_DE", "OESTE_DE"}: return 0
    if t == "DENTRO_DE": return 1
    if t == "CONECTA":   return 2
    if t == "CERCA_DE":  return 3
    return 4

# ------------------------------------------------------------
# 6. Solver incremental
# ------------------------------------------------------------
def solve_with_z3(lugares, relaciones):
    rels = sorted(relaciones[:REL_LIMIT], key=priority)
    s = Solver(); s.set(timeout=SOLVE_TIMEOUT_MS)
    x, y, R = {}, {}, {}

    # contención explícita (para permitir solapamiento como contención)
    contains = {(r["origen"], r["destino"]) for r in relaciones if r["tipo"].upper() == "DENTRO_DE"}

    # variables
    for i, p in enumerate(lugares):
        x[p], y[p], R[p] = Int(f"x_{i}"), Int(f"y_{i}"), Int(f"R_{i}")
        s.add(And(x[p] >= 0, x[p] <= WIDTH, y[p] >= 0, y[p] <= HEIGHT))
        if is_region(p):
            s.add(R[p] >= radio_hint(p, SIDE), R[p] <= max(WIDTH, HEIGHT))
        else:
            s.add(R[p] == 0)

    # anclas de pivotes si existen
    pivs = [p for p in lugares if is_pivot(p)]
    if pivs:
        # ordena por score y ancla hasta 3 para dar estructura
        pivs.sort(key=lambda p: (pivot_score(p), p.lower()), reverse=True)
        if len(pivs) >= 1: s.add(x[pivs[0]] == WIDTH // 2,     y[pivs[0]] == HEIGHT // 2)
        if len(pivs) >= 2: s.add(x[pivs[1]] == (WIDTH * 5)//6, y[pivs[1]] == HEIGHT // 2)
        if len(pivs) >= 3: s.add(x[pivs[2]] == (WIDTH * 1)//6, y[pivs[2]] == HEIGHT // 2)
    else:
        # fallback: una ancla suave si no hay pivotes
        if len(lugares) >= 1:
            p0 = lugares[0]; s.add(x[p0] == WIDTH // 2, y[p0] == HEIGHT // 2)

    # separación mínima global
    for i in range(len(lugares)):
        for j in range(i + 1, len(lugares)):
            A, B = lugares[i], lugares[j]
            add_min_sep(s, x[A] - x[B], y[A] - y[B], MIN_SEP)

    # NO OVERLAP entre REGIONES salvo contención explícita
    region_nodes = [p for p in lugares if is_region(p)]
    for i in range(len(region_nodes)):
        for j in range(i + 1, len(region_nodes)):
            A, B = region_nodes[i], region_nodes[j]
            if (A, B) in contains or (B, A) in contains:
                continue
            dx, dy = x[A] - x[B], y[A] - y[B]
            bound_expr = R[A] + R[B] + IntVal(REGION_CLEARANCE)
            add_nonoverlap_var(s, dx, dy, bound_expr)

    # añadir relaciones por prioridad (aceptación incremental)
    for rel in rels:
        A, B, t = rel["origen"], rel["destino"], rel["tipo"].upper()
        if A not in x or B not in x:
            continue
        cons = rel_to_constraints(A, B, t, x, y, R)
        if not cons:
            continue
        s.push()
        for c in cons:
            if isinstance(c, tuple):
                tag = c[0]
                if tag == "abs_le":
                    _, expr, bound = c; add_abs_le(s, expr, bound)
                elif tag == "abs_le_var":
                    _, expr, varb = c; s.add(expr <= varb, expr >= -varb)
            else:
                s.add(c)
        if s.check() == sat:
            s.pop()
            for c in cons:
                if isinstance(c, tuple):
                    tag = c[0]
                    if tag == "abs_le":
                        _, expr, bound = c; add_abs_le(s, expr, bound)
                    elif tag == "abs_le_var":
                        _, expr, varb = c; s.add(expr <= varb, expr >= -varb)
                else:
                    s.add(c)
        else:
            s.pop()

    # modelo final
    if s.check() != sat:
        print("⚠️ Z3 no pudo satisfacer todas las restricciones. Se generará un layout parcial.")
        try:
            m = s.model()
        except Exception:
            coords = {p: {"x": random.randint(0, WIDTH), "y": random.randint(0, HEIGHT), "r": 0} for p in lugares}
            rel_eval = [{"origen": r["origen"], "tipo": r["tipo"], "destino": r["destino"], "satisface": False} for r in relaciones]
            return {"coords": coords, "CSR": 0.0, "rel_eval": rel_eval, "width": WIDTH, "height": HEIGHT}
    m = s.model()

    coords = {p: {"x": int(m.eval(x[p]).as_long()),
                  "y": int(m.eval(y[p]).as_long()),
                  "r": int(m.eval(R[p]).as_long())}
              for p in lugares}

    def satisfied(rel):
        A, B, t = rel["origen"], rel["destino"], rel["tipo"].upper()
        dx, dy = coords[A]["x"] - coords[B]["x"], coords[A]["y"] - coords[B]["y"]
        ok = True
        if t == "NORTE_DE": ok &= (dy >= MARGIN_DIR)
        elif t == "SUR_DE": ok &= (dy <= -MARGIN_DIR)
        elif t == "ESTE_DE": ok &= (dx >= MARGIN_DIR)
        elif t == "OESTE_DE": ok &= (dx <= -MARGIN_DIR)
        elif t == "CERCA_DE": ok &= (abs(dx) <= DIST_CLOSE and abs(dy) <= DIST_CLOSE)
        elif t == "CONECTA":  ok &= (abs(dx) <= DIST_CONNECT and abs(dy) <= DIST_CONNECT)
        elif t == "DENTRO_DE":
            if is_region(B): ok &= (abs(dx) <= coords[B]["r"] and abs(dy) <= coords[B]["r"])
            else: ok = False
        return bool(ok)

    rel_eval = [{"origen": r["origen"], "tipo": r["tipo"], "destino": r["destino"],
                 "satisface": satisfied(r)} for r in relaciones]
    CSR = sum(1 for r in rel_eval if r["satisface"]) / max(1, len(rel_eval))
    return {"coords": coords, "CSR": CSR, "rel_eval": rel_eval, "width": WIDTH, "height": HEIGHT}

solution = solve_with_z3(clean_places, clean_relations)
print(f"CSR: {solution['CSR']:.3f}")

# Mostrar pivotes usados (labels humanos)
if pivotes_list:
    print("\n=== Pivotes usados (desde extractor) ===")
    for i, p in enumerate(pivotes_list, 1):
        print(f"{i}. {p}")
else:
    print("\n(No hay pivotes definidos en el JSON)")

# Mostrar resumen de regiones (desde extractor si viene; si no, desde meta)
print("\n=== Regiones usadas por el solver ===")
if regiones_from_extractor:
    macro = regiones_from_extractor.get("macro", [])
    urban = regiones_from_extractor.get("urbanas", [])
    if not macro and not urban:
        print("(ninguna)")
    else:
        if macro:
            print("[Macro-regiones]")
            for i, p in enumerate(macro, 1):
                print(f"{i}. {p}")
        if urban:
            print("[Regiones urbanas (elevadas por pivote)]")
            for i, p in enumerate(urban, 1):
                print(f"{i}. {p}")
else:
    # Derivar de meta (is_region=True)
    regs = [p for p in clean_places if is_region(p)]
    if regs:
        for i, p in enumerate(regs, 1):
            print(f"{i}. {p}")
    else:
        print("(ninguna)")

# ------------------------------------------------------------
# 7. Graficar resultado (sin nodos para regiones)
# ------------------------------------------------------------
ZOOM = 6
FIGSIZE = (20, 14)
DPI = 240
NODE_SIZE = 950
FONT_NODES = 9
FONT_EDGES = 10
SAVE_SVG = True

# Colores (azules para regiones, si quieres)
REGION_FACE_COLOR = "#d0e7ff"
REGION_EDGE_COLOR = "#2b6cb0"
REGION_LABEL_COLOR = "#1f4e79"
REGION_ALPHA      = 0.28

pos2 = {p: (solution["coords"][p]["x"], solution["coords"][p]["y"]) for p in clean_places}
pos_zoom = {p: (x * ZOOM, y * ZOOM) for p, (x, y) in pos2.items()}

# --- NUEVO: trabajar solo con nodos NO-región para el grafo
non_region_nodes = [p for p in clean_places if not is_region(p)]

# Calcular satisfacción por par, pero solo para pares no-región
pair_sat = {}
for r in solution["rel_eval"]:
    a, b = r["origen"], r["destino"]
    if is_region(a) or is_region(b):
        continue  # omitimos cualquier relación donde participe una región
    key = tuple(sorted((a, b)))
    pair_sat[key] = pair_sat.get(key, False) or r["satisface"]

# Grafo sin regiones
G2 = nx.Graph()
for p in non_region_nodes:
    G2.add_node(p)

for r in clean_relations:
    a, b = r["origen"], r["destino"]
    if is_region(a) or is_region(b):
        continue  # no dibujamos aristas hacia/desde regiones
    key = tuple(sorted((a, b)))
    if not G2.has_edge(a, b):
        G2.add_edge(a, b, tipo=r["tipo"], satisface=pair_sat.get(key, False))

plt.figure(figsize=FIGSIZE, dpi=DPI)
ax = plt.gca()

# Dibujar regiones como SÓLO círculos + etiqueta (sin nodo)
for p in clean_places:
    r = solution["coords"][p]["r"]
    if r > 0:  # es región (el solver ya puso R>0)
        cx, cy = pos2[p]
        circ = Circle(
            (cx * ZOOM, cy * ZOOM), r * ZOOM,
            facecolor=REGION_FACE_COLOR,
            edgecolor=REGION_EDGE_COLOR,
            alpha=REGION_ALPHA,
            linewidth=1.8
        )
        ax.add_patch(circ)
        ax.text(
            cx * ZOOM, cy * ZOOM, p,
            fontsize=13, weight="bold", ha="center", va="center",
            color=REGION_LABEL_COLOR,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2)
        )

# Nodos (solo no-región), resaltando pivotes si quieres
node_edges = []
node_colors = []
for p in non_region_nodes:
    if is_pivot(p):
        node_edges.append("#1f3d1f")  # borde más oscuro
        node_colors.append("#b8e6b8")
    else:
        node_edges.append("#2d7a41")
        node_colors.append("#b8e6b8")

nx.draw_networkx_nodes(
    G2, pos_zoom, nodelist=non_region_nodes,
    node_color=node_colors, node_size=NODE_SIZE,
    edgecolors=node_edges, linewidths=1.2
)

# Aristas (solo entre no-región)
edge_colors = ["green" if G2[u][v]["satisface"] else "red" for u, v in G2.edges()]
nx.draw_networkx_edges(G2, pos_zoom, edge_color=edge_colors, width=1.8, alpha=0.9)

# Etiquetas (solo no-región)
labels_regular = {p: p for p in non_region_nodes}
nx.draw_networkx_labels(
    G2, pos_zoom, labels=labels_regular, font_size=FONT_NODES,
    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0)
)

# Etiquetas de aristas violadas (solo no-región)
violadas_labels = {(u, v): G2[u][v]["tipo"] for u, v in G2.edges() if not G2[u][v]["satisface"]}
nx.draw_networkx_edge_labels(
    G2, pos_zoom, edge_labels=violadas_labels,
    font_size=FONT_EDGES, bbox=dict(alpha=0.35, facecolor="white", edgecolor="none")
)

plt.title(f"Mapa resuelto por Z3 · CSR={solution['CSR']:.3f}\nVerde=satisfechas · Rojo=violadas")
plt.axis("equal"); plt.axis("off")
plt.tight_layout()

if SAVE_SVG:
    plt.savefig("mapa.svg", format="svg", bbox_inches="tight")

plt.show()

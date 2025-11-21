# solver_grafos.py
import json
import random
import networkx as nx
import matplotlib.pyplot as plt
from z3 import *

# ------------------------------------------------------------
# 1. Cargar datos desde JSON (con meta y pivotes si existen)
# ------------------------------------------------------------
with open("map_relations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

clean_places = data.get("lugares", [])
clean_relations = data.get("relaciones", [])
meta = data.get("lugares_meta", {})        # meta por lugar (opcional)

# pivotes: distinguir entre “no viene” y “lista vacía”
if "pivotes" in data:
    pivotes_list = data.get("pivotes", [])
    if not pivotes_list:
        print("⚠️ Aviso: 'pivotes' está presente pero vacío. Anclas de pivotes deshabilitadas.")
else:
    pivotes_list = []
    print("⚠️ Aviso: 'pivotes' no viene en el JSON. Anclas de pivotes deshabilitadas.")

print(f"Cargados {len(clean_places)} lugares y {len(clean_relations)} relaciones.")
if not meta:
    print("⚠️ Aviso: 'lugares_meta' no viene en el JSON. Usaré defaults.")

# ------------------------------------------------------------
# 2. Normalización auxiliar
# ------------------------------------------------------------
def norm_place(s: str) -> str:
    return (s or "").strip().lower()

meta_by_norm = {norm_place(k): v for k, v in meta.items()}
pivots_norm = {norm_place(p) for p in pivotes_list}

# ------------------------------------------------------------
# 3. API desde meta
# ------------------------------------------------------------
def pivot_score(name: str) -> int:
    m = meta_by_norm.get(norm_place(name), {})
    return int(m.get("pivot_score", 0))

def is_pivot(name: str) -> bool:
    return norm_place(name) in pivots_norm

# ------------------------------------------------------------
# 4. Parámetros del lienzo y distancias
# ------------------------------------------------------------
N = max(1, len(clean_places))
SIDE = max(600, 20 * N)
WIDTH, HEIGHT = SIDE, SIDE

MARGIN_DIR   = max(8, SIDE // 28)   # N/S/E/O
DIST_CLOSE   = max(14, SIDE // 9)   # CERCA_DE
DIST_CONNECT = max(16, SIDE // 8)   # CONECTA
MIN_SEP      = max(8, SIDE // 22)   # separación mínima nodos

SOLVE_TIMEOUT_MS = 2500
REL_LIMIT = 900

# ------------------------------------------------------------
# 5. Utilidades Z3
# ------------------------------------------------------------
def add_abs_le(s, expr, bound):
    s.add(expr <= bound, expr >= -bound)

def add_min_sep(s, dx, dy, d):
    s.add(Or(dx >= d, dx <= -d, dy >= d, dy <= -d))

def rel_to_constraints(A, B, tipo, x, y):
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
    return cons

def priority(rel):
    t = rel["tipo"].upper()
    if t in {"NORTE_DE", "SUR_DE", "ESTE_DE", "OESTE_DE"}: return 0
    if t == "CERCA_DE": return 1
    if t == "CONECTA":  return 2
    return 3

# ------------------------------------------------------------
# 6. Solver incremental Z3
# ------------------------------------------------------------
def solve_with_z3(lugares, relaciones):
    rels = sorted(relaciones[:REL_LIMIT], key=priority)
    s = Solver()
    s.set(timeout=SOLVE_TIMEOUT_MS)
    x, y = {}, {}

    # Variables de posición
    for i, p in enumerate(lugares):
        x[p], y[p] = Int(f"x_{i}"), Int(f"y_{i}")
        s.add(And(x[p] >= 0, x[p] <= WIDTH,
                  y[p] >= 0, y[p] <= HEIGHT))

    # Anclas de pivotes
    pivs = [p for p in lugares if is_pivot(p)]
    if pivs:
        pivs.sort(key=lambda p: (pivot_score(p), p.lower()), reverse=True)
        if len(pivs) >= 1: s.add(x[pivs[0]] == WIDTH // 2,     y[pivs[0]] == HEIGHT // 2)
        if len(pivs) >= 2: s.add(x[pivs[1]] == (WIDTH * 5)//6, y[pivs[1]] == HEIGHT // 2)
        if len(pivs) >= 3: s.add(x[pivs[2]] == (WIDTH * 1)//6, y[pivs[2]] == HEIGHT // 2)
    else:
        if len(lugares) >= 1:
            p0 = lugares[0]
            s.add(x[p0] == WIDTH // 2, y[p0] == HEIGHT // 2)

    # Separación mínima global
    for i in range(len(lugares)):
        for j in range(i + 1, len(lugares)):
            A, B = lugares[i], lugares[j]
            add_min_sep(s, x[A] - x[B], y[A] - y[B], MIN_SEP)

    # Añadir relaciones incrementalmente
    for rel in rels:
        A, B, t = rel["origen"], rel["destino"], rel["tipo"].upper()
        if A not in x or B not in x:
            continue
        cons = rel_to_constraints(A, B, t, x, y)
        if not cons:
            continue
        s.push()
        for c in cons:
            if isinstance(c, tuple):
                tag = c[0]
                if tag == "abs_le":
                    _, expr, bound = c
                    add_abs_le(s, expr, bound)
            else:
                s.add(c)
        if s.check() == sat:
            s.pop()
            # consolidar
            for c in cons:
                if isinstance(c, tuple):
                    tag = c[0]
                    if tag == "abs_le":
                        _, expr, bound = c
                        add_abs_le(s, expr, bound)
                else:
                    s.add(c)
        else:
            s.pop()

    # Modelo final
    if s.check() != sat:
        print("⚠️ Z3 no pudo satisfacer todas las restricciones. Se generará un layout parcial.")
        try:
            m = s.model()
        except Exception:
            coords = {p: {"x": random.randint(0, WIDTH), "y": random.randint(0, HEIGHT)} for p in lugares}
            rel_eval = [{"origen": r["origen"], "tipo": r["tipo"], "destino": r["destino"], "satisface": False}
                        for r in relaciones]
            return {"coords": coords, "CSR": 0.0, "rel_eval": rel_eval,
                    "width": WIDTH, "height": HEIGHT}
    m = s.model()

    coords = {
        p: {
            "x": int(m.eval(x[p]).as_long()),
            "y": int(m.eval(y[p]).as_long())
        }
        for p in lugares
    }

    def satisfied(rel):
        A, B, t = rel["origen"], rel["destino"], rel["tipo"].upper()
        dx, dy = coords[A]["x"] - coords[B]["x"], coords[A]["y"] - coords[B]["y"]
        ok = True
        if t == "NORTE_DE": ok &= (dy >= MARGIN_DIR)
        elif t == "SUR_DE": ok &= (dy <= -MARGIN_DIR)
        elif t == "ESTE_DE": ok &= (dx >= MARGIN_DIR)
        elif t == "OESTE_DE": ok &= (dx <= -MARGIN_DIR)
        elif t == "CERCA_DE": ok &= (abs(dx) <= DIST_CLOSE and abs(dy) <= DIST_CLOSE)
        elif t == "CONECTA": ok &= (abs(dx) <= DIST_CONNECT and abs(dy) <= DIST_CONNECT)
        return bool(ok)

    rel_eval = [{
        "origen": r["origen"],
        "tipo": r["tipo"],
        "destino": r["destino"],
        "satisface": satisfied(r)
    } for r in relaciones]

    CSR = sum(1 for r in rel_eval if r["satisface"]) / max(1, len(rel_eval))
    return {"coords": coords, "CSR": CSR, "rel_eval": rel_eval,
            "width": WIDTH, "height": HEIGHT}

# ------------------------------------------------------------
# 7. Resolver + guardar
# ------------------------------------------------------------
solution = solve_with_z3(clean_places, clean_relations)
print(f"CSR: {solution['CSR']:.3f}")

with open("solution.json", "w", encoding="utf-8") as f:
    json.dump(solution, f, indent=2, ensure_ascii=False)
print("Solución (coordenadas del solver) guardada en solution.json")

if pivotes_list:
    print("\n=== Pivotes usados (desde extractor) ===")
    for i, p in enumerate(pivotes_list, 1):
        print(f"{i}. {p}")
else:
    print("\n(No hay pivotes definidos en el JSON)")

# ------------------------------------------------------------
# 8. Graficar usando coords del solver
# ------------------------------------------------------------
FIGSIZE = (20, 10)
DPI = 240
NODE_SIZE = 700
FONT_NODES = 8
FONT_EDGES = 10
SAVE_SVG = True

# Todos los nodos se tratan como nodos estándar
non_region_nodes = list(clean_places)

# map (origen,destino) -> ¿alguna relación satisfecha?
pair_sat = {}
for r in solution["rel_eval"]:
    a, b = r["origen"], r["destino"]
    key = tuple(sorted((a, b)))
    pair_sat[key] = pair_sat.get(key, False) or r["satisface"]

G2 = nx.Graph()
for p in non_region_nodes:
    G2.add_node(p)

for r in clean_relations:
    a, b = r["origen"], r["destino"]
    key = tuple(sorted((a, b)))
    if not G2.has_edge(a, b):
        G2.add_edge(a, b, tipo=r["tipo"], satisface=pair_sat.get(key, False))

plt.figure(figsize=FIGSIZE, dpi=DPI)
ax = plt.gca()

# Coordenadas del solver (opcionalmente escaladas)
ZOOM = 1.0
pos_zoom = {
    p: (solution["coords"][p]["x"] * ZOOM,
        solution["coords"][p]["y"] * ZOOM)
    for p in non_region_nodes
}

# Nodos conectados vs aislados
deg_dict = dict(G2.degree())
nodes_conectados = [p for p in non_region_nodes if deg_dict.get(p, 0) > 0]
nodes_aislados   = [p for p in non_region_nodes if deg_dict.get(p, 0) == 0]

# --- conectados ---
node_edges_con = []
node_colors_con = []
for p in nodes_conectados:
    if is_pivot(p):
        node_edges_con.append("#1f3d1f")  # borde más oscuro para pivote
        node_colors_con.append("#b8e6b8")
    else:
        node_edges_con.append("#2d7a41")
        node_colors_con.append("#b8e6b8")

nx.draw_networkx_nodes(
    G2, pos_zoom, nodelist=nodes_conectados,
    node_color=node_colors_con, node_size=NODE_SIZE,
    edgecolors=node_edges_con, linewidths=1.2
)

# --- aislados ---
DIBUJAR_AISLADOS = True

if DIBUJAR_AISLADOS and nodes_aislados:
    NODE_SIZE_ISO = int(NODE_SIZE * 0.5)
    node_edges_iso = ["#2d7a41"] * len(nodes_aislados)
    node_colors_iso = ["#cfe9cf"] * len(nodes_aislados)

    nx.draw_networkx_nodes(
        G2, pos_zoom, nodelist=nodes_aislados,
        node_color=node_colors_iso, node_size=NODE_SIZE_ISO,
        edgecolors=node_edges_iso, linewidths=0.8
    )

# Aristas
edge_colors = ["green" if G2[u][v]["satisface"] else "red" for u, v in G2.edges()]
nx.draw_networkx_edges(
    G2, pos_zoom,
    edge_color=edge_colors, width=1.8, alpha=0.9
)

# Etiquetas: sólo para conectados
labels_regular = {p: p for p in nodes_conectados}
nx.draw_networkx_labels(
    G2, pos_zoom, labels=labels_regular, font_size=FONT_NODES,
    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0)
)

# Etiquetas de aristas violadas
violadas_labels = {
    (u, v): G2[u][v]["tipo"] for u, v in G2.edges()
    if not G2[u][v]["satisface"]
}
nx.draw_networkx_edge_labels(
    G2, pos_zoom, edge_labels=violadas_labels,
    font_size=FONT_EDGES,
    bbox=dict(alpha=0.35, facecolor="white", edgecolor="none")
)

plt.title(f"Mapa de Luthadel · CSR={solution['CSR']:.3f}\nVerde=satisfechas · Rojo=violadas")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()

if SAVE_SVG:
    plt.savefig("mapa.svg", format="svg", bbox_inches="tight")

plt.show()

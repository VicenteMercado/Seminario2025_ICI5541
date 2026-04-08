# solver_grafos.py
import json
import random
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from z3 import Solver, Int, And, Or, sat  # type: ignore[import-untyped]

_ROOT = Path(__file__).resolve().parent.parent
_JSON_DIR = _ROOT / "json"
_IMG_DIR = _ROOT / "img"

# ------------------------------------------------------------
# 1. Cargar inecuaciones preprocesadas
# ------------------------------------------------------------
with open(_JSON_DIR / "inequalities.json", "r", encoding="utf-8") as f:
    data = json.load(f)

clean_places = data["lugares"]
ineq_constraints = data["constraints"]
pivotes_list = data.get("pivotes", [])
params = data["params"]

WIDTH = params["WIDTH"]
HEIGHT = params["HEIGHT"]
MARGIN_DIR = params["MARGIN_DIR"]
RADIUS_DIR = params["RADIUS_DIR"]
DIST_CLOSE = params["DIST_CLOSE"]
DIST_CONNECT = params["DIST_CONNECT"]
MIN_SEP = params["MIN_SEP"]
METROS_POR_PIXEL = params.get("metros_por_pixel")

SOLVE_TIMEOUT_MS = 30000
REL_LIMIT = 900


# ------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------
def norm_place(s: str) -> str:
    return (s or "").strip().lower()


pivots_norm = {norm_place(p) for p in pivotes_list}


def is_pivot(name: str) -> bool:
    return norm_place(name) in pivots_norm


def add_abs_le(s, expr, bound):
    s.add(expr <= bound, expr >= -bound)


def add_min_sep(s, dx, dy, d):
    s.add(Or(dx >= d, dx <= -d, dy >= d, dy <= -d))


# ------------------------------------------------------------
# 3. Solver incremental usando JSON de inecuaciones
# ------------------------------------------------------------
def apply_constraint(s, c, dx, dy):
    """Aplica una constraint serializada al solver."""
    kind = c["kind"]
    if kind == "directional":
        expr = dx if c["dir_expr"] == "dx" else dy
        op = c["dir_op"]
        value = c["dir_value"]
        if op == ">=":
            s.add(expr >= value)
        elif op == "<=":
            s.add(expr <= value)
        add_abs_le(s, dx, c["dx_max"])
        add_abs_le(s, dy, c["dy_max"])
    elif kind == "abs_box":
        add_abs_le(s, dx, c["dx_max"])
        add_abs_le(s, dy, c["dy_max"])
    elif kind == "ineq":
        expr = dx if c["expr"] == "dx" else dy
        op = c["op"]
        value = c["value"]
        if op == ">=":
            s.add(expr >= value)
        elif op == "<=":
            s.add(expr <= value)


def priority(item):
    t = item["tipo"].upper()
    if t in {"NORTE_DE", "SUR_DE", "ESTE_DE", "OESTE_DE"}: return 0
    if t == "CERCA_DE": return 1
    if t == "CONECTA":  return 2
    return 3


def solve_with_z3(lugares, constraints):
    items = sorted(constraints[:REL_LIMIT], key=priority)
    s = Solver()
    s.set(timeout=SOLVE_TIMEOUT_MS)

    x, y = {}, {}

    # Variables de posición
    for i, p in enumerate(lugares):
        x[p] = Int(f"x_{i}")
        y[p] = Int(f"y_{i}")
        s.add(And(x[p] >= 0, x[p] <= WIDTH,
                  y[p] >= 0, y[p] <= HEIGHT))

    # --------------------------------------------------------
    # Anclas de pivotes
    # --------------------------------------------------------
    pivs = [p for p in lugares if is_pivot(p)]
    if pivs:
        if len(pivs) >= 1: s.add(x[pivs[0]] == WIDTH // 2,     y[pivs[0]] == HEIGHT // 2)
        if len(pivs) >= 2: s.add(x[pivs[1]] == (WIDTH * 5)//6, y[pivs[1]] == HEIGHT // 2)
        if len(pivs) >= 3: s.add(x[pivs[2]] == (WIDTH * 1)//6, y[pivs[2]] == HEIGHT // 2)
    else:
        if lugares:
            p0 = lugares[0]
            s.add(x[p0] == WIDTH // 2, y[p0] == HEIGHT // 2)

    # --------------------------------------------------------
    # Separación mínima (solo pares conectados por relación)
    # --------------------------------------------------------
    sep_pairs = set()
    for item in items:
        A, B = item["origen"], item["destino"]
        if A in x and B in x:
            pair = (min(A, B), max(A, B))
            sep_pairs.add(pair)
    for A, B in sep_pairs:
        add_min_sep(s, x[A] - x[B], y[A] - y[B], MIN_SEP)

    # --------------------------------------------------------
    # Añadir restricciones incrementalmente (push/pop)
    # --------------------------------------------------------
    for item in items:
        A, B = item["origen"], item["destino"]
        c = item["constraint"]
        if A not in x or B not in x:
            continue
        dx = x[A] - x[B]
        dy = y[A] - y[B]

        s.push()
        apply_constraint(s, c, dx, dy)
        if s.check() == sat:
            s.pop()
            apply_constraint(s, c, dx, dy)  # consolidar
        else:
            s.pop()  # descartar restricción conflictiva

    # --------------------------------------------------------
    # Modelo final
    # --------------------------------------------------------
    if s.check() != sat:
        print("⚠️ Z3 no pudo satisfacer las restricciones. Layout parcial.")
        try:
            m = s.model()
        except Exception:
            coords = {p: {"x": random.randint(0, WIDTH), "y": random.randint(0, HEIGHT)} for p in lugares}
            rel_eval = [{"origen": it["origen"], "tipo": it["tipo"], "destino": it["destino"], "satisface": False}
                        for it in constraints]
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

    # --------------------------------------------------------
    # Evaluar satisfacción por relación
    # --------------------------------------------------------
    def satisfied(item):
        A, B, t = item["origen"], item["destino"], item["tipo"].upper()
        ddx = coords[A]["x"] - coords[B]["x"]
        ddy = coords[A]["y"] - coords[B]["y"]
        c = item["constraint"]
        kind = c["kind"]
        ok = True
        if kind == "directional":
            expr_val = ddx if c["dir_expr"] == "dx" else ddy
            if c["dir_op"] == ">=":
                ok &= (expr_val >= c["dir_value"])
            elif c["dir_op"] == "<=":
                ok &= (expr_val <= c["dir_value"])
            ok &= (abs(ddx) <= c["dx_max"]) and (abs(ddy) <= c["dy_max"])
        elif kind == "abs_box":
            ok &= (abs(ddx) <= c["dx_max"]) and (abs(ddy) <= c["dy_max"])
        return bool(ok)

    rel_eval = [{
        "origen": it["origen"],
        "tipo": it["tipo"],
        "destino": it["destino"],
        "satisface": satisfied(it)
    } for it in constraints]

    CSR = sum(1 for r in rel_eval if r["satisface"]) / max(1, len(rel_eval))
    result = {"coords": coords, "CSR": CSR, "rel_eval": rel_eval,
              "width": WIDTH, "height": HEIGHT,
              "MARGIN_DIR": MARGIN_DIR, "RADIUS_DIR": RADIUS_DIR,
              "DIST_CLOSE": DIST_CLOSE, "DIST_CONNECT": DIST_CONNECT}
    if METROS_POR_PIXEL is not None:
        result["metros_por_pixel"] = METROS_POR_PIXEL
    return result


# ------------------------------------------------------------
# 4. Ejecutar solver
# ------------------------------------------------------------
solution = solve_with_z3(clean_places, ineq_constraints)

print(f"CSR: {solution['CSR']:.3f}")

solution_path = _JSON_DIR / "solution.json"
with open(solution_path, "w", encoding="utf-8") as f:
    json.dump(solution, f, indent=2, ensure_ascii=False)

print(f"Solución guardada en {solution_path}")


# ------------------------------------------------------------
# 5. Graficar usando coords del solver
# ------------------------------------------------------------
FIGSIZE = (20, 10)
DPI = 240
NODE_SIZE = 700
FONT_NODES = 8
FONT_EDGES = 10

# map (origen,destino) -> ¿alguna relación satisfecha?
pair_sat = {}
for r in solution["rel_eval"]:
    a, b = r["origen"], r["destino"]
    key = tuple(sorted((a, b)))
    pair_sat[key] = pair_sat.get(key, False) or r["satisface"]

G = nx.Graph()
for p in clean_places:
    G.add_node(p)

for item in ineq_constraints:
    a, b = item["origen"], item["destino"]
    key = tuple(sorted((a, b)))
    if not G.has_edge(a, b):
        G.add_edge(a, b, tipo=item["tipo"], satisface=pair_sat.get(key, False))

plt.figure(figsize=FIGSIZE, dpi=DPI)

pos = {
    p: (solution["coords"][p]["x"], solution["coords"][p]["y"])
    for p in clean_places
}

# Nodos conectados vs aislados
deg_dict = dict(G.degree())
nodes_conectados = [p for p in clean_places if deg_dict.get(p, 0) > 0]
nodes_aislados   = [p for p in clean_places if deg_dict.get(p, 0) == 0]

# --- conectados ---
node_edges_con = []
node_colors_con = []
for p in nodes_conectados:
    if is_pivot(p):
        node_edges_con.append("#1f3d1f")
        node_colors_con.append("#b8e6b8")
    else:
        node_edges_con.append("#2d7a41")
        node_colors_con.append("#b8e6b8")

nx.draw_networkx_nodes(
    G, pos, nodelist=nodes_conectados,
    node_color=node_colors_con, node_size=NODE_SIZE,
    edgecolors=node_edges_con, linewidths=1.2
)

# --- aislados ---
if nodes_aislados:
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodes_aislados,
        node_color=["#cfe9cf"] * len(nodes_aislados),
        node_size=int(NODE_SIZE * 0.5),
        edgecolors=["#2d7a41"] * len(nodes_aislados), linewidths=0.8
    )

# Aristas: verde=satisfechas, rojo=violadas
edge_colors = ["green" if G[u][v]["satisface"] else "red" for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.8, alpha=0.9)

# Etiquetas nodos conectados
nx.draw_networkx_labels(
    G, pos, labels={p: p for p in nodes_conectados}, font_size=FONT_NODES,
    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0)
)

# Etiquetas de aristas violadas
violadas_labels = {
    (u, v): G[u][v]["tipo"] for u, v in G.edges()
    if not G[u][v]["satisface"]
}
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=violadas_labels, font_size=FONT_EDGES,
    bbox=dict(alpha=0.35, facecolor="white", edgecolor="none")
)

scale_info = f" · {solution['metros_por_pixel']:.0f} m/px" if "metros_por_pixel" in solution else ""
plt.title(f"Mapa generado · CSR={solution['CSR']:.3f}{scale_info}\nVerde=satisfechas · Rojo=violadas")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()

_IMG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(_IMG_DIR / "mapa.svg", format="svg", bbox_inches="tight")

plt.show()
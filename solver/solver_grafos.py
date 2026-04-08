# solver_grafos.py
import json
import random
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from z3 import *

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
MIN_SEP = params["MIN_SEP"]

SOLVE_TIMEOUT_MS = 2500


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
# 3. Solver usando JSON de inecuaciones
# ------------------------------------------------------------
def solve_with_z3(lugares, constraints):
    s = Solver()
    s.set(timeout=SOLVE_TIMEOUT_MS)

    x, y = {}, {}

    # Variables de posición
    for i, p in enumerate(lugares):
        x[p] = Int(f"x_{i}")
        y[p] = Int(f"y_{i}")

        s.add(
            And(
                x[p] >= 0,
                x[p] <= WIDTH,
                y[p] >= 0,
                y[p] <= HEIGHT
            )
        )

    # --------------------------------------------------------
    # Ancla inicial
    # --------------------------------------------------------
    if lugares:
        first = lugares[0]
        s.add(x[first] == WIDTH // 2)
        s.add(y[first] == HEIGHT // 2)

    # --------------------------------------------------------
    # Separación mínima global
    # --------------------------------------------------------
    for i in range(len(lugares)):
        for j in range(i + 1, len(lugares)):
            A = lugares[i]
            B = lugares[j]
            add_min_sep(s, x[A] - x[B], y[A] - y[B], MIN_SEP)

    # --------------------------------------------------------
    # Aplicar restricciones desde inequalities.json
    # --------------------------------------------------------
    for item in constraints:
        A = item["origen"]
        B = item["destino"]
        c = item["constraint"]

        if A not in x or B not in x:
            continue

        dx = x[A] - x[B]
        dy = y[A] - y[B]

        kind = c["kind"]

        if kind == "ineq":
            expr = dx if c["expr"] == "dx" else dy
            op = c["op"]
            value = c["value"]

            if op == ">=":
                s.add(expr >= value)
            elif op == "<=":
                s.add(expr <= value)

        elif kind == "abs_box":
            add_abs_le(s, dx, c["dx_max"])
            add_abs_le(s, dy, c["dy_max"])

    # --------------------------------------------------------
    # Resolver
    # --------------------------------------------------------
    if s.check() != sat:
        print("⚠️ Layout no satisfacible. Generando layout aleatorio.")
        return {
            "coords": {
                p: {
                    "x": random.randint(0, WIDTH),
                    "y": random.randint(0, HEIGHT)
                }
                for p in lugares
            },
            "CSR": 0.0,
            "width": WIDTH,
            "height": HEIGHT
        }

    model = s.model()

    coords = {
        p: {
            "x": int(model.eval(x[p]).as_long()),
            "y": int(model.eval(y[p]).as_long())
        }
        for p in lugares
    }

    return {
        "coords": coords,
        "CSR": 1.0,
        "width": WIDTH,
        "height": HEIGHT
    }


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
# 5. Graficar
# ------------------------------------------------------------
G = nx.Graph()

for p in clean_places:
    G.add_node(p)

for item in ineq_constraints:
    G.add_edge(item["origen"], item["destino"])

plt.figure(figsize=(20, 10), dpi=240)

pos = {
    p: (
        solution["coords"][p]["x"],
        solution["coords"][p]["y"]
    )
    for p in clean_places
}

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=700
)

plt.title(f"Mapa generado desde inequalities.json · CSR={solution['CSR']:.3f}")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()

_IMG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(_IMG_DIR / "mapa.svg", format="svg", bbox_inches="tight")

plt.show()
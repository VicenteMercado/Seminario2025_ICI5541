# generar_inecuaciones.py
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_JSON_DIR = _ROOT / "json"


# ------------------------------------------------------------
# 1. Cargar relaciones originales
# ------------------------------------------------------------
input_path = _JSON_DIR / "map_relations.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

lugares = data.get("lugares", [])
relaciones = data.get("relaciones", [])
meta = data.get("lugares_meta", {})
pivotes = data.get("pivotes", [])

print(f"Cargados {len(lugares)} lugares y {len(relaciones)} relaciones.")


# ------------------------------------------------------------
# 2. Parámetros geométricos
# ------------------------------------------------------------
N = max(1, len(lugares))
SIDE = max(600, 20 * N)

WIDTH = SIDE
HEIGHT = SIDE

MARGIN_DIR   = max(8, SIDE // 28)    # N/S/E/O margen mínimo direccional
RADIUS_DIR   = max(20, SIDE // 3)    # N/S/E/O radio máximo de cercanía
DIST_CLOSE   = max(14, SIDE // 7)    # CERCA_DE
DIST_CONNECT = max(16, SIDE // 5)    # CONECTA
MIN_SEP      = max(8, SIDE // 22)    # separación mínima nodos

# ------------------------------------------------------------
# 2b. Escala metros → píxeles (si hay distancias concretas)
# ------------------------------------------------------------
all_dist_m = [r["distancia_m"] for r in relaciones if "distancia_m" in r]
if all_dist_m:
    max_dist_m = max(all_dist_m)
    METROS_POR_PIXEL = max_dist_m / (SIDE * 0.4)
    print(f"Escala: {METROS_POR_PIXEL:.1f} m/px (dist. máx: {max_dist_m:.0f}m, lienzo: {SIDE}px)")
else:
    METROS_POR_PIXEL = None

def dist_m_to_px(metros):
    """Convierte metros a píxeles del lienzo. Retorna None si no hay escala."""
    if METROS_POR_PIXEL is None or metros is None:
        return None
    return max(MIN_SEP + 1, int(metros / METROS_POR_PIXEL))


# ------------------------------------------------------------
# 3. Traducción relación -> inecuación serializable
# ------------------------------------------------------------
def relation_to_constraint(rel):
    A = rel["origen"]
    B = rel["destino"]
    tipo = rel["tipo"].upper()
    dpx = dist_m_to_px(rel.get("distancia_m"))

    margin = dpx if dpx else MARGIN_DIR
    radius = dpx if dpx else RADIUS_DIR

    # Relaciones direccionales (margen mínimo + radio máximo)
    if tipo == "NORTE_DE":
        return {
            "kind": "directional",
            "dir_expr": "dy",
            "dir_op": ">=",
            "dir_value": margin,
            "dx_max": radius,
            "dy_max": radius
        }

    elif tipo == "SUR_DE":
        return {
            "kind": "directional",
            "dir_expr": "dy",
            "dir_op": "<=",
            "dir_value": -margin,
            "dx_max": radius,
            "dy_max": radius
        }

    elif tipo == "ESTE_DE":
        return {
            "kind": "directional",
            "dir_expr": "dx",
            "dir_op": ">=",
            "dir_value": margin,
            "dx_max": radius,
            "dy_max": radius
        }

    elif tipo == "OESTE_DE":
        return {
            "kind": "directional",
            "dir_expr": "dx",
            "dir_op": "<=",
            "dir_value": -margin,
            "dx_max": radius,
            "dy_max": radius
        }

    # Cercanía
    elif tipo == "CERCA_DE":
        bound = int((dpx if dpx else DIST_CLOSE) * 1.2)
        return {
            "kind": "abs_box",
            "dx_max": bound,
            "dy_max": bound
        }

    # Conexión
    elif tipo == "CONECTA":
        bound = int((dpx if dpx else DIST_CONNECT) * 1.2)
        return {
            "kind": "abs_box",
            "dx_max": bound,
            "dy_max": bound
        }

    return None


# ------------------------------------------------------------
# 4. Construcción del JSON de inecuaciones
# ------------------------------------------------------------
ineq_data = {
    "lugares": lugares,
    "pivotes": pivotes,
    "params": {
        "WIDTH": WIDTH,
        "HEIGHT": HEIGHT,
        "MARGIN_DIR": MARGIN_DIR,
        "RADIUS_DIR": RADIUS_DIR,
        "DIST_CLOSE": DIST_CLOSE,
        "DIST_CONNECT": DIST_CONNECT,
        "MIN_SEP": MIN_SEP,
        **({"metros_por_pixel": round(METROS_POR_PIXEL, 2)} if METROS_POR_PIXEL else {})
    },
    "constraints": []
}

for rel in relaciones:
    c = relation_to_constraint(rel)

    if c is None:
        continue

    entry = {
        "origen": rel["origen"],
        "destino": rel["destino"],
        "tipo": rel["tipo"],
        "constraint": c
    }
    if "distancia_m" in rel:
        entry["distancia_m"] = rel["distancia_m"]
    ineq_data["constraints"].append(entry)


# ------------------------------------------------------------
# 5. Guardar inequalities.json
# ------------------------------------------------------------
output_path = _JSON_DIR / "inequalities.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(ineq_data, f, indent=2, ensure_ascii=False)

print(f"Inecuaciones guardadas en: {output_path}")
print(f"Total constraints: {len(ineq_data['constraints'])}")
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

MARGIN_DIR = max(8, SIDE // 28)
DIST_CLOSE = max(14, SIDE // 9)
DIST_CONNECT = max(16, SIDE // 8)
MIN_SEP = max(8, SIDE // 22)


# ------------------------------------------------------------
# 3. Traducción relación -> inecuación serializable
# ------------------------------------------------------------
def relation_to_constraint(rel):
    A = rel["origen"]
    B = rel["destino"]
    tipo = rel["tipo"].upper()

    # Relaciones direccionales
    if tipo == "NORTE_DE":
        return {
            "kind": "ineq",
            "expr": "dy",
            "op": ">=",
            "value": MARGIN_DIR
        }

    elif tipo == "SUR_DE":
        return {
            "kind": "ineq",
            "expr": "dy",
            "op": "<=",
            "value": -MARGIN_DIR
        }

    elif tipo == "ESTE_DE":
        return {
            "kind": "ineq",
            "expr": "dx",
            "op": ">=",
            "value": MARGIN_DIR
        }

    elif tipo == "OESTE_DE":
        return {
            "kind": "ineq",
            "expr": "dx",
            "op": "<=",
            "value": -MARGIN_DIR
        }

    # Cercanía
    elif tipo == "CERCA_DE":
        return {
            "kind": "abs_box",
            "dx_max": DIST_CLOSE,
            "dy_max": DIST_CLOSE
        }

    # Conexión
    elif tipo == "CONECTA":
        return {
            "kind": "abs_box",
            "dx_max": DIST_CONNECT,
            "dy_max": DIST_CONNECT
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
        "DIST_CLOSE": DIST_CLOSE,
        "DIST_CONNECT": DIST_CONNECT,
        "MIN_SEP": MIN_SEP
    },
    "constraints": []
}

for rel in relaciones:
    c = relation_to_constraint(rel)

    if c is None:
        continue

    ineq_data["constraints"].append({
        "origen": rel["origen"],
        "destino": rel["destino"],
        "tipo": rel["tipo"],
        "constraint": c
    })


# ------------------------------------------------------------
# 5. Guardar inequalities.json
# ------------------------------------------------------------
output_path = _JSON_DIR / "inequalities.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(ineq_data, f, indent=2, ensure_ascii=False)

print(f"Inecuaciones guardadas en: {output_path}")
print(f"Total constraints: {len(ineq_data['constraints'])}")
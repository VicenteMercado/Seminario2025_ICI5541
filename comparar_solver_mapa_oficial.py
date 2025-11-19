import json
import math
import itertools
from typing import Dict, Tuple, List, Set


# -------------------------------------------------------------------
# 1. Lectura de layouts
# -------------------------------------------------------------------

def cargar_layout(path: str) -> Tuple[Dict[str, Tuple[float, float]], float, float]:
    """
    Carga un layout desde JSON.

    Formatos aceptados:
    - layout_solution.json de Z3:
        { "coords": { nombre: {"x":..., "y":..., "r":...}, ... },
          "width": ..., "height": ... }

    - layout oficial anotado a mano:
        { "coords": { nombre: {"x":..., "y":...}, ... },
          "width": ..., "height": ... }

    Devuelve:
        coords: dict nombre -> (x, y)
        width, height: tamaño del lienzo
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    coords_raw = data["coords"]
    coords = {name: (float(v["x"]), float(v["y"])) for name, v in coords_raw.items()}
    width = float(data.get("width", 1.0))
    height = float(data.get("height", 1.0))
    return coords, width, height


# -------------------------------------------------------------------
# 2. Extracción de relaciones geométricas desde coordenadas
# -------------------------------------------------------------------

def extraer_relaciones_geom(
    coords: Dict[str, Tuple[float, float]],
    width: float,
    height: float,
    cerca_ratio: float = 0.18,
    margen_dir_ratio: float = 0.05,
) -> List[dict]:
    """
    A partir de coordenadas (x,y) de cada lugar, genera relaciones
    cualitativas del tipo:

        - NORTE_DE / SUR_DE
        - ESTE_DE / OESTE_DE
        - CERCA_DE   (simétrica)

    La idea es aproximar lo que el extractor textual hace,
    pero directamente sobre un mapa (solver u oficial).

    Parámetros:
        cerca_ratio      -> umbral de "cerca" en múltiplos del tamaño del mapa
        margen_dir_ratio -> umbral para considerar que hay dirección clara N/S/E/O

    Devuelve una lista de dicts:
        {"origen": A, "tipo": "NORTE_DE", "destino": B}
        ...
    """
    nombres = list(coords.keys())
    xs = [coords[n][0] for n in nombres]
    ys = [coords[n][1] for n in nombres]

    # Aproximamos el tamaño típico con la diagonal del bounding box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    diag = math.hypot(max_x - min_x, max_y - min_y)

    # Si el layout trae width/height, usamos el máximo para escalar
    side = max(width, height, diag)
    thr_cerca = cerca_ratio * side
    thr_dir = margen_dir_ratio * side

    relaciones: List[dict] = []

    for i, j in itertools.combinations(range(len(nombres)), 2):
        A, B = nombres[i], nombres[j]
        xA, yA = coords[A]
        xB, yB = coords[B]

        # Definimos deltas como en solver_grafos: dx = xA - xB, dy = yA - yB
        dx = xA - xB
        dy = yA - yB

        # ----------------------
        # Direcciones cardinales
        # ----------------------
        # NORTE / SUR
        if dy >= thr_dir:
            relaciones.append({"origen": A, "tipo": "NORTE_DE", "destino": B})
            relaciones.append({"origen": B, "tipo": "SUR_DE", "destino": A})
        elif dy <= -thr_dir:
            relaciones.append({"origen": A, "tipo": "SUR_DE", "destino": B})
            relaciones.append({"origen": B, "tipo": "NORTE_DE", "destino": A})

        # ESTE / OESTE
        if dx >= thr_dir:
            relaciones.append({"origen": A, "tipo": "ESTE_DE", "destino": B})
            relaciones.append({"origen": B, "tipo": "OESTE_DE", "destino": A})
        elif dx <= -thr_dir:
            relaciones.append({"origen": A, "tipo": "OESTE_DE", "destino": B})
            relaciones.append({"origen": B, "tipo": "ESTE_DE", "destino": A})

        # ----------
        # CERCA_DE
        # ----------
        dist = math.hypot(xA - xB, yA - yB)
        if dist <= thr_cerca:
            relaciones.append({"origen": A, "tipo": "CERCA_DE", "destino": B})
            relaciones.append({"origen": B, "tipo": "CERCA_DE", "destino": A})

    return relaciones


# -------------------------------------------------------------------
# 3. Comparación entre mapa del solver y mapa oficial
# -------------------------------------------------------------------

def comparar_layouts_por_relaciones(
    path_solver_json: str,
    path_oficial_json: str,
    cerca_ratio: float = 0.18,
    margen_dir_ratio: float = 0.05,
) -> dict:
    """
    Compara el layout del solver con el layout oficial a nivel de
    relaciones espaciales cualitativas.

    Pasos:
        1) Cargar coords solver y coords oficial.
        2) Quedarse sólo con los lugares comunes.
        3) Extraer relaciones geométricas en cada mapa.
        4) Calcular precisión / recall / F1 por tipo y global.

    Devuelve un dict con:
        - lugares_comunes
        - num_relaciones_solver / oficial
        - precision_global, recall_global, f1_global
        - metrics_por_tipo (CERCA_DE, NORTE_DE, ...)
        - ejemplos_fp / fn globales y por tipo
    """
    # 1. Cargar layouts
    coords_solver, w_s, h_s = cargar_layout(path_solver_json)
    coords_oficial, w_o, h_o = cargar_layout(path_oficial_json)

    # 2. Intersección de lugares
    comunes = sorted(set(coords_solver.keys()) & set(coords_oficial.keys()))
    if len(comunes) < 2:
        raise ValueError(
            f"No hay suficientes lugares en común para comparar (encontrados: {len(comunes)})"
        )

    coords_solver_c = {name: coords_solver[name] for name in comunes}
    coords_oficial_c = {name: coords_oficial[name] for name in comunes}

    # 3. Extraer relaciones en cada mapa
    rel_solver = extraer_relaciones_geom(
        coords_solver_c, w_s, h_s,
        cerca_ratio=cerca_ratio,
        margen_dir_ratio=margen_dir_ratio,
    )
    rel_oficial = extraer_relaciones_geom(
        coords_oficial_c, w_o, h_o,
        cerca_ratio=cerca_ratio,
        margen_dir_ratio=margen_dir_ratio,
    )

    # Pasar a tuplas para poder hacer intersecciones de conjuntos
    def to_tuple(r):
        return (r["origen"], r["tipo"], r["destino"])

    set_solver: Set[Tuple[str, str, str]] = {to_tuple(r) for r in rel_solver}
    set_oficial: Set[Tuple[str, str, str]] = {to_tuple(r) for r in rel_oficial}

    inter = set_solver & set_oficial
    solo_solver = set_solver - set_oficial
    solo_oficial = set_oficial - set_solver

    tp = len(inter)
    fp = len(solo_solver)
    fn = len(solo_oficial)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # Métricas por tipo
    tipos = sorted({t for _, t, _ in (set_solver | set_oficial)})
    metrics_por_tipo = {}

    for t in tipos:
        s_t = {rel for rel in set_solver if rel[1] == t}
        o_t = {rel for rel in set_oficial if rel[1] == t}
        inter_t = s_t & o_t
        solo_s_t = s_t - o_t
        solo_o_t = o_t - s_t

        tp_t = len(inter_t)
        fp_t = len(solo_s_t)
        fn_t = len(solo_o_t)

        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        rec_t  = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        if prec_t + rec_t > 0:
            f1_t = 2 * prec_t * rec_t / (prec_t + rec_t)
        else:
            f1_t = 0.0

        metrics_por_tipo[t] = {
            "tp": tp_t,
            "fp": fp_t,
            "fn": fn_t,
            "precision": prec_t,
            "recall": rec_t,
            "f1": f1_t,
            "ejemplos_fp": sorted(list(solo_s_t))[:10],
            "ejemplos_fn": sorted(list(solo_o_t))[:10],
        }

    resultado = {
        "lugares_comunes": comunes,
        "num_relaciones_solver": len(set_solver),
        "num_relaciones_oficial": len(set_oficial),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_global": precision,
        "recall_global": recall,
        "f1_global": f1,
        "metrics_por_tipo": metrics_por_tipo,
        "ejemplos_fp_global": sorted(list(solo_solver))[:20],
        "ejemplos_fn_global": sorted(list(solo_oficial))[:20],
    }
    return resultado


# -------------------------------------------------------------------
# 4. Ejemplo mínimo de uso (puedes adaptarlo o borrarlo)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Aquí se asume que YA tienes un JSON con las coordenadas
    # del mapa oficial (por ejemplo "layout_oficial.json"),
    # con el mismo formato general que layout_solution.json.
    #
    # Este bloque es sólo un ejemplo; puedes llamarlo también desde
    # un notebook importando la función.

    res = comparar_layouts_por_relaciones(
        path_solver_json="layout_solution.json",
        path_oficial_json="layout_oficial.json",  # <-- este lo haces tú
    )

    print("Lugares comunes:", len(res["lugares_comunes"]))
    print("Relaciones (solver):", res["num_relaciones_solver"])
    print("Relaciones (oficial):", res["num_relaciones_oficial"])
    print(f"Precisión global: {res['precision_global']:.3f}")
    print(f"Recall global:    {res['recall_global']:.3f}")
    print(f"F1 global:        {res['f1_global']:.3f}")

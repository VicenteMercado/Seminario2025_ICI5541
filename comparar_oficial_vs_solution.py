#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import argparse
import unicodedata
from typing import Dict, Tuple, Any, List, Optional, Set


# -----------------------------------
# Utilidades generales
# -----------------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


STOPWORDS = {
    "de", "del", "la", "el", "las", "los",
    "sede", "casa", "plaza", "calle",
    "barrios", "suburbios",
    "ministerio", "laberintos", "canal", "jardines",
    "torre", "torreon", "torreón", "fortaleza",
    "canton", "cantón", "cantones",
}


def normalize_tokens(name: str) -> Set[str]:
    s = strip_accents(name.lower())
    for ch in ",.;:¡!¿?()[]{}-_/'\"":
        s = s.replace(ch, " ")
    tokens = [t for t in s.split() if t and t not in STOPWORDS]
    if not tokens:
        # fallback: usar la cadena completa sin acentos/espacios extremos
        return {s.strip()}
    return set(tokens)


# -----------------------------------
# Carga del grafo oficial
# -----------------------------------

def load_official_graph(path: str) -> Dict[str, Any]:
    data = load_json(path)

    width = data.get("width")
    height = data.get("height")

    # 1) lugares / places / nodes
    lugares_raw = None
    for key in ["lugares", "places", "nodes"]:
        if key in data:
            lugares_raw = data[key]
            break

    if lugares_raw is None:
        raise ValueError("No se encontraron claves 'lugares/places/nodes' en el grafo oficial.")

    official_places: List[str] = []

    if isinstance(lugares_raw, dict):
        # { "Nombre": {...}, ... }
        official_places = list(lugares_raw.keys())
    elif isinstance(lugares_raw, list):
        # lista de strings o dicts
        for item in lugares_raw:
            if isinstance(item, str):
                official_places.append(item)
            elif isinstance(item, dict):
                nombre = item.get("nombre") or item.get("name") or item.get("id")
                if nombre:
                    official_places.append(nombre)
    else:
        raise ValueError("Formato de 'lugares/places/nodes' en oficial no soportado.")

    # 2) relaciones / relations / edges
    relaciones_raw = None
    for key in ["relaciones", "relations", "edges", "aristas"]:
        if key in data:
            relaciones_raw = data[key]
            break

    if relaciones_raw is None:
        raise ValueError("No se encontraron claves 'relaciones/relations/edges' en el grafo oficial.")

    official_relations = []
    for r in relaciones_raw:
        if not isinstance(r, dict):
            continue
        origen = r.get("origen") or r.get("source") or r.get("from")
        destino = r.get("destino") or r.get("target") or r.get("to")
        tipo = r.get("tipo") or r.get("type")
        if origen and destino and tipo:
            official_relations.append({
                "origen": origen,
                "destino": destino,
                "tipo": tipo
            })

    return {
        "places": official_places,
        "relations": official_relations,
        "width": width,
        "height": height
    }


# -----------------------------------
# Carga de coords del solver (solution.json)
# -----------------------------------

def load_solution_coords(path: str) -> Dict[str, Any]:
    data = load_json(path)

    width = data.get("width")
    height = data.get("height")

    coords_raw = None

    # Caso 1: tu formato actual -> "coords": { "Nombre": {x,y}, ... }
    if "coords" in data and isinstance(data["coords"], dict):
        coords_raw = data["coords"]
    else:
        # Casos alternativos si en algún momento usas otro formato:
        for key in ["lugares", "places", "nodes"]:
            if key in data and isinstance(data[key], dict):
                coords_raw = data[key]
                break

    if coords_raw is None:
        raise ValueError("No se encontraron claves 'coords/lugares/places/nodes' en solution.json.")

    coords: Dict[str, Tuple[float, float]] = {}

    for nombre, val in coords_raw.items():
        if isinstance(val, dict):
            x = val.get("x") or val.get("X")
            y = val.get("y") or val.get("Y")
        elif isinstance(val, (list, tuple)) and len(val) == 2:
            x, y = val
        else:
            continue

        if x is None or y is None:
            continue

        coords[nombre] = (float(x), float(y))

    return {
        "coords": coords,
        "width": width,
        "height": height
    }


# -----------------------------------
# Mapeo de nombres oficial -> solver
# -----------------------------------

def build_name_mapping(official_places: List[str],
                       solver_places: List[str],
                       min_jaccard: float = 0.4) -> Dict[str, Optional[str]]:
    solver_index = [(name, normalize_tokens(name)) for name in solver_places]
    mapping: Dict[str, Optional[str]] = {}

    for off_name in official_places:
        off_tokens = normalize_tokens(off_name)
        if not off_tokens:
            mapping[off_name] = None
            continue

        best_label = None
        best_score = 0.0

        for sol_name, sol_tokens in solver_index:
            if not sol_tokens:
                continue
            inter = len(off_tokens & sol_tokens)
            union = len(off_tokens | sol_tokens)
            jaccard = inter / union if union > 0 else 0.0
            if jaccard > best_score:
                best_score = jaccard
                best_label = sol_name

        if best_label is not None and best_score >= min_jaccard:
            mapping[off_name] = best_label
        else:
            mapping[off_name] = None

    return mapping


# -----------------------------------
# Chequeo de relaciones geométricas
# -----------------------------------

def check_relation(tipo: str,
                   p_origen: Tuple[float, float],
                   p_destino: Tuple[float, float],
                   margin_dir: float,
                   dist_close: float,
                   dist_connect: float) -> bool:
    x_o, y_o = p_origen
    x_d, y_d = p_destino

    dx = x_o - x_d
    dy = y_o - y_d
    dist = math.hypot(dx, dy)

    tipo = tipo.upper()

    if tipo == "CERCA_DE":
        return dist <= dist_close
    if tipo == "CONECTA":
        return dist <= dist_connect

    # IMPORTANTE:
    # Mantengo la convención que ya usaste (que produjo el CSR≈0.206),
    # donde el eje Y está "invertido" respecto a norte/sur de la narrativa:
    # - NORTE_DE(A,B): y_A > y_B + margin_dir
    # - SUR_DE(A,B)  : y_A < y_B - margin_dir
    # - ESTE_DE(A,B) : x_A > x_B + margin_dir
    # - OESTE_DE(A,B): x_A < x_B - margin_dir

    if tipo == "NORTE_DE":
        return y_o > (y_d + margin_dir)
    if tipo == "SUR_DE":
        return y_o < (y_d - margin_dir)
    if tipo == "ESTE_DE":
        return x_o > (x_d + margin_dir)
    if tipo == "OESTE_DE":
        return x_o < (x_d - margin_dir)

    # Si aparece un tipo desconocido, por defecto lo consideramos no satisfecho
    return False


# -----------------------------------
# Comparación
# -----------------------------------

def compare_graphs(official_path: str,
                   solution_path: str,
                   margin_dir: float = 21.0,
                   dist_close: float = 66.0,
                   dist_connect: float = 75.0,
                   output_json_path: str = "comparison_official_vs_solution.json") -> None:
    # Cargar datos
    official = load_official_graph(official_path)
    solution = load_solution_coords(solution_path)

    official_places = official["places"]
    official_relations = official["relations"]

    coords = solution["coords"]
    solver_places = list(coords.keys())

    width = official.get("width") or solution.get("width")
    height = official.get("height") or solution.get("height")

    # Mapeo nombres
    mapping = build_name_mapping(official_places, solver_places)

    # Resumen mapeo
    lugares_mapeados = {k: v for k, v in mapping.items() if v is not None}
    lugares_sin_mapa = [k for k, v in mapping.items() if v is None]

    print(f"Grafo oficial: {len(official_places)} lugares, {len(official_relations)} relaciones.")
    print(f"Solución cargada: {len(coords)} lugares con coordenadas (width={width}, height={height})")
    print(f"Umbrales usados -> MARGIN_DIR={margin_dir}, DIST_CLOSE={dist_close}, DIST_CONNECT={dist_connect}")
    print(f"Lugares oficiales mapeados a nodos del solver: {len(lugares_mapeados)}/{len(official_places)}")

    if lugares_sin_mapa:
        print("Lugares oficiales SIN mapeo en coords del solver:")
        for name in lugares_sin_mapa:
            print(f"  - {name}")
    print()

    # Estadísticas por tipo
    tipos = ["CERCA_DE", "CONECTA", "ESTE_DE", "NORTE_DE", "OESTE_DE", "SUR_DE"]
    stats_por_tipo = {
        t: {"total": 0, "evaluables": 0, "satisfechas": 0}
        for t in tipos
    }

    detalle_relaciones = []

    total_rel = 0
    evaluables = 0
    satisfechas = 0

    for r in official_relations:
        total_rel += 1
        tipo = r["tipo"].upper()
        if tipo not in stats_por_tipo:
            # Lo podemos contar igual en "total", pero no evaluamos
            stats_por_tipo.setdefault(tipo, {"total": 0, "evaluables": 0, "satisfechas": 0})
        stats_por_tipo[tipo]["total"] += 1

        origen_of = r["origen"]
        destino_of = r["destino"]

        origen_sol = mapping.get(origen_of)
        destino_sol = mapping.get(destino_of)

        evaluado = False
        satisface = None

        if origen_sol is not None and destino_sol is not None:
            p_o = coords.get(origen_sol)
            p_d = coords.get(destino_sol)
            if p_o is not None and p_d is not None:
                evaluado = True
                evaluables += 1
                stats_por_tipo[tipo]["evaluables"] += 1

                es_ok = check_relation(tipo, p_o, p_d, margin_dir, dist_close, dist_connect)
                satisface = bool(es_ok)
                if es_ok:
                    satisfechas += 1
                    stats_por_tipo[tipo]["satisfechas"] += 1

        detalle_relaciones.append({
            "origen_oficial": origen_of,
            "destino_oficial": destino_of,
            "tipo": tipo,
            "origen_solver": origen_sol,
            "destino_solver": destino_sol,
            "evaluado": evaluado,
            "satisface": satisface
        })

    csr = satisfechas / evaluables if evaluables > 0 else 0.0

    # Print resumen global
    print("=== Resultados globales ===")
    print(f"Relaciones oficiales totales           : {total_rel}")
    print(f"Relaciones oficiales evaluables       : {evaluables}")
    print(f"Relaciones oficiales satisfechas      : {satisfechas}")
    print(f"CSR respecto al grafo oficial (solver): {csr:.3f}")
    print()

    # Print por tipo
    print("=== Detalle por tipo de relación ===")
    for t in tipos:
        s = stats_por_tipo[t]
        csr_t = s["satisfechas"] / s["evaluables"] if s["evaluables"] > 0 else 0.0
        print(f"- {t:<7} -> total={s['total']:3d}, eval={s['evaluables']:3d}, ok={s['satisfechas']:3d}, CSR={csr_t:.3f}")
    print()

    # Construir JSON de salida
    resumen_tipo_json = {}
    for t, s in stats_por_tipo.items():
        evaluables_t = s["evaluables"]
        csr_t = s["satisfechas"] / evaluables_t if evaluables_t > 0 else 0.0
        resumen_tipo_json[t] = {
            "total": s["total"],
            "evaluables": evaluables_t,
            "satisfechas": s["satisfechas"],
            "csr": csr_t
        }

    resumen_global = {
        "lugares_oficiales": len(official_places),
        "lugares_solution": len(coords),
        "lugares_oficiales_mapeados": len(lugares_mapeados),
        "lugares_oficiales_sin_mapa": lugares_sin_mapa,
        "relaciones_oficiales_total": total_rel,
        "relaciones_oficiales_evaluables": evaluables,
        "relaciones_oficiales_satisfechas": satisfechas,
        "csr_oficial": csr
    }

    output_data = {
        "width": width,
        "height": height,
        "MARGIN_DIR": margin_dir,
        "DIST_CLOSE": dist_close,
        "DIST_CONNECT": dist_connect,
        "resumen_global": resumen_global,
        "mapeo_lugares": mapping,
        "resumen_tipo": resumen_tipo_json,
        "detalle": detalle_relaciones
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Detalle guardado en '{output_json_path}'")


# -----------------------------------
# CLI
# -----------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comparar grafo oficial (relaciones textuales) con solución de coords (solution.json)."
    )
    parser.add_argument("official", help="Ruta a official_graph.json")
    parser.add_argument("solution", help="Ruta a solution.json (con 'coords')")
    parser.add_argument("--margin-dir", type=float, default=21.0,
                        help="Umbral de píxeles para relaciones direccionales (NORTE/SUR/ESTE/OESTE)")
    parser.add_argument("--dist-close", type=float, default=66.0,
                        help="Umbral de distancia para CERCA_DE")
    parser.add_argument("--dist-connect", type=float, default=75.0,
                        help="Umbral de distancia para CONECTA")
    parser.add_argument("--output", type=str, default="comparison_official_vs_solution.json",
                        help="Ruta del JSON de salida")

    args = parser.parse_args()

    compare_graphs(
        official_path=args.official,
        solution_path=args.solution,
        margin_dir=args.margin_dir,
        dist_close=args.dist_close,
        dist_connect=args.dist_connect,
        output_json_path=args.output
    )


if __name__ == "__main__":
    main()

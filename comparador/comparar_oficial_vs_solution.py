#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import argparse
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional, Set

_JSON_DIR = Path(__file__).resolve().parent.parent / "json"
_DEFAULT_OFFICIAL = str(_JSON_DIR / "official_graph.json")
_DEFAULT_SOLUTION = str(_JSON_DIR / "solution.json")
_DEFAULT_COMPARISON_OUT = str(_JSON_DIR / "comparison_official_vs_solution.json")
_DEFAULT_ALIASES = str(_JSON_DIR / "aliases.json")
_DEFAULT_OFFICIAL_CSV = str(Path(__file__).resolve().parent.parent / "csv" / "official_nodes.csv")


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
        # si todo eran stopwords, usamos la cadena básica como respaldo
        return {s.strip()}
    return set(tokens)


def canonical_text(name: str) -> str:
    s = strip_accents((name or "").lower())
    for ch in ",.;:¡!¿?()[]{}-_/'\"":
        s = s.replace(ch, " ")
    return " ".join(s.split())


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
        # diccionario { "Nombre": {...}, ... }
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

    # Caso principal: "coords": { "Nombre": {x,y}, ... }
    if "coords" in data and isinstance(data["coords"], dict):
        coords_raw = data["coords"]
    else:
        # Alternativa por si en algún momento se guarda con otra clave
        for key in ["lugares", "places", "nodes"]:
            if key in data and isinstance(data[key], dict):
                coords_raw = data[key]
                break

    if coords_raw is None:
        raise ValueError("No se encontraron claves 'coords/lugares/places/nodes' en solution.json.")

    coords: Dict[str, Tuple[float, float]] = {}

    for nombre, val in coords_raw.items():
        if isinstance(val, dict):
            x = val.get("x", None)
            if x is None:
                x = val.get("X", None)
            y = val.get("y", None)
            if y is None:
                y = val.get("Y", None)
        elif isinstance(val, (list, tuple)) and len(val) == 2:
            x, y = val
        else:
            continue

        if x is None or y is None:
            continue

        coords[nombre] = (float(x), float(y))

    thresholds = {
        "MARGIN_DIR": data.get("MARGIN_DIR"),
        "DIST_CLOSE": data.get("DIST_CLOSE"),"DIST_CONNECT": data.get("DIST_CONNECT"),
        }
    return {
        "coords": coords,
        "width": width,
        "height": height,
        "thresholds": thresholds
        }


def load_aliases(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = load_json(str(p))
    aliases_raw = data.get("aliases", data) if isinstance(data, dict) else {}
    aliases: Dict[str, str] = {}
    if isinstance(aliases_raw, dict):
        for k, v in aliases_raw.items():
            if isinstance(k, str) and isinstance(v, str):
                aliases[canonical_text(k)] = canonical_text(v)
    return aliases


def load_official_positions_csv(path: Optional[str]) -> Dict[str, Tuple[float, float]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    import csv
    out: Dict[str, Tuple[float, float]] = {}
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name")
            x = row.get("x")
            y = row.get("y")
            if not name or x is None or y is None:
                continue
            try:
                out[name] = (float(x), float(y))
            except ValueError:
                continue
    return out



# -----------------------------------
# Mapeo de nombres oficial -> solver
# -----------------------------------

def build_name_mapping(official_places: List[str],
                       solver_places: List[str],
                       aliases: Optional[Dict[str, str]] = None,
                       min_jaccard: float = 0.4) -> Dict[str, Optional[str]]:
    aliases = aliases or {}
    mapping: Dict[str, Optional[str]] = {name: None for name in official_places}
    solver_by_canon: Dict[str, str] = {canonical_text(s): s for s in solver_places}

    used_solver: Set[str] = set()

    # 1) Alias exacto (prioridad alta)
    for off_name in official_places:
        off_c = canonical_text(off_name)
        target_c = aliases.get(off_c)
        if target_c and target_c in solver_by_canon:
            sol_name = solver_by_canon[target_c]
            mapping[off_name] = sol_name
            used_solver.add(sol_name)

    # 2) Matching 1-a-1 por similitud global (greedy)
    candidates: List[Tuple[float, str, str]] = []
    for off_name in official_places:
        if mapping[off_name] is not None:
            continue
        off_tokens = normalize_tokens(off_name)
        if not off_tokens:
            continue
        for sol_name in solver_places:
            if sol_name in used_solver:
                continue
            sol_tokens = normalize_tokens(sol_name)
            if not sol_tokens:
                continue
            inter = len(off_tokens & sol_tokens)
            union = len(off_tokens | sol_tokens)
            jaccard = inter / union if union > 0 else 0.0
            if jaccard >= min_jaccard:
                candidates.append((jaccard, off_name, sol_name))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_official: Set[str] = set()
    for _score, off_name, sol_name in candidates:
        if off_name in used_official or sol_name in used_solver:
            continue
        mapping[off_name] = sol_name
        used_official.add(off_name)
        used_solver.add(sol_name)

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
    #dist = math.hypot(dx, dy)

    tipo = tipo.upper()

    if tipo == "CERCA_DE":
        return abs(dx) <= dist_close and abs(dy) <= dist_close
    if tipo == "CONECTA":
        return abs(dx) <= dist_connect and abs(dy) <= dist_connect

    # Convención de signos:
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
                   aliases_path: Optional[str] = _DEFAULT_ALIASES,
                   official_csv_path: Optional[str] = _DEFAULT_OFFICIAL_CSV,
                   margin_dir: float = 21.0,
                   dist_close: float = 66.0,
                   dist_connect: float = 75.0,
                   output_json_path: str = _DEFAULT_COMPARISON_OUT) -> None:
    # Cargar datos de ambos grafos
    official = load_official_graph(official_path)
    solution = load_solution_coords(solution_path)
    
    saved_th = solution.get("thresholds", {})
    if saved_th.get("MARGIN_DIR") is not None:
        margin_dir = saved_th["MARGIN_DIR"]
    if saved_th.get("DIST_CLOSE") is not None:
        dist_close = saved_th["DIST_CLOSE"]
    if saved_th.get("DIST_CONNECT") is not None:
        dist_connect = saved_th["DIST_CONNECT"]


    official_places = official["places"]
    official_relations = official["relations"]

    coords = solution["coords"]
    solver_places = list(coords.keys())

    width = official.get("width") or solution.get("width")
    height = official.get("height") or solution.get("height")

    # Mapeo de nombres (aliases + similitud, forzando 1-a-1)
    aliases = load_aliases(aliases_path)
    mapping = build_name_mapping(official_places, solver_places, aliases=aliases)

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
            # si aparece un tipo nuevo lo agregamos al dict para no perderlo
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

    # Resumen global
    print("=== Resultados globales ===")
    print(f"Relaciones oficiales totales           : {total_rel}")
    print(f"Relaciones oficiales evaluables       : {evaluables}")
    print(f"Relaciones oficiales satisfechas      : {satisfechas}")
    print(f"CSR respecto al grafo oficial (solver): {csr:.3f}")
    print()

    cobertura_lugares = len(lugares_mapeados) / len(official_places) if official_places else 0.0
    csr_total_rel = satisfechas / total_rel if total_rel > 0 else 0.0
    score_combinado = csr * cobertura_lugares

    print(f"Cobertura de mapeo de lugares         : {cobertura_lugares:.3f}")
    print(f"CSR penalizado por no-evaluables      : {csr_total_rel:.3f}")
    print(f"Score combinado (CSR*cobertura)       : {score_combinado:.3f}")
    print()

    # Detalle por tipo
    print("=== Detalle por tipo de relación ===")
    for t in tipos:
        s = stats_por_tipo[t]
        csr_t = s["satisfechas"] / s["evaluables"] if s["evaluables"] > 0 else 0.0
        print(f"- {t:<7} -> total={s['total']:3d}, eval={s['evaluables']:3d}, ok={s['satisfechas']:3d}, CSR={csr_t:.3f}")
    print()

    # Comparación simétrica por tipo (official vs solver)
    mapped_official = {off: sol for off, sol in mapping.items() if sol is not None}
    inverse_mapping = {sol: off for off, sol in mapped_official.items()}

    official_sets_by_type: Dict[str, Set[Tuple[str, str]]] = {}
    for r in official_relations:
        t = r["tipo"].upper()
        o = r["origen"]
        d = r["destino"]
        if o in mapped_official and d in mapped_official:
            official_sets_by_type.setdefault(t, set()).add((o, d))

    full_solution_data = load_json(solution_path)
    rel_eval = full_solution_data.get("rel_eval", [])
    solver_sets_by_type: Dict[str, Set[Tuple[str, str]]] = {}
    for r in rel_eval:
        t = str(r.get("tipo", "")).upper()
        o_sol = r.get("origen")
        d_sol = r.get("destino")
        if not o_sol or not d_sol:
            continue
        o_off = inverse_mapping.get(o_sol)
        d_off = inverse_mapping.get(d_sol)
        if not o_off or not d_off:
            continue
        solver_sets_by_type.setdefault(t, set()).add((o_off, d_off))

    comparacion_simetrica_por_tipo: Dict[str, Dict[str, float]] = {}
    tipos_union = sorted(set(official_sets_by_type.keys()) | set(solver_sets_by_type.keys()))
    all_tp = all_fp = all_fn = 0
    for t in tipos_union:
        set_off = official_sets_by_type.get(t, set())
        set_sol = solver_sets_by_type.get(t, set())
        tp = len(set_off & set_sol)
        fp = len(set_sol - set_off)
        fn = len(set_off - set_sol)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        comparacion_simetrica_por_tipo[t] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        all_tp += tp
        all_fp += fp
        all_fn += fn

    precision_micro = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall_micro = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro)) if (precision_micro + recall_micro) > 0 else 0.0

    # Distancia official-vs-solver por nodo mapeado (coordenadas normalizadas por bbox)
    distancia_nodos_mapeados = {"pares": 0, "mae": 0.0, "rmse": 0.0}
    official_pos = load_official_positions_csv(official_csv_path)
    if official_pos and mapped_official:
        pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for off_name, sol_name in mapped_official.items():
            if off_name in official_pos and sol_name in coords:
                pairs.append((official_pos[off_name], coords[sol_name]))

        if pairs:
            off_xs = [p[0][0] for p in pairs]
            off_ys = [p[0][1] for p in pairs]
            sol_xs = [p[1][0] for p in pairs]
            sol_ys = [p[1][1] for p in pairs]
            off_dx = max(max(off_xs) - min(off_xs), 1e-9)
            off_dy = max(max(off_ys) - min(off_ys), 1e-9)
            sol_dx = max(max(sol_xs) - min(sol_xs), 1e-9)
            sol_dy = max(max(sol_ys) - min(sol_ys), 1e-9)
            off_min_x, off_min_y = min(off_xs), min(off_ys)
            sol_min_x, sol_min_y = min(sol_xs), min(sol_ys)

            dists = []
            for (ox, oy), (sx, sy) in pairs:
                oxn = (ox - off_min_x) / off_dx
                oyn = (oy - off_min_y) / off_dy
                sxn = (sx - sol_min_x) / sol_dx
                syn = (sy - sol_min_y) / sol_dy
                dists.append(math.hypot(oxn - sxn, oyn - syn))

            if dists:
                mae = sum(dists) / len(dists)
                rmse = math.sqrt(sum(d * d for d in dists) / len(dists))
                distancia_nodos_mapeados = {"pares": len(dists), "mae": mae, "rmse": rmse}

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
        "cobertura_lugares": cobertura_lugares,
        "lugares_oficiales_sin_mapa": lugares_sin_mapa,
        "relaciones_oficiales_total": total_rel,
        "relaciones_oficiales_evaluables": evaluables,
        "relaciones_oficiales_satisfechas": satisfechas,
        "csr_oficial": csr,
        "csr_total_rel": csr_total_rel,
        "score_combinado": score_combinado,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "distancia_nodos_mapeados": distancia_nodos_mapeados,
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
        "comparacion_simetrica_por_tipo": comparacion_simetrica_por_tipo,
        "detalle": detalle_relaciones
    }

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument(
        "official",
        nargs="?",
        default=_DEFAULT_OFFICIAL,
        help=f"Ruta a official_graph.json (por defecto: {_DEFAULT_OFFICIAL})",
    )
    parser.add_argument(
        "solution",
        nargs="?",
        default=_DEFAULT_SOLUTION,
        help=f"Ruta a solution.json con coords (por defecto: {_DEFAULT_SOLUTION})",
    )
    parser.add_argument("--aliases", type=str, default=_DEFAULT_ALIASES,
                        help=f"Ruta a aliases.json (opcional, por defecto: {_DEFAULT_ALIASES})")
    parser.add_argument("--official-csv", type=str, default=_DEFAULT_OFFICIAL_CSV,
                        help=f"Ruta a official_nodes.csv para métricas de distancia (por defecto: {_DEFAULT_OFFICIAL_CSV})")
    parser.add_argument("--margin-dir", type=float, default=21.0,
                        help="Umbral de píxeles para relaciones direccionales (NORTE/SUR/ESTE/OESTE)")
    parser.add_argument("--dist-close", type=float, default=66.0,
                        help="Umbral de distancia para CERCA_DE")
    parser.add_argument("--dist-connect", type=float, default=75.0,
                        help="Umbral de distancia para CONECTA")
    parser.add_argument("--output", type=str, default=_DEFAULT_COMPARISON_OUT,
                        help="Ruta del JSON de salida")

    args = parser.parse_args()

    compare_graphs(
        official_path=args.official,
        solution_path=args.solution,
        aliases_path=args.aliases,
        official_csv_path=args.official_csv,
        margin_dir=args.margin_dir,
        dist_close=args.dist_close,
        dist_connect=args.dist_connect,
        output_json_path=args.output
    )


if __name__ == "__main__":
    main()

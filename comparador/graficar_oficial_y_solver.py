#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Superpone nodos oficiales (CSV, sin aristas entre ellos) y el grafo del solver
(solution.json con aristas). Las posiciones del solver se escalan al recuadro
de los nodos oficiales. Colores distintos por capa.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CSV = _ROOT / "csv" / "official_nodes.csv"
_DEFAULT_SOLUTION = _ROOT / "json" / "solution.json"
_DEFAULT_OUT = _ROOT / "img" / "overlay_oficial_solver.png"

# Colores: oficial (azules) vs solver (naranja / aristas verde-rojo)
COLOR_OFF_NODE = "#1d4ed8"
COLOR_MAP_LINK = "#7c3aed"  # líneas discontinuas oficial->solver (distancia de mapeo)
COLOR_SOL_NODE = "#ea580c"
COLOR_SOL_EDGE_OK = "#16a34a"
COLOR_SOL_EDGE_BAD = "#dc2626"

STOPWORDS = {
    "de",
    "del",
    "la",
    "el",
    "las",
    "los",
    "sede",
    "casa",
    "plaza",
    "calle",
    "barrios",
    "suburbios",
    "ministerio",
    "laberintos",
    "canal",
    "jardines",
    "torre",
    "torreon",
    "torreon",
    "torreon",
    "torreón",
    "fortaleza",
    "canton",
    "cantón",
    "cantones",
    "taller",
    "guarnicion",
    "guarida",
    "guarnición",
}


def _strip_accents(s: str) -> str:
    # Normaliza acentos para hacer matching robusto
    import unicodedata

    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _normalize_tokens(name: str) -> Set[str]:
    s = _strip_accents((name or "").lower())
    for ch in ",.;:¡!¿?()[]{}-_/'\"":
        s = s.replace(ch, " ")
    tokens = [t for t in s.split() if t and t not in STOPWORDS]
    return set(tokens) if tokens else {s.strip()}


def build_name_mapping(
    official_names: List[str],
    solver_names: List[str],
    min_jaccard: float = 0.40,
) -> Dict[str, Optional[str]]:
    """
    Mapea nombre oficial (CSV) -> nombre del solver (solution.json) usando Jaccard sobre tokens.
    """
    solver_index = [(name, _normalize_tokens(name)) for name in solver_names]

    mapping: Dict[str, Optional[str]] = {}
    for off_name in official_names:
        off_tokens = _normalize_tokens(off_name)

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

        mapping[off_name] = best_label if (best_label is not None and best_score >= min_jaccard) else None
    return mapping


def load_official_positions(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    df = pd.read_csv(csv_path)
    need = {"name", "x", "y"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV debe incluir columnas {need}")
    return {str(row["name"]): (float(row["x"]), float(row["y"])) for _, row in df.iterrows()}


def load_solution(path: Path) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    coords_raw = data.get("coords")
    if not isinstance(coords_raw, dict):
        raise ValueError("solution.json debe tener 'coords' como diccionario.")
    coords: Dict[str, Dict[str, float]] = {}
    for name, val in coords_raw.items():
        if isinstance(val, dict):
            x, y = val.get("x"), val.get("y")
            if x is not None and y is not None:
                coords[name] = {"x": float(x), "y": float(y)}
    rel_eval = data.get("rel_eval") or []
    return coords, rel_eval


def bbox_points(
    pts: List[Tuple[float, float]], margin: float = 0.04
) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    mx, my = w * margin, h * margin
    return x0 - mx, x1 + mx, y0 - my, y1 + my


def map_solver_into_rect(
    coords: Dict[str, Dict[str, float]],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> Dict[str, Tuple[float, float]]:
    xs = [c["x"] for c in coords.values()]
    ys = [c["y"] for c in coords.values()]
    sx0, sx1 = min(xs), max(xs)
    sy0, sy1 = min(ys), max(ys)
    sw = max(sx1 - sx0, 1e-6)
    sh = max(sy1 - sy0, 1e-6)
    out: Dict[str, Tuple[float, float]] = {}
    for name, c in coords.items():
        tx = x0 + (c["x"] - sx0) / sw * (x1 - x0)
        ty = y0 + (c["y"] - sy0) / sh * (y1 - y0)
        out[name] = (tx, ty)
    return out


def draw_overlay(
    csv_path: Path,
    solution_path: Path,
    out_path: Path,
    figsize: Tuple[float, float] = (16, 12),
    dpi: int = 150,
) -> None:
    pos_off = load_official_positions(csv_path)
    sol_coords, rel_eval = load_solution(solution_path)

    G_off = nx.Graph()
    for n, p in pos_off.items():
        G_off.add_node(n, pos=p)

    # Recuadro común (nodos oficiales + margen)
    bx0, bx1, by0, by1 = bbox_points(list(pos_off.values()), margin=0.03)
    pos_sol = map_solver_into_rect(sol_coords, bx0, bx1, by0, by1)

    G_sol = nx.Graph()
    for n, p in pos_sol.items():
        G_sol.add_node(n, pos=p)
    for r in rel_eval:
        a, b = r.get("origen"), r.get("destino")
        if not a or not b or a == b:
            continue
        if a not in pos_sol or b not in pos_sol:
            continue
        sat = bool(r.get("satisface", False))
        if G_sol.has_edge(a, b):
            continue
        G_sol.add_edge(a, b, satisface=sat)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # --- Capa 1: oficial (solo nodos y etiquetas; sin aristas del grafo oficial) ---
    pos_off_draw = {n: pos_off[n] for n in G_off.nodes()}
    nx.draw_networkx_nodes(
        G_off,
        pos_off_draw,
        ax=ax,
        node_color=COLOR_OFF_NODE,
        node_size=420,
        edgecolors="white",
        linewidths=0.6,
        alpha=0.92,
    )

    # --- Capa 2: solver ---
    pos_sol_draw = {n: G_sol.nodes[n]["pos"] for n in G_sol.nodes()}
    edge_colors = [
        COLOR_SOL_EDGE_OK if G_sol[u][v].get("satisface", True) else COLOR_SOL_EDGE_BAD
        for u, v in G_sol.edges()
    ]
    nx.draw_networkx_edges(
        G_sol,
        pos_sol_draw,
        ax=ax,
        edge_color=edge_colors,
        width=1.4,
        alpha=0.85,
    )
    nx.draw_networkx_nodes(
        G_sol,
        pos_sol_draw,
        ax=ax,
        node_color=COLOR_SOL_NODE,
        node_size=380,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.9,
    )

    # --- Líneas de correspondencia: cada nodo oficial -> su nodo del solver mapeado ---
    # Esto visualiza "qué tan lejos" está cada nodo oficial respecto al layout del solver.
    mapping = build_name_mapping(
        official_names=list(pos_off_draw.keys()),
        solver_names=list(pos_sol_draw.keys()),
    )
    for off_name, sol_name in mapping.items():
        if not sol_name:
            continue
        if sol_name not in pos_sol_draw:
            continue
        x_off, y_off = pos_off_draw[off_name]
        x_sol, y_sol = pos_sol_draw[sol_name]
        ax.plot(
            [x_off, x_sol],
            [y_off, y_sol],
            linestyle="--",
            color=COLOR_MAP_LINK,
            linewidth=1.0,
            alpha=0.45,
            zorder=4,
        )

    # Etiquetas: solver primero, oficial encima (zorder) para que el mapa tenga nombre en cada nodo
    lab_sol = nx.draw_networkx_labels(
        G_sol,
        pos_sol_draw,
        ax=ax,
        font_size=6,
        font_color="#431407",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.55, edgecolor="none"),
    )
    lab_off = nx.draw_networkx_labels(
        G_off,
        pos_off_draw,
        ax=ax,
        font_size=6,
        font_color="#1e3a8a",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#dbeafe", alpha=0.78, edgecolor="none"),
    )
    for t in lab_sol.values():
        t.set_zorder(11)
    for t in lab_off.values():
        t.set_zorder(12)

    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.06)
    # Mismo criterio que grafoOficial.draw_official_graph: coordenadas CSV tal cual (sin invert_yaxis)
    ax.axis("off")
    ax.set_title(
        "Lugares oficiales y generados por el solver \n",
        fontsize=11,
    )

    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_OFF_NODE, markersize=9, label="Lugares oficiales"),
        Line2D([0], [0], color=COLOR_MAP_LINK, lw=1.2, linestyle="--", label="Conexión oficial->solver (mapeado)"),
        Line2D([0], [0], color=COLOR_SOL_EDGE_OK, lw=3, label="Solver restricción cumplida"),
        Line2D([0], [0], color=COLOR_SOL_EDGE_BAD, lw=3, label="Solver restricción incumplida"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SOL_NODE, markersize=9, label="Lugares generados por el solver"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9, framealpha=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Imagen guardada en: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Superpone nodos oficiales (CSV, sin aristas) y grafo del solver (solution.json)."
    )
    p.add_argument("--csv", type=Path, default=_DEFAULT_CSV, help="Nodos oficiales con x,y")
    p.add_argument("--solution", type=Path, default=_DEFAULT_SOLUTION, help="Salida del solver")
    p.add_argument("-o", "--output", type=Path, default=_DEFAULT_OUT, help="PNG de salida")
    args = p.parse_args()

    draw_overlay(
        csv_path=args.csv,
        solution_path=args.solution,
        out_path=args.output,
    )


if __name__ == "__main__":
    main()

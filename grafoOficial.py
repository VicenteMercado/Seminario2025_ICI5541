import math
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

CSV_PATH = "official_nodes.csv"       # nodos anotados
OUTPUT_JSON = "official_graph.json"   # grafo oficial en formato JSON

NEAR_RATIO = 0.05    # 5% del tamaño -> CERCA_DE
CONNECT_RATIO = 0.10 # 10% del tamaño -> CONECTA


def load_nodes(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"name", "x", "y"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"El CSV debe tener columnas {required_cols}, tiene {df.columns.tolist()}")

    # coordenadas de imagen
    W = df["x"].max() + 10
    H = df["y"].max() + 10
    df["yc"] = H - df["y"]  # y cartesiano (arriba = valor mayor)

    size = max(W, H)
    near_th = NEAR_RATIO * size
    conn_th = CONNECT_RATIO * size

    print(f"Imagen estimada: W={W}, H={H}, near_th={near_th:.1f}, conn_th={conn_th:.1f}")

    return df, near_th, conn_th


def build_relations(df: pd.DataFrame, near_th: float, conn_th: float):
    """Construye lista de relaciones (diccionarios) a partir de las coordenadas."""
    relations = []
    records = df.to_dict("records")

    for a in records:
        for b in records:
            if a["name"] == b["name"]:
                continue

            dx = b["x"]  - a["x"]
            dy = b["yc"] - a["yc"]
            dist = math.hypot(dx, dy)

            # eje dominante para dirección
            if abs(dx) >= abs(dy):
                tipo_dir = "ESTE_DE" if dx > 0 else "OESTE_DE"
            else:
                tipo_dir = "NORTE_DE" if dy > 0 else "SUR_DE"

            # CERCA_DE / CONECTA
            if dist <= near_th:
                relations.append({
                    "origen": a["name"],
                    "tipo": "CERCA_DE",
                    "destino": b["name"],
                    "dist_px": dist
                })
            elif dist <= conn_th:
                relations.append({
                    "origen": a["name"],
                    "tipo": "CONECTA",
                    "destino": b["name"],
                    "dist_px": dist
                })

            # relación direccional (siempre)
            relations.append({
                "origen": a["name"],
                "tipo": tipo_dir,
                "destino": b["name"],
                "dist_px": dist
            })

    print(f"Relaciones derivadas: {len(relations)}")
    return relations


def draw_official_graph(df: pd.DataFrame):
    """
    Dibuja SOLO LOS NODOS oficiales usando las coordenadas del CSV.
    No se dibujan aristas: sirve como referencia visual de las posiciones.
    """
    G = nx.Graph()
    pos = {}

    # nodos con posiciones de imagen
    for _, row in df.iterrows():
        name = row["name"]
        x = float(row["x"])
        y = float(row["y"])
        G.add_node(name)
        pos[name] = (x, y)

    plt.figure(figsize=(8, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=600,
        font_size=7
    )

    # ax = plt.gca()
    # ax.invert_yaxis()

    plt.title("Grafo oficial (solo nodos, coordenadas del mapa)")
    plt.tight_layout()
    plt.show()


def main():
    df, near_th, conn_th = load_nodes(CSV_PATH)
    relations = build_relations(df, near_th, conn_th)

    # JSON de salida (para comparación)
    lugares = sorted(df["name"].unique().tolist())
    relaciones_json = [
        {"origen": r["origen"], "tipo": r["tipo"], "destino": r["destino"]}
        for r in relations
    ]

    output = {
        "lugares": lugares,
        "relaciones": relaciones_json
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Grafo oficial guardado en {OUTPUT_JSON}")

    # Dibujo solo de nodos
    draw_official_graph(df)


if __name__ == "__main__":
    main()

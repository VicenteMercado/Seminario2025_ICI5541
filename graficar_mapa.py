# graficar_mapa.py
import json
import networkx as nx
import matplotlib.pyplot as plt

# Cargar el JSON generado por el otro script
with open("map_relations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

lugares = data["lugares"]
relaciones = data["relaciones"]

# Crear grafo
G = nx.Graph()
for p in lugares:
    G.add_node(p)
for r in relaciones:
    if not G.has_edge(r["origen"], r["destino"]):
        G.add_edge(r["origen"], r["destino"], label=r["tipo"])

# Dibujar
plt.figure(figsize=(12,8))
pos = nx.spring_layout(G, seed=42, k=0.5)
nx.draw_networkx_nodes(G, pos, node_color="#b3d9ff", node_size=700)
nx.draw_networkx_edges(G, pos, edge_color="#999999")
nx.draw_networkx_labels(G, pos, font_size=9)

plt.title("Mapa de relaciones (desde JSON)")
plt.axis("off")
plt.show()
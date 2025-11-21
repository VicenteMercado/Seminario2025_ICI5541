import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

CSV_PATH = "official_nodes.csv"

df = pd.read_csv(CSV_PATH)

G = nx.Graph()
pos = {row["name"]: (row["x"], row["y"]) for _, row in df.iterrows()}
G.add_nodes_from(pos.keys())

plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=8)
plt.title("Nodos oficiales (desde official_nodes.csv)")
#plt.gca().invert_yaxis()  # para que se parezca al mapa
plt.tight_layout()
plt.show()
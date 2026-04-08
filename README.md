# Seminario2025_ICI5541
Proyecto de Seminario de Título: GENERACIÓN DE MAPAS A PARTIR DE NARRACIONES LITERARIAS

## Pipeline de ejecución

Ejecutar en orden:

```bash
# 1. Extracción de relaciones espaciales desde el texto (PDF)
python extraccion_relaciones/relaciones_espaciales.py

# 2. Generación de inecuaciones a partir de las relaciones extraídas
python solver/generar_inecuaciones.py

# 3. Resolución del sistema de restricciones y generación del mapa
python solver/solver_grafos.py

# 4. (Opcional) Comparación con grafo oficial
python comparador/comparar_oficial_vs_solution.py json/official_graph.json json/solution.json
```

## Descripción de cada etapa

1. **relaciones_espaciales.py** - Extrae relaciones espaciales (NORTE_DE, SUR_DE, CERCA_DE, etc.) desde un texto narrativo usando un LLM. Genera `json/map_relations.json`.

2. **generar_inecuaciones.py** - Traduce las relaciones espaciales a inecuaciones serializables (con radios de cercanía y distancias concretas si existen). Genera `json/inequalities.json`.

3. **solver_grafos.py** - Resuelve las inecuaciones con Z3 (solver incremental) y genera las coordenadas del mapa. Produce `json/solution.json` y `img/mapa.svg`.

4. **comparar_oficial_vs_solution.py** - Compara el mapa generado contra un grafo oficial de referencia.

## Archivos de datos

- `json/map_relations.json` - Relaciones extraídas por el LLM
- `json/inequalities.json` - Inecuaciones generadas para el solver
- `json/solution.json` - Coordenadas resueltas y evaluación (CSR)
- `json/official_graph.json` - Grafo oficial de referencia (opcional)
- `img/mapa.svg` - Mapa generado

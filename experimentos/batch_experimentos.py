import subprocess
import json
import csv
import os
import shutil
from pathlib import Path
from datetime import datetime

# Configuración general
N_RUNS = 15

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
EXTRACCION_DIR = ROOT / "extraccion_relaciones"
SOLVER_DIR = ROOT / "solver"
COMPARADOR_DIR = ROOT / "comparador"

EXTRACT_CMD = ["python", "relaciones_espaciales.py"]
SOLVER_CMD = ["python", "solver_grafos.py"]

COMPARE_SCRIPT = COMPARADOR_DIR / "comparar_oficial_vs_solution.py"
OFFICIAL_GRAPH = JSON_DIR / "official_graph.json"

RUNS_DIR = ROOT / "runs_experimentos"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RUNS_DIR / "resultados_experimentos.csv"


def ejecutar(cmd, cwd=None):
    """
    Ejecuta un comando y lanza excepción si falla.
    """
    print(f"\n>>> Ejecutando: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Comando falló: {' '.join(cmd)}")
    return res


def asegurar_csv_con_header(path_csv):
    """
    Crea el CSV con encabezado si aún no existe.
    """
    if not path_csv.exists():
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run",
                "timestamp",
                "solver_CSR",              # CSR interno del solver_grafos
                "csr_oficial",             # CSR comparado con grafo oficial
                "lugares_oficiales",
                "lugares_solution",
                "lugares_oficiales_mapeados",
                "relaciones_oficiales_total",
                "relaciones_oficiales_evaluables",
                "relaciones_oficiales_satisfechas"
            ])


def registrar_fila(path_csv, row):
    with open(path_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    asegurar_csv_con_header(CSV_PATH)

    for run_idx in range(1, N_RUNS + 1):
        print("\n" + "=" * 60)
        print(f" CORRIDA {run_idx}/{N_RUNS}")
        print("=" * 60)

        # Carpeta propia para esta corrida
        run_dir = RUNS_DIR / f"run_{run_idx:02d}"
        run_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().isoformat(timespec="seconds")
        solver_CSR = None
        csr_oficial = None
        resumen_global = {}

        try:
            # 1) EXTRAER RELACIONES ESPACIALES (map_relations.json)
            ejecutar(EXTRACT_CMD, cwd=EXTRACCION_DIR)

            map_rel = JSON_DIR / "map_relations.json"
            if map_rel.exists():
                shutil.copyfile(map_rel, run_dir / "map_relations.json")

            # 2) CORRER SOLVER (solver_grafos.py -> solution.json)
            ejecutar(SOLVER_CMD, cwd=SOLVER_DIR)

            sol_src = JSON_DIR / "solution.json"
            sol_dst = run_dir / "solution.json"
            if sol_src.exists():
                shutil.copyfile(sol_src, sol_dst)
            else:
                print(" No se encontró solution.json después del solver.")
            
            # Leer CSR interno del solver
            if sol_dst.exists():
                with open(sol_dst, "r", encoding="utf-8") as f:
                    sol_data = json.load(f)
                solver_CSR = float(sol_data.get("CSR", 0.0))

            # 3) COMPARAR CON GRAFO OFICIAL
            comp_dst = run_dir / "comparison_official_vs_solution.json"
            comparar_cmd = [
                "python",
                str(COMPARE_SCRIPT),
                str(OFFICIAL_GRAPH),
                str(sol_dst),
                "--output",
                str(comp_dst),
            ]
            ejecutar(comparar_cmd, cwd=COMPARADOR_DIR)

            if not comp_dst.exists():
                print("⚠️ No se encontró comparison_official_vs_solution.json tras comparar.")

            # Leer métricas globales de la comparación
            if comp_dst.exists():
                with open(comp_dst, "r", encoding="utf-8") as f:
                    comp_data = json.load(f)
                resumen_global = comp_data.get("resumen_global", {})
                csr_oficial = float(resumen_global.get("csr_oficial", 0.0))

            # 4) Registrar en CSV
            row = [
                run_idx,
                timestamp,
                solver_CSR,
                csr_oficial,
                resumen_global.get("lugares_oficiales"),
                resumen_global.get("lugares_solution"),
                resumen_global.get("lugares_oficiales_mapeados"),
                resumen_global.get("relaciones_oficiales_total"),
                resumen_global.get("relaciones_oficiales_evaluables"),
                resumen_global.get("relaciones_oficiales_satisfechas"),
            ]
            registrar_fila(CSV_PATH, row)
            print(f"Corrida {run_idx} registrado en {CSV_PATH}")

        except Exception as e:
            print(f" Error en la corrida {run_idx}: {e}")
            # También registramos el fallo en el CSV
            row = [
                run_idx,
                timestamp,
                solver_CSR,
                csr_oficial,
                resumen_global.get("lugares_oficiales"),
                resumen_global.get("lugares_solution"),
                resumen_global.get("lugares_oficiales_mapeados"),
                resumen_global.get("relaciones_oficiales_total"),
                resumen_global.get("relaciones_oficiales_evaluables"),
                resumen_global.get("relaciones_oficiales_satisfechas"),
            ]
            registrar_fila(CSV_PATH, row)

    print("\nTerminaron todas las corridas.")
    print(f"Resultados agregados en: {CSV_PATH}")
    print(f"Detalle de cada run en carpetas dentro de: {RUNS_DIR}")


if __name__ == "__main__":
    main()

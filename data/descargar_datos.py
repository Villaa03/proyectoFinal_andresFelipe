"""Descarga el dataset Adult (UCI, conjunto 2) a data/adult_income.csv."""

from pathlib import Path
import sys

from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import preparar_dataset

DESTINO = ROOT / "data" / "adult_income.csv"


def main() -> None:
    print("Descargando Adult Income desde UCI ML Repository (ID=2)...")
    adult = fetch_ucirepo(id=2)
    datos = adult.data.features.join(adult.data.targets)
    datos = preparar_dataset(datos)
    datos.to_csv(DESTINO, index=False)
    print(f"Dataset guardado en {DESTINO} ({len(datos):,} filas).")
    print("La copia local no se versiona; el script permite regenerarla.")


if __name__ == "__main__":
    main()

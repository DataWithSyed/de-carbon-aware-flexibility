"""Build data/processed/hourly_de.csv.gz from cached Energy-Charts JSON (run fetch_data.py first)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carbonflex import data  # noqa: E402

df = data.load_raw(ROOT / "data" / "raw")
df = df.loc["2023-10-31 23:00":"2026-10-31 22:00"]
out = ROOT / "data" / "processed" / "hourly_de.csv.gz"
out.parent.mkdir(parents=True, exist_ok=True)
df.round(2).to_csv(out, index_label="utc")
print(out, df.shape, df.index.min(), df.index.max())
print(df.isna().sum()[df.isna().sum() > 0])

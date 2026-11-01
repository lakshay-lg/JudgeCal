"""Download lmsys/mt_bench_human_judgments (CC-BY-4.0) into data/raw/."""

from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/main/data/"
FILES = {
    "human.parquet": "human-00000-of-00001-25f4910818759289.parquet",
    "gpt4_pair.parquet": "gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet",
}

out = Path(__file__).resolve().parents[1] / "data" / "raw"
out.mkdir(parents=True, exist_ok=True)
for name, remote in FILES.items():
    dest = out / name
    if not dest.exists():
        print(f"downloading {name}")
        urlretrieve(BASE + remote, dest)
print("done:", sorted(p.name for p in out.iterdir()))

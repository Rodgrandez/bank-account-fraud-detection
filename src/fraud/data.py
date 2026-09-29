import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from fraud import config


def download(dest: Path = config.DATA_RAW) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / config.RAW_FILE
    if not target.exists():
        subprocess.run(["kaggle", "datasets", "download", config.KAGGLE_DATASET, "-f", config.RAW_FILE,
                        "-p", str(dest)], check=True)
        archive = dest / f"{config.RAW_FILE}.zip"
        if archive.exists():                         # Kaggle zips large single-file downloads
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
            archive.unlink()
    return target


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = raw.copy()
    for c in config.SENTINEL_NEGATIVE:
        if c in df.columns:
            df[c] = df[c].where(df[c] >= 0)
    constant = [c for c in df.columns
                if c not in (config.TARGET, config.MONTH) and df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=constant), constant


def build_interim(raw_path: Path, out: Path = config.INTERIM) -> Path:
    df, dropped = clean(pd.read_csv(raw_path))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    (out.parent / "dropped.json").write_text(json.dumps(dropped), encoding="utf-8")
    return out


def load_interim(path: Path = config.INTERIM) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    return pd.read_parquet(path), json.loads((path.parent / "dropped.json").read_text(encoding="utf-8"))

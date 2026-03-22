"""
Unified table writer — supports parquet and CSV with a single call.

Used by both Octus ingestion and SimFin ingestion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def write_table(
    df: pd.DataFrame,
    path: str | Path,
    fmt: Literal["parquet", "csv"],
) -> None:
    """
    Write *df* to *path* in either parquet or CSV format.

    Creates parent directories automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table_format: {fmt!r}. Use 'parquet' or 'csv'.")


def read_table(path: str | Path) -> pd.DataFrame:
    """
    Read a parquet or CSV file, auto-detected from the file extension.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Cannot auto-detect format for extension {suffix!r}")

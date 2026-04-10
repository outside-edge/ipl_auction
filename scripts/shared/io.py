#!/usr/bin/env python3
"""
File I/O utilities for consistent data handling.

Supports parquet as primary format with CSV fallback for human-readable files.
"""

from pathlib import Path

import pandas as pd


def save_dataset(df: pd.DataFrame, path: Path, format: str = "parquet") -> Path:
    """
    Save DataFrame with consistent format.

    Args:
        df: DataFrame to save
        path: Target path (extension will be adjusted based on format)
        format: "parquet" or "csv"

    Returns:
        Actual path written
    """
    path = Path(path)

    if format == "parquet":
        out_path = path.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
    else:
        out_path = path.with_suffix(".csv")
        df.to_csv(out_path, index=False)

    return out_path


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load dataset, preferring parquet over CSV if both exist.

    Args:
        path: Base path (with or without extension)

    Returns:
        Loaded DataFrame
    """
    path = Path(path)
    parquet_path = path.with_suffix(".parquet")
    csv_path = path.with_suffix(".csv")

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    elif path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"No dataset found at {path} (.parquet or .csv)")


def dataset_exists(path: Path) -> bool:
    """
    Check if dataset exists in either parquet or CSV format.

    Args:
        path: Base path (with or without extension)

    Returns:
        True if file exists
    """
    path = Path(path)
    return (
        path.with_suffix(".parquet").exists()
        or path.with_suffix(".csv").exists()
        or path.exists()
    )

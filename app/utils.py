import pandas as pd
import numpy as np

def valeurs_uniques_triees(df: pd.DataFrame, col: str) -> list:
    if col in df.columns:
        vals = df[col].dropna().unique()
        return sorted([v for v in vals if str(v).strip() != ""])
    return []

def plage_numerique(series: pd.Series, default_min: float, default_max: float) -> tuple[float, float]:
    if series.notna().any():
        return float(np.nanmin(series)), float(np.nanmax(series))
    return default_min, default_max

def top_n_comptes(s: pd.Series, n: int = 10) -> pd.DataFrame:
    return s.value_counts(dropna=False).head(n).rename_axis("valeur").reset_index(name="nb")

def colonnes_numeriques_disponibles(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

def fmt_eur(x) -> str:
    try:
        return f"{int(x):,} €".replace(",", " ")
    except Exception:
        return str(x)

def quantile_labels(cats) -> list[str]:
    return [f"{int(iv.left):,} € – {int(iv.right):,} €".replace(',', ' ') for iv in cats.categories]
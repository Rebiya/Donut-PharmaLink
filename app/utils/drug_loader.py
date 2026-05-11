"""FDA drug dataset loading and caching utilities."""

import json
import logging
from pathlib import Path
from typing import List

import kagglehub
import pandas as pd

logger = logging.getLogger(__name__)


POTENTIAL_DRUG_COLUMNS = [
    "drug_name",
    "brand_name",
    "generic_name",
    "substance_name",
    "name",
]


def _extract_names_from_dataframe(df: pd.DataFrame) -> List[str]:
    names: List[str] = []
    for col in POTENTIAL_DRUG_COLUMNS:
        if col in df.columns:
            values = (
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .tolist()
            )
            names.extend(values)
    return names


def load_fda_drug_names(
    dataset_ref: str, cache_path: Path, local_files_only: bool = False
) -> List[str]:
    """Download FDA dataset from KaggleHub and return cleaned unique drug names."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        logger.info("Loading cached drug names from %s", cache_path)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if local_files_only:
        raise FileNotFoundError(
            f"Drug cache missing and local_files_only is enabled: {cache_path}"
        )

    logger.info("Downloading FDA dataset: %s", dataset_ref)
    dataset_path = Path(kagglehub.dataset_download(dataset_ref))
    csv_files = sorted(dataset_path.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in downloaded FDA dataset.")

    all_names: List[str] = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            all_names.extend(_extract_names_from_dataframe(df))
        except Exception as exc:  # pragma: no cover - defensive for malformed files
            logger.warning("Skipping %s due to parsing error: %s", csv_file, exc)

    cleaned = sorted({name.lower() for name in all_names if len(name.strip()) > 2})
    cache_path.write_text(json.dumps(cleaned, ensure_ascii=True), encoding="utf-8")
    logger.info("Stored %s unique FDA drug names", len(cleaned))
    return cleaned

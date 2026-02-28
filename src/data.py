import shutil
from pathlib import Path

import kagglehub
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DATE = pd.Timestamp("2012-01-01")


def download_dunnhumby(dest: Path = Path("data/raw")) -> Path:
    """Downloads the Complete Journey dataset from Kaggle."""
    load_dotenv()
    path = kagglehub.dataset_download("frtgnn/dunnhumby-the-complete-journey")

    # Create destination directory if it doesn't exist
    dest.mkdir(parents=True, exist_ok=True)

    # Copy downloaded files
    for f in Path(path).glob("*.csv"):
        shutil.copy(f, dest / f.name)

    print(f"✅ Files copied to: {dest}")
    return dest


def load_transactions(raw_path: str = "data/raw") -> pd.DataFrame:
    """Loads transactions and correctly converts the DAY field to dates."""
    df = pd.read_csv(f"{raw_path}/transaction_data.csv")
    df["date"] = BASE_DATE + pd.to_timedelta(df["DAY"] - 1, unit="D")
    return df


def validate_raw_files(raw_path: str = "data/raw") -> None:
    """Verifies that all expected raw files are present."""
    expected_files = [
        "transaction_data.csv",
        "hh_demographic.csv",
        "coupon.csv",
        "coupon_redempt.csv",
        "campaign_table.csv",
        "campaign_desc.csv",
        "product.csv",
    ]

    base_dir = Path(raw_path)
    missing = []

    for filename in expected_files:
        if not (base_dir / filename).exists():
            missing.append(filename)

    assert len(missing) == 0, f"Missing files in {raw_path}: {missing}"
    print(f"✅ All {len(expected_files)} expected files are present.")

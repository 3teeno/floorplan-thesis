import logging
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# CubiCasa5K is licensed CC BY-NC 4.0 (non-commercial use only): https://zenodo.org/record/2613548
DATASET_URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "cubicasa5k.zip"


def download(url: str = DATASET_URL, dest: Path = ZIP_PATH, min_free_gb: float = 12.0) -> None:
    """Stream the 5.5GB dataset zip to disk. Refuses to start if there isn't
    enough free space for the zip plus its extracted contents.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(dest.parent).free / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free at {dest.parent.resolve()}; need ~{min_free_gb:.0f} GB "
            "for the 5.5 GB zip plus its extracted contents. Free up space first."
        )

    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    log.info(f"Downloaded to {dest}")


def extract(zip_path: Path = ZIP_PATH, dest: Path = DATA_DIR) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    log.info(f"Extracted to {dest}")


if __name__ == "__main__":
    download()
    extract()

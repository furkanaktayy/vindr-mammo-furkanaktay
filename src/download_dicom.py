import pandas as pd
import requests
from pathlib import Path
import time

CSV_PATH = "data/processed/subset_samples.csv"
OUT_DIR = Path("data/raw/dicoms")

# Doğrudan indirilebilen gerçek PhysioNet dosya yolu
BASE_URL = "https://physionet.org/static/published-projects/vindr-mammo/1.0.0/images"


def download_dicom(study_id, image_id):
    """
    Verilen study_id ve image_id'ye ait DICOM dosyasını indirir.
    """
    study_folder = OUT_DIR / str(study_id)
    study_folder.mkdir(parents=True, exist_ok=True)

    out_path = study_folder / f"{image_id}.dicom"

    # Dosya zaten varsa indirme
    if out_path.exists():
        print(f"[SKIP] Already exists → {out_path}")
        return True

    # Gerçek download linki
    url = f"{BASE_URL}/{study_id}/{image_id}.dicom?download=1"

    try:
        response = requests.get(url, timeout=15)

        if response.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(response.content)
            print(f"[OK] Downloaded → {out_path}")
            return True
        
        else:
            print(f"[ERROR] Failed {response.status_code}: {url}")
            return False

    except Exception as e:
        print(f"[ERROR] Exception while downloading {url}: {e}")
        return False


def download_all():
    print("Reading subset_samples.csv...")
    df = pd.read_csv(CSV_PATH)

    print(f"Total samples to download: {len(df)}")

    success = 0
    fail = 0

    for idx, row in df.iterrows():
        study_id = row["study_id"]
        image_id = row["image_id"]

        ok = download_dicom(study_id, image_id)
        time.sleep(0.2)  # çok hızlı indirmeyi engelle (site limiti)

        if ok:
            success += 1
        else:
            fail += 1

    print("\n============================")
    print("Download completed.")
    print(f"Successful: {success}")
    print(f"Failed: {fail}")
    print("============================")


if __name__ == "__main__":
    download_all()
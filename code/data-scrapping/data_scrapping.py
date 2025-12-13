import os
import zipfile
import shutil
from pathlib import Path

DATASET = "sulianova/cardiovascular-disease-dataset"
OUT_DIR = Path("/Users/jishan/Documents/AIBAS_Heart-Disease-Predictor/data/Cardiovascular_disease_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.system(f'kaggle datasets download -d {DATASET} -p "{OUT_DIR}"')

zip_files = sorted(OUT_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
if not zip_files:
    raise SystemExit("No zip downloaded. Kaggle auth is not set correctly.")
zip_path = zip_files[0]

wanted = "cardio_train.csv"
with zipfile.ZipFile(zip_path, "r") as z:
    names = z.namelist()
    if wanted not in names:
        raise SystemExit(f"{wanted} not found. Files in zip: {names}")
    z.extract(wanted, OUT_DIR)

zip_path.unlink()

main_csv = OUT_DIR / wanted
joint_path = OUT_DIR / "joint_data_collection.csv"
shutil.copyfile(main_csv, joint_path)


main_csv.unlink()

print("Task Complete. CSV remains:")
print([p.name for p in OUT_DIR.glob("*.csv")])

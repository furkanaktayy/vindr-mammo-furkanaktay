import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

TRAIN_COUNT = 45
VAL_COUNT = 5

subset_path = "data/processed/subset_samples.csv"
images_path = "data/processed/images"
labels_path = "data/processed/labels_yolo"
output_root = "data/processed/yolo"

def ensure_dir(p):
 p = Path(p)
 p.mkdir(parents=True, exist_ok=True)

def main():
 print("Loading subset_samples.csv...")
 df = pd.read_csv(subset_path)

 df_pairs = df[["study_id", "image_id"]].drop_duplicates()

 label_files = {Path(f).stem for f in Path(labels_path).glob("*.txt")}
 df_pairs["has_lesion"] = df_pairs["image_id"].isin(label_files).astype(int)

 print("\nLesion distribution:")
 print(df_pairs["has_lesion"].value_counts())

 # Train-Val split
 df_train, df_val = train_test_split(
  df_pairs,
  test_size=VAL_COUNT,
  stratify=df_pairs["has_lesion"],
  random_state=42
 )

    # TRAIN = total - val
 assert len(df_val) == VAL_COUNT
 assert len(df_train) == TRAIN_COUNT

 print("Final split sizes:")
 print(f"Train = {len(df_train)}")
 print(f"Val   = {len(df_val)}")

 # YOLO folder structure
 for split in ["train", "val"]:
  ensure_dir(f"{output_root}/images/{split}")
  ensure_dir(f"{output_root}/labels/{split}")

  # Copy files
  def copy_files(df_split, split_name):
   for _, row in df_split.iterrows():
    img_id = row["image_id"]

    png_src = Path(images_path) / f"{img_id}.png"
    txt_src = Path(labels_path) / f"{img_id}.txt"

    png_dst = Path(output_root) / "images" / split_name / f"{img_id}.png"
    txt_dst = Path(output_root) / "labels" / split_name / f"{img_id}.txt"

    if png_src.exists():
     shutil.copy(png_src, png_dst)

    if txt_src.exists():  # lezyonsuzlarda txt yok, o yüzden kontrol
     shutil.copy(txt_src, txt_dst)

 copy_files(df_train, "train")
 copy_files(df_val, "val")

 # data.yaml oluştur
 yaml_path = Path(output_root) / "data.yaml"
 with open(yaml_path, "w") as f:
  f.write(
   "train: images/train\n"
   "val: images/val\n\n"
   "nc: 1\n"
   "names: ['lesion']\n"
  )

 print("\n✔ YOLO train/val dataset created successfully!")
 print(f"data.yaml saved to: {yaml_path}")

if __name__ == "__main__":
    main()
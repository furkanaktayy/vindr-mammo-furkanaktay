import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil


TRAIN_COUNT = 112
VAL_COUNT = 12
TEST_COUNT = 12

subset_path = "data/processed/subset_samples_150.csv"
images_path = "data/processed/images"
labels_path = "data/processed/labels_yolo"
output_root = "data/processed/yolo"


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading subset_samples.csv...")
    df = pd.read_csv(subset_path)

    # ------------------------------
    # 1) study_id kolonunu otomatik bul
    # ------------------------------
    study_cols = [c for c in df.columns if c.startswith("study_id")]
    if len(study_cols) == 0:
        raise ValueError("❌ ERROR: No study_id column found in CSV.")

    study_col = study_cols[0]
    print(f"Detected study_id column → {study_col}")

    # ------------------------------
    # 2) Unique study-image pairs
    # ------------------------------
    df_pairs = df[[study_col, "image_id"]].drop_duplicates()

    # ------------------------------
    # 3) has_lesion flag
    # ------------------------------
    label_files = {Path(f).stem for f in Path(labels_path).glob("*.txt")}
    df_pairs["has_lesion"] = df_pairs["image_id"].isin(label_files).astype(int)

    print("\nLesion distribution:")
    print(df_pairs["has_lesion"].value_counts())

    # ------------------------------
    # 4) Split test first
    # ------------------------------
    df_temp, df_test = train_test_split(
        df_pairs,
        test_size=TEST_COUNT,
        stratify=df_pairs["has_lesion"],
        random_state=42
    )

    # ------------------------------
    # 5) Split train + val
    # ------------------------------
    df_train, df_val = train_test_split(
        df_temp,
        test_size=VAL_COUNT,
        stratify=df_temp["has_lesion"],
        random_state=42
    )

    # ------------------------------
    # 6) Assertions
    # ------------------------------
    assert len(df_train) == TRAIN_COUNT
    assert len(df_val) == VAL_COUNT
    assert len(df_test) == TEST_COUNT

    print("\nFinal split sizes:")
    print(f"Train = {len(df_train)}")
    print(f"Val   = {len(df_val)}")
    print(f"Test  = {len(df_test)}")

    # ------------------------------
    # 7) Prepare folders
    # ------------------------------
    for split in ["train", "val", "test"]:
        ensure_dir(f"{output_root}/images/{split}")
        ensure_dir(f"{output_root}/labels/{split}")

    # ------------------------------
    # 8) File copy function
    # ------------------------------
    def copy_files(df_split, split_name):
        for _, row in df_split.iterrows():
            img_id = row["image_id"]

            png_src = Path(images_path) / f"{img_id}.png"
            txt_src = Path(labels_path) / f"{img_id}.txt"

            png_dst = Path(output_root) / "images" / split_name / f"{img_id}.png"
            txt_dst = Path(output_root) / "labels" / split_name / f"{img_id}.txt"

            if png_src.exists():
                shutil.copy(png_src, png_dst)

            if txt_src.exists():  # boş veya dolu txt'ler için
                shutil.copy(txt_src, txt_dst)

    copy_files(df_train, "train")
    copy_files(df_val, "val")
    copy_files(df_test, "test")

    # ------------------------------
    # 9) data.yaml oluştur
    # ------------------------------
    yaml_path = Path(output_root) / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "nc: 1\n"
            "names: ['lesion']\n"
        )

    print("\n✔ YOLO train/val/test dataset created successfully!")
    print(f"data.yaml saved to: {yaml_path}")


if __name__ == "__main__":
    main()
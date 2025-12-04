import pandas as pd
import numpy as np
from pathlib import Path

finding_path = "data/raw/finding_annotations.csv"
breast_path = "data/raw/breast-level_annotations.csv"
metadata_path = "data/raw/metadata.csv"
existing_csv = "data/processed/subset_samples.csv"

TARGET_LESION = 110
TARGET_NO_LESION = 40
OUT_CSV = "data/processed/subset_samples_150.csv"


def load_data():
    findings = pd.read_csv(finding_path)
    breast = pd.read_csv(breast_path)
    meta = pd.read_csv(metadata_path)

    if "SOP Instance UID" in meta.columns:
        meta = meta.rename(columns={"SOP Instance UID": "image_id"})

    existing = pd.read_csv(existing_csv)
    return findings, breast, meta, existing


def extract_category(x):
    try:
        lst = eval(x)
        return lst[0] if len(lst) > 0 else None
    except:
        return None


def main():
    print("Loading data...")
    findings, breast, meta, existing = load_data()

    # kategori parse
    findings["category"] = findings["finding_categories"].apply(extract_category)
    valid_cats = ["Mass", "Suspicious Calcification", "Asymmetry", "Architectural Distortion"]

    lesion_df = findings[findings["category"].isin(valid_cats)]

    lesion_ids = set(lesion_df["image_id"].unique())
    exist_ids = set(existing["image_id"].unique())

    exist_lesion = existing[existing["image_id"].isin(lesion_ids)]
    exist_no_lesion = existing[~existing["image_id"].isin(lesion_ids)]

    print(f"Existing lesion count     = {len(exist_lesion)}")
    print(f"Existing no-lesion count = {len(exist_no_lesion)}")

    need_lesion = max(0, TARGET_LESION - len(exist_lesion))
    need_no_lesion = max(0, TARGET_NO_LESION - len(exist_no_lesion))

    print(f"Need to add lesion     = {need_lesion}")
    print(f"Need to add no-lesion = {need_no_lesion}")

    # yeni veriler için pool
    lesion_pool = lesion_df[
        ~lesion_df["image_id"].isin(exist_ids)
    ][["study_id", "image_id"]].drop_duplicates()

    all_pairs = breast[["study_id", "image_id"]].drop_duplicates()
    no_lesion_pool = all_pairs[
        ~all_pairs["image_id"].isin(lesion_ids) &
        ~all_pairs["image_id"].isin(exist_ids)
    ]

    rng = np.random.default_rng(seed=42)
    new_lesion = lesion_pool.sample(n=need_lesion, random_state=42)
    new_no_lesion = no_lesion_pool.sample(n=need_no_lesion, random_state=42)

    final_df = pd.concat(
        [existing[["study_id", "image_id"]], new_lesion, new_no_lesion],
        ignore_index=True
    )

    print(f"Final total count = {len(final_df)}")

    # merge etme
    df_merge = final_df.merge(meta, on="image_id", how="left")
    df_merge = df_merge.merge(findings, on="image_id", how="left")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_merge.to_csv(OUT_CSV, index=False)

    print("Completed. Saved:", OUT_CSV)
    print("Unique image count:", df_merge["image_id"].nunique())


if __name__ == "__main__":
    main()
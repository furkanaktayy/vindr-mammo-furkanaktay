import pandas as pd
from pathlib import Path


EXISTING_CSV = "data/processed/subset_50_samples.csv"
METADATA_CSV = "data/raw/metadata.csv"
FINDINGS_CSV = "data/raw/finding_annotations.csv"

targer_lesion = 110
target_no_lesion = 40
out_csv = "data/processed/subset_150_samples.csv"

def load_data():
 df_existing = pd.read_csv(EXISTING_CSV)
 df_meta = pd.read_csv(METADATA_CSV)
 df_find = pd.read_csv(FINDINGS_CSV)
 return df_existing, df_meta, df_find

def main():
 df_existing, df_meta, df_find = load_data()

 print("Loaded existing subset:", len(df_existing))

 lesion_ids = df_find["image_id"].unique()
 df_meta["has_lesion"] = df_meta["image_id"].isin(lesion_ids).astype(int)

 # Varolan subset'teki dağılım
 exist_lesion = df_existing[df_existing["image_id"].isin(lesion_ids)]
 exist_non = df_existing[~df_existing["image_id"].isin(lesion_ids)]

 print(f"Mevcut lesion     : {len(exist_lesion)}")
 print(f"Mevcut non-lesion: {len(exist_non)}")

 need_lesion = max(0, targer_lesion - len(exist_lesion))
 need_non = max(0, target_no_lesion - len(exist_non))

 print(f"\nWill be added lesion: {need_lesion}")
 print(f"Will be added no lesion : {need_non}")

 existing_ids = set(df_existing["image_id"].unique())

 df_lesion_pool = df_meta[(df_meta["has_lesion"] == 1) & (~df_meta["image_id"].isin(existing_ids))]

 df_non_pool = df_meta[(df_meta["has_lesion"] == 0) & (~df_meta["image_id"].isin(existing_ids))]

 df_new_lesion = df_lesion_pool.sample(n=need_lesion, random_state=42)
 df_new_non = df_non_pool.sample(n=need_non, random_state=42)

 df_new = pd.concat([df_existing, df_new_lesion, df_new_non], ignore_index=True)

 print("\nTotal new data:", len(df_new))

 df_new.to_csv(out_csv, index=False)
 print(f"✔ Yeni subset kaydedildi: {out_csv}")


if __name__ == "__main__":
    main()
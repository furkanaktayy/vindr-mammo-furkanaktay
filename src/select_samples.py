import pandas as pd
import numpy as np
from pathlib import Path

finding_path = "/Users/furkanaktay/Desktop/tübitak/2247-c_IlkayOksuz/task/vindr-mammo-furkanaktay/data/raw/finding_annotations.csv"
breast_path = "/Users/furkanaktay/Desktop/tübitak/2247-c_IlkayOksuz/task/vindr-mammo-furkanaktay/data/raw/breast-level_annotations.csv"
metadata_path = "data/raw/metadata.csv"


# Total 50 adet sample (40 adet lezyonlu, 10 adet lezyonsuz) seçilir
# Lezyonlular çeşitli lezyon tiplerinden seçilir. Farklı hastalardan veri alınmasına dikkat edilir
def select_samples(finding_path, breast_path, metadata_path, out_csv= "data/processed/subset_samples.csv", number_lesion=40, number_noLesion=10):
 print("reading csv files...")
 findings = pd.read_csv(finding_path)
 breast = pd.read_csv(breast_path)
 metadata = pd.read_csv(metadata_path)
 print("csv file read.")

# finding_categories kolonu parse edilir
 def extract_category(x):
  try:
   lst = eval(x)
   return lst[0] if len(lst) > 0 else None
  except:
   return None
 
 findings["category"] = findings["finding_categories"].apply(extract_category)
 
 valid_categories = ["Mass", "Suspicious Calcification", "Asymmetry", "Architectural Distortion"]
 lesion_df = findings[findings["category"].isin(valid_categories)]

 # Her kategori için image_id ve study_id 
 def get_pairs(df, category_name):
  sub = df[df["category"] == category_name]
  return sub[["study_id", "image_id"]].drop_duplicates().values.tolist()
 
 mass_pairs = get_pairs(lesion_df, "Mass")
 print(f"Number of Mass: {len(mass_pairs)}")
 calc_pairs = get_pairs(lesion_df, "Suspicious Calcification")
 print(f"Number of Suspicious Calcification: {len(calc_pairs)}")
 asym_pairs = get_pairs(lesion_df, "Asymmetry")
 print(f"Number of Asymmetry: {len(asym_pairs)}")
 dist_pairs = get_pairs(lesion_df, "Architectural Distortion")
 print(f"Number of Architectural Distortion: {len(dist_pairs)}")

 rng = np.random.default_rng(seed=20)
 
 def random_select(pairs, n):
  if len(pairs) == 0:
   return []
  n_actual = min(n, len(pairs))
  idx = rng.choice(len(pairs), n_actual, replace=False)
  return [pairs[i] for i in idx]
 
 mass_selection = random_select(mass_pairs, 10)
 calc_selection = random_select(calc_pairs, 10)
 asym_selection = random_select(asym_pairs, 10)
 dist_selection = random_select(dist_pairs, 10)

 lesion_pairs = mass_selection + calc_selection + asym_selection + dist_selection
 lesion_pairs = list(set(tuple(x) for x in lesion_pairs))
 print(f"Total selected lesion image number: {len(lesion_pairs)}")

 print("Selecting no lesion samples...")
 all_pairs = breast[["study_id", "image_id"]].drop_duplicates().values.tolist()
 lesion_set = set(lesion_pairs)
 no_lesion_candidates = [tuple(x) for x in all_pairs if tuple(x) not in lesion_set]

 n_no_lesion_actual = min(number_noLesion, len(no_lesion_candidates))
 no_lesion_selection = rng.choice(no_lesion_candidates, n_no_lesion_actual, replace=False)
 print(f"Number of no lesion sample selected: {len(no_lesion_selection)}")

 final_pairs = lesion_pairs + list(no_lesion_selection)
 pair_df = pd.DataFrame(final_pairs, columns=["study_id", "image_id"])
 subset_breast = pair_df.merge(breast, on=["study_id", "image_id"], how="left")
 
 possible_cols = [c for c in metadata.columns if "image" in c.lower()]

 if len(possible_cols) == 0:
  print("⚠ ERROR: No column in metadata looks like an image_id!")
  print("Metadata columns:", metadata.columns)
  return

 meta_img_col = possible_cols[0]
 print(f"Metadata image column detected: {meta_img_col}\n")

 subset_df = subset_breast.merge(metadata, left_on="image_id", right_on=meta_img_col, how="left")

 
 Path("data/processed").mkdir(parents=True, exist_ok=True)
 subset_df.to_csv(out_csv, index=False)

 print("Process completed.")
 print(f"Number of final selected sample image: {len(subset_df)}")
 print(f"Output file: {out_csv}")

if __name__ == "__main__":
 select_samples(finding_path, breast_path, metadata_path)
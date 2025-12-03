import pandas as pd
import ast
from pathlib import Path

finding_path = "data/raw/finding_annotations.csv"
subset_path = "data/processed/subset_samples.csv"
labels_dir = "data/processed/labels_yolo"

CLASS_MAP = {
    "Mass": 0,
    "Suspicious Calcification": 1,
    "Asymmetry": 2,
    "Architectural Distortion": 3,
}

def extract_category(cat_str):
 try:
  lst = ast.literal_eval(cat_str)
  if not lst:
   return None
  c = lst[0]
  if "Asymmetry" in c:
   return "Asymmetry" 
  return c
 except:
  return None

def convert_to_yolo(xmin, ymin, xmax, ymax, imgw, imgh):
 x_center = (xmin + xmax) / 2.0 / imgw
 y_center = (ymin + ymax) / 2.0 / imgw
 w = (xmax - xmin) / imgw
 h = (ymax - ymin) / imgh
 return x_center, y_center, w, h

def create_labels():
 print("Reading the csv file...")
 find_df = pd.read_csv(finding_path)
 subset_df = pd.read_csv(subset_path)

 print(f"Subset image count (rows): {len(subset_df)}")
 find_df["category"] = find_df["finding_categories"].apply(extract_category)
 valid_categories = list(CLASS_MAP.keys())
 lesion_df = find_df[find_df["category"].isin(valid_categories)]

 merged = lesion_df.merge(subset_df[["study_id", "image_id"]],
                          on = ["study_id", "image_id"],
                          how = "inner")
 print(f"Number of lesion annotations matched with subset: {len(merged)}")

 labels_root = Path(labels_dir)
 labels_root.mkdir(parents=True, exist_ok=True)

 file_count = 0
 skipped = 0

 for (study_id, image_id), group in merged.groupby(["study_id", "image_id"]):
  out_path = labels_root / f"{image_id}.txt"
  lines = []
  for _, row in group.iterrows():
   category = row["category"]
   if category not in CLASS_MAP:
    skipped += 1
    continue

   class_id = CLASS_MAP[category]

   xmin = row["xmin"]
   xmax = row["xmax"]
   ymin = row["ymin"]
   ymax = row["ymax"]
   img_w = row["width"]
   img_h = row["height"]

   xc, yc, w, h = convert_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
   lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

   if lines:
    with open(out_path, "w") as f:
     f.write("\n".join(lines))
    file_count += 1

 print(f"Completed. YOLO label files created: {file_count}")
 print(f"Skipped annotations: {skipped}")
 print(f"Labels saved to: {labels_dir}")

if __name__ == "__main__":
 create_labels()
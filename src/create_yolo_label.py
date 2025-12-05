import pandas as pd
import ast
from pathlib import Path

finding_path = "data/raw/finding_annotations.csv"
subset_path = "data/processed/subset_samples_150.csv"
labels_dir = "data/processed/labels_yolo"
images_dir = "data/processed/images"  # boş txt için

# çok sınıflı YOLO dedeksiyonu için zorunludur.
# her kategori YOLO'da numerik sınıfa dönüşür.
CLASS_MAP = {
    "Mass": 0,
    "Suspicious Calcification": 1,
    "Asymmetry": 2,
    "Architectural Distortion": 3,
}
# annotation dosyasındaki liste formatındaki kategori fieldını çözer
# alt tipler tek sınıfta toplanır, listenin ilk elemanı kategori olarak alınır
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

def detect_study_id_column(df):
    for col in df.columns:
        if "study_id" in col.lower():
            return col
    raise ValueError("there is no column found about study_id in the subset csv!")

# YOLO bbox dönüşümü. yolo formatında hepsi [0,1] arasında olmalıdır. dönüşüm yapılır.
def convert_to_yolo(xmin, ymin, xmax, ymax, imgw, imgh):
    xc = (xmin + xmax) / 2.0 / imgw
    yc = (ymin + ymax) / 2.0 / imgh
    w = (xmax - xmin) / imgw
    h = (ymax - ymin) / imgh
    return xc, yc, w, h

def create_labels():
    print("Reading CSV files...")
    find_df = pd.read_csv(finding_path)
    subset_df = pd.read_csv(subset_path)

    print(f"Subset row count = {len(subset_df)}")

    study_col = detect_study_id_column(subset_df)
    print(f"Detected study_id column → {study_col}")

    # lesion olmayan category'ler elenir sadece dört main category kalır
    find_df["category"] = find_df["finding_categories"].apply(extract_category)
    valid_categories = list(CLASS_MAP.keys())
    lesion_df = find_df[find_df["category"].isin(valid_categories)]

    # subset içinde olan görüntülerden sadece anotasyon olanları alınır
    merged = lesion_df.merge(
        subset_df[[study_col, "image_id"]],
        left_on=["study_id", "image_id"],
        right_on=[study_col, "image_id"],
        how="inner"
    )

    print(f"Matched lesion annotations: {len(merged)}")

    # yolo dosyaları yazılır
    labels_root = Path(labels_dir)
    labels_root.mkdir(parents=True, exist_ok=True)

    count_label = 0

    # her görüntü için bbox yazılır, bir görüntüde birden fazla lezyon olabilir o yüzden groupby kullanılır
    # her boundingbox dönüştürülür ve dosyaya yazılır
    for (study_id, image_id), group in merged.groupby(["study_id", "image_id"]):
        out_file = labels_root / f"{image_id}.txt"
        lines = []

        for _, row in group.iterrows():
            cls = CLASS_MAP[row["category"]]

            xmin, ymin = row["xmin"], row["ymin"]
            xmax, ymax = row["xmax"], row["ymax"]
            imgw, imgh = row["width"], row["height"]

            xc, yc, w, h = convert_to_yolo(xmin, ymin, xmax, ymax, imgw, imgh)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        with open(out_file, "w") as f:
            f.write("\n".join(lines))

        count_label += 1

    print(f"Label files created (lesion): {count_label}")
    # yolo için her png'nin bir txt karşılığı olmalıdır. lezyonsuzlar için boş txt oluşturulur.
    png_ids = {p.stem for p in Path(images_dir).glob("*.png")}
    txt_ids = {p.stem for p in labels_root.glob("*.txt")}

    no_label_ids = png_ids - txt_ids

    for img_id in no_label_ids:
        open(labels_root / f"{img_id}.txt", "w").close()

    print(f"Empty label files created (no-lesion): {len(no_label_ids)}")
    print(f"Total label files: {len(list(labels_root.glob('*.txt')))}")

if __name__ == "__main__":
    create_labels()
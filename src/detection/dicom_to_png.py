import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image

def dicom_to_png(subset_csv_path = "data/processed/subset_samples_150.csv", dicom_root = "data/raw/dicoms", out_dir = "data/processed/images"):
 subset_csv_path = Path(subset_csv_path)
 dicom_root = Path(dicom_root)
 out_dir = Path(out_dir)

 print(f"Reading subset csv from: {subset_csv_path}")
 subset_df = pd.read_csv(subset_csv_path)

# csv'den benzersiz study_id ve image_id çiftleri extract edilir.
 df_pairs = subset_df[["study_id_x", "image_id"]].drop_duplicates()
 print(f"Number of unique pairs: {len(df_pairs)}")

 out_dir.mkdir(parents=True, exist_ok=True)
 num_ok = 0
 num_fail = 0

# dicom işleme döngüsü
 for _, row in df_pairs.iterrows():
  study_id = str(row["study_id_x"])
  image_id = str(row["image_id"])

  dicom_path = dicom_root / study_id / f"{image_id}.dicom"
  png_path = out_dir / f"{image_id}.png"

 # dosya yoksa hata sayısı arttırır. herhangi bir dosyada ya da dosya yolunda hata oluşursa fark etmek için gerekli.
  if not dicom_path.exists():
   print(f"DICOM not found: {dicom_path}")
   num_fail += 1
   continue

  try:
   # dicom okuma, raw pixel matrisleri yüklenir.
   ds = pydicom.dcmread(dicom_path)
   pixel_array = ds.pixel_array

   try:
    # tıbbi görüntülerde pixel değerleri doğrudan görünür değildir.
    # dicom görüntüsünün gerçek radyodensite değerinin görünür hale gelmesini sağlar.
    pixel_array = apply_voi_lut(pixel_array, ds)
   except Exception:
    pass
   
   img = pixel_array.astype(np.float32) 

   # mamografi görüntülerinin bir kısmı ters gelmektedir, ters görüntüleri düzeltir.
   photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
   if photometric == "MONOCHROME1":
    img = np.max(img) - img
   
   # normalizasyon. kontrast eşitlenir, model her görüntüyü benzer dinamik aralıkta görür
   img_min = np.min(img)
   img_max = np.max(img)
    
   if img_max > img_min:
    img = (img - img_min) / (img_max - img_min)
   else:
    img = img - img_min

   # YOLO modeli 8-bit PNG ister. dönüşüm yapılır.
   img = (img * 255.0).clip(0, 255).astype(np.uint8)

   # png olarak kaydetme
   pil_img = Image.fromarray(img)
   pil_img.save(png_path)
   num_ok += 1
  
  except Exception as e:
   print(f"Failied the process {dicom_path}: {e}")
   num_fail += 1
 
 print(f"Done. Succesfully converted: {num_ok}\nFailed/missing: {num_fail}")
 print(f"Output PNG folder: {out_dir}")

if __name__ == "__main__":
 dicom_to_png()

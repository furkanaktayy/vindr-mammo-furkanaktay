import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.transforms import (
    Compose, EnsureChannelFirstd, ScaleIntensityd,
    ToTensord, Resized, SpatialPadd, MapTransform
)
from monai.data import Dataset

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# =======================
# 1. Custom DICOM Loader
# =======================
class LoadDicomVOI(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            ds = pydicom.dcmread(d[k])
            img = ds.pixel_array.astype(np.float32)
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            try:
                img = apply_voi_lut(img, ds).astype(np.float32)
            except Exception:
                pass
            d[k] = img
        return d

# =======================
# 2. RADIMAGENET TERCÜMANI (Key Mapping)
# =======================
def load_medical_pretrained_fixed(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f" HATA: {checkpoint_path} bulunamadı!")
        return False

    print(f"\n" + "="*45)
    print(f" RADIMAGENET ANALİZ EDİLİYOR: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # RadImageNet'in sayısal kodlarını standart isimlere çeviren sözlük
    mapping = {
        "backbone.0.": "conv1.",
        "backbone.1.": "bn1.",
        "backbone.4.": "layer1.",
        "backbone.5.": "layer2.",
        "backbone.6.": "layer3.",
        "backbone.7.": "layer4.",
    }

    model_dict = model.state_dict()
    final_dict = {}
    load_count = 0

    for k, v in state_dict.items():
        # 1. Anahtar ismini (Key) tercüme et
        mapped_key = k
        for old, new in mapping.items():
            if k.startswith(old):
                mapped_key = k.replace(old, new, 1)
                break
        
        # 'module.' gibi prefixleri temizle
        mapped_key = mapped_key.replace("module.", "").replace("encoder.", "")

        # 2. Özel Durum: Conv1 (Grayscale Dönüşümü)
        if mapped_key == "conv1.weight":
            print("Conv1 bulundu, grayscale'e dönüştürülüyor...")
            if v.shape[1] == 3:
                v = v.mean(dim=1, keepdim=True)
            if model.conv1.weight.shape == v.shape:
                model.conv1.weight.data.copy_(v)
                load_count += 1
                continue

        # 3. Geri Kalan Katmanları Eşleştir (FC hariç)
        if mapped_key in model_dict and "fc" not in mapped_key:
            if model_dict[mapped_key].shape == v.shape:
                final_dict[mapped_key] = v
                load_count += 1
            else:
                print(f"⚠️ Boyut hatası (atlandı): {mapped_key}")

    # Ağırlıkları modele enjekte et
    model.load_state_dict(final_dict, strict=False)
    
    print(f"RadImageNet'ten {load_count} katman yüklendi!")
    print("="*45 + "\n")
    return True

# =======================
# 3. Main Execution Block
# =======================
def main():
    # --- Config ---
    CSV_PATH = r"C:\Users\Furkan Aktay\Desktop\mammo_dataset\density_train.csv"
    MEDICAL_WEIGHTS_PATH = r"C:\Users\Furkan Aktay\Desktop\CBIS-DDSM-breast-density\ResNet50.pt"
    SAVE_PATH = "best_medical_pretrained_model.pth"
    
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 8 
    EPOCHS = 15
    LR = 1e-4
    NUM_CLASSES = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42

    # --- Data Prep ---
    df = pd.read_csv(CSV_PATH)
    df["label"] = pd.to_numeric(df["label"], errors='coerce') - 1
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

    train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df["label"], random_state=RANDOM_SEED)

    def to_files(dframe):
        return [{"image": r.image_path, "label": r.label} for _, r in dframe.iterrows()]

    train_files, val_files, test_files = to_files(train_df), to_files(val_df), to_files(test_df)

    tf = Compose([
        LoadDicomVOI(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=512, size_mode="longest"),
        SpatialPadd(keys=["image"], spatial_size=IMG_SIZE, mode="constant"),
        ToTensord(keys=["image", "label"]),
    ])

    train_loader = DataLoader(Dataset(train_files, tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(Dataset(val_files, tf),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(Dataset(test_files, tf),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model Definition ---
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # DÜZELTİLMİŞ YÜKLEME FONKSİYONU
    success = load_medical_pretrained_fixed(model, MEDICAL_WEIGHTS_PATH)
    
    if not success:
        print("Medical weights error! Falling back to ImageNet...")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler("cuda")

    # --- Training ---
    print(f"Starting Training Phase on {DEVICE}...")
    best_val_auc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            x, y = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        y_true_v, y_probs_v = [], []
        with torch.no_grad():
            for batch in val_loader:
                img, lbl = batch["image"].to(DEVICE), batch["label"]
                with torch.amp.autocast("cuda"):
                    out = model(img).softmax(dim=1).cpu().numpy()
                y_true_v.extend(lbl.numpy())
                y_probs_v.extend(out)

        y_true_v = np.array(y_true_v)
        y_probs_v = np.array(y_probs_v).astype("float64")
        y_probs_v = y_probs_v / y_probs_v.sum(axis=1, keepdims=True)
        y_pred_v = np.argmax(y_probs_v, axis=1)

        v_auc = roc_auc_score(y_true_v, y_probs_v, multi_class='ovr', average='macro')
        v_acc = accuracy_score(y_true_v, y_pred_v)
        
        print(f"Epoch {epoch+1} Results | Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {v_acc:.4f} | Val AUC: {v_auc:.4f}")

        if v_auc > best_val_auc:
            best_val_auc = v_auc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  --> Best model updated (AUC: {v_auc:.4f})")

    # --- Final Test ---
    print("\nStarting Final Test Phase...")
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    y_true_t, y_probs_t = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            img, lbl = batch["image"].to(DEVICE), batch["label"]
            with torch.amp.autocast("cuda"):
                out = model(img).softmax(dim=1).cpu().numpy()
            y_true_t.extend(lbl.numpy())
            y_probs_t.extend(out)

    y_true_t = np.array(y_true_t)
    y_probs_t = np.array(y_probs_t).astype("float64")
    y_probs_t = y_probs_t / y_probs_t.sum(axis=1, keepdims=True)
    y_preds_t = np.argmax(y_probs_t, axis=1)

    print("\n" + "="*45)
    print(f"FINAL TEST RESULTS (Medical/RadImageNet Verified)")
    print("-" * 45)
    print(f"Accuracy   : {accuracy_score(y_true_t, y_preds_t):.4f}")
    print(f"Macro F1   : {f1_score(y_true_t, y_preds_t, average='macro'):.4f}")
    print(f"Overall AUC: {roc_auc_score(y_true_t, y_probs_t, multi_class='ovr'):.4f}")
    print("-" * 45)
    
    class_names = ["Density A", "Density B", "Density C", "Density D"]
    for i in range(NUM_CLASSES):
        auc_i = roc_auc_score((y_true_t == i).astype(int), y_probs_t[:, i])
        print(f"{class_names[i]:<12} AUC: {auc_i:.4f}")
    print("="*45)

if __name__ == "__main__":
    main()
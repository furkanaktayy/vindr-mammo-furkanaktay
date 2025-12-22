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
# 2. Main Execution Block
# =======================
def main():
    # --- Config ---
    CSV_PATH = r"C:\Users\Furkan Aktay\Desktop\mammo_dataset\density_train.csv"
    MODEL_PATH = "resnet50_density_standard_orientation.pth"
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 8 
    EPOCHS = 15
    LR = 1e-4
    NUM_CLASSES = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42

    # --- Data Prep & Balancing ---
    df = pd.read_csv(CSV_PATH)
    df["label"] = pd.to_numeric(df["label"], errors='coerce') - 1
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Handling class imbalance with weights
    weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

    train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df["label"], random_state=RANDOM_SEED)

    def to_files(dframe):
        return [{"image": r.image_path, "label": r.label} for _, r in dframe.iterrows()]

    train_files = to_files(train_df)
    val_files = to_files(val_df)
    test_files = to_files(test_df)

    # --- Information Header ---
    print("\n" + "="*40)
    print("Data Loaded Successfully:")
    print(f" - Training samples:   {len(train_files)}")
    print(f" - Validation samples: {len(val_files)}")
    print(f" - Testing samples:    {len(test_files)}")
    print("-" * 40)
    print("Class Distribution (Train):")
    print(train_df['label'].value_counts().sort_index().rename(index={0:'A', 1:'B', 2:'C', 3:'D'}))
    print("="*40 + "\n")

    # --- Transforms ---
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

    # --- Model, Optimizer, Scheduler ---
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler() 

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
            with torch.cuda.amp.autocast():
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
                out = model(img).softmax(dim=1).cpu().numpy()
                y_true_v.extend(lbl.numpy())
                y_probs_v.extend(out)

        v_auc = roc_auc_score(y_true_v, y_probs_v, multi_class='ovr', average='macro')
        v_acc = accuracy_score(y_true_v, np.argmax(y_probs_v, axis=1))
        print(f"Epoch {epoch+1} Results | Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {v_acc:.4f} | Val AUC: {v_auc:.4f}")

        if v_auc > best_val_auc:
            best_val_auc = v_auc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  --> Best model updated (AUC: {v_auc:.4f})")

    # --- Final Evaluation ---
    print("\nStarting Final Test Phase...")
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    
    model.eval()
    y_true_t, y_probs_t = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            img, lbl = batch["image"].to(DEVICE), batch["label"]
            out = model(img).softmax(dim=1).cpu().numpy()
            y_true_t.extend(lbl.numpy())
            y_probs_t.extend(out)

    y_true_t = np.array(y_true_t)
    y_probs_t = np.array(y_probs_t)
    y_preds_t = np.argmax(y_probs_t, axis=1)

    print("\n" + "="*45)
    print("FINAL TEST RESULTS")
    print("-" * 45)
    print(f"Test Samples    : {len(y_true_t)}")
    print(f"Final Accuracy  : {accuracy_score(y_true_t, y_preds_t):.4f}")
    print(f"Macro F1-Score  : {f1_score(y_true_t, y_preds_t, average='macro'):.4f}")
    print(f"Overall AUC     : {roc_auc_score(y_true_t, y_probs_t, multi_class='ovr'):.4f}")
    print("-" * 45)
    
    # Per-Class AUC Reporting
    class_names = ["Density A", "Density B", "Density C", "Density D"]
    for i in range(NUM_CLASSES):
        class_auc = roc_auc_score((y_true_t == i).astype(int), y_probs_t[:, i])
        print(f"{class_names[i]:<15} | AUC: {class_auc:.4f}")
    print("="*45)

    # Confusion Matrix
    cm = confusion_matrix(y_true_t, y_preds_t)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["A", "B", "C", "D"], yticklabels=["A", "B", "C", "D"])
    plt.title("Confusion Matrix (Standard Orientation)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    main()
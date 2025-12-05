# Breast Lesion Detection Pipeline with YOLO on VinDr-Mammo Dataset -- Furkan Aktay

This project presents an end-to-end deep learning pipeline for the automatic detection of **breast lesions** in selected mammography images from the **VinDr-Mammo** dataset. The project includes:

- DICOM to png conversion
- Data subset selection
- Conversion of labels to YOLO format
- Data splitting (train/val/test)
- YOLO model training with YOLOv8 (YOLOv12 compatible)
- Analysis of training outputs

All processes, except data downloading, are carried out fully automatically.

## Aim

The aim is to enable the detection of lesions such as:

- **Mass**
- **Suspicious Calcification**
- **Asymmetry**
- **Architectural Distortion**

Using YOLO-based models in mammography images used in breast cancer screenings, and to examine the performance of object detection models.

## Project Folder Structure

```plaintext
vindr-mammo-furkanaktay/
│
├── data/
│   ├── raw/                  # original VinDr-Mammo CSV and DICOM files
│   ├── processed/
│   │   ├── subset_samples.csv
│   │   ├── subset_samples_150.csv
│   │   ├── images/           # PNG outputs
│   │   ├── labels_yolo/      # YOLO .txt label files
│   │   └── yolo/             # Train/val/test dataset + data.yaml
│
├── src/
│   ├── select_samples.py
│   ├── expand_subset.py
│   ├── dicom_to_png.py
│   ├── create_yolo_label.py
│   └── prepare_yolo_dataset.py
│
├── report/
│   └── figures/              # confusion matrix, PR curve, batch figures etc.
│
├── requirements.txt
└── README.md
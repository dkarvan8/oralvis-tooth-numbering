# OralVis Tooth Numbering (YOLOv8)

Automatic detection and **FDI tooth numbering (11–48)** in oral images using Ultralytics YOLOv8, with an **optional anatomical post‑processing** step that improves ID consistency.

---

## 📁 Project Structure (recommended)

```
ORALVIS/
├─ .gitignore
├─ README.md
├─ data.yaml
├─ requirements.txt
├─ scripts/
│  ├─ postprocess_fdi_v4.py
│  ├─ eval_id_accuracy.py
│      
├─ results/
│  ├─ training/
│  │  └─ results.png
│  ├─ eval/
│  │  ├─ val_confusion_matrix.png
│  │  ├─ test_confusion_matrix.png
│  │  ├─ val_metrics.txt
│  │  └─ test_metrics.txt
│  ├─ predictions_raw/               (3–10 images)
│  ├─ predictions_post/              (same 3–10 images)                 
│  └─ weights/
│     └─ best.pt                     (trained weights)
└─ venv/                             (ignored)


> **Note:** Paths above reflect what’s created by the commands below. Your actual run names may vary (e.g., `yv8s_fdi640`).

---

## 🧰 Environment Setup

**Python**: 3.10 recommended

### Create and activate venv (Windows)
```bat
python -m venv venv
venv\Scripts\activate
```

### Install Ultralytics + common deps
```bat
pip install ultralytics opencv-python pyyaml
```

### (GPU) Install CUDA-enabled PyTorch
Find your CUDA version with `nvidia-smi`. Then install the matching wheel:

- **CUDA 12.1**
  ```bat
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **CUDA 11.8**
  ```bat
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

Verify:
```bat
python - <<PY
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

---

## 🗂️ Dataset

Organize into YOLO format:

```
oralvis_dataset/
  images/{train,val,test}/...
  labels/{train,val,test}/...   # .txt per image, YOLO (cls xc yc w h) normalized
```

Your `data.yaml` should point to these splits and list the **32 FDI classes in order**. Example keys:

```yaml
path: /absolute/path/to/oralvis_dataset
train: images/train
val: images/val
test: images/test
names:
  0: Canine (13)
  1: Canine (23)
  ...
  31: Third Molar (48)
```

---

## 🏋️ Train

**Windows CMD (single line):**
```bat
yolo train data=data.yaml model=yolov8s.pt imgsz=640 epochs=100 batch=16 project=runs_oralvis name=yv8s_fdi640 plots=True
```

Artifacts:
- Training curves → `runs_oralvis\detect\yv8s_fdi640\results.png`
- Best weights → `runs_oralvis\detect\yv8s_fdi640\weights\best.pt`
- Logs → `runs_oralvis\detect\yv8s_fdi640\results.csv` (if present)

> If you trained earlier with a different name, adjust paths accordingly.

---

## ✅ Evaluate (Val/Test)

**Validation:**
```bat
yolo val data=data.yaml model=runs_oralvis/detect/yv8s_fdi640/weights/best.pt split=val imgsz=640 project=runs_oralvis name=yv8s_val > val_metrics.txt 2>&1
```

**Test:**
```bat
yolo val data=data.yaml model=runs_oralvis/detect/yv8s_fdi640/weights/best.pt split=test imgsz=640 project=runs_oralvis name=yv8s_test > test_metrics.txt 2>&1
```

Look for **Precision, Recall, mAP@50, mAP@50–95** in the console logs (captured to the `.txt` files above). Ultralytics also writes `predictions.json`, and may produce `confusion_matrix.png` in each run folder.

---

## 🔎 Predict (save images + YOLO TXT labels)

```bat
yolo predict model=runs_oralvis/detect/yv8s_fdi640/weights/best.pt ^
  source=oralvis_dataset/images/test imgsz=640 conf=0.25 save=True save_txt=True save_conf=True ^
  project=runs_oralvis name=yv8s_preds
```

Outputs:
- Annotated images → `runs_oralvis\detect\yv8s_preds\*.jpg`
- Prediction labels → `runs_oralvis\detect\yv8s_preds\labels\*.txt`

---

## 🧭 Optional: Anatomical Post‑Processing (applied)

We include a conservative “snapper” that fixes obvious quadrant/order errors while **preserving correct YOLO IDs**. It auto‑reads the FDI mapping from `data.yaml`.

Run (try both viewer assumptions and keep the better result):
```bat
python scripts\postprocess_fdi_v4.py --pred_dir "runs_oralvis\detect\yv8s_preds" --out_dir "runs_oralvis\detect\yv8s_post_v4_rad" --data_yaml data.yaml --viewer radiograph --draw
python scripts\postprocess_fdi_v4.py --pred_dir "runs_oralvis\detect\yv8s_preds" --out_dir "runs_oralvis\detect\yv8s_post_v4_front" --data_yaml data.yaml --viewer frontal  --draw
```

### Compare ID‑accuracy (IoU ≥ 0.5) before vs after
Edit paths inside `scripts\eval_id_accuracy.py` to point to:
```
PRED_BEFORE = runs_oralvis\detect\yv8s_preds\labels
PRED_AFTER  = runs_oralvis\detect\yv8s_post_v4_<rad_or_front>\labels
```
Then run:
```bat
python scripts\eval_id_accuracy.py
```

**Observed on this project:**  
Baseline **87.91% (1302/1481)** → Post‑processed **90.61% (1342/1481)**  
= **+2.70 pp** (~**3.1%** relative), ~**22%** error reduction.

(Optional) build collages for the report:
```bat
python scripts\compare_before_after.py
```

---

## 📊 What to Put in the Report

- **Training curves:** `runs_oralvis\detect\yv8s_fdi640\results.png`
- **Metrics (Val/Test):** Precision, Recall, mAP@50, mAP@50–95 (from `val_metrics.txt` / `test_metrics.txt`)
- **Confusion matrix:** from `yv8s_val` / `yv8s_test` folders
- **Sample predictions:** ≥3 images from `yv8s_preds` (and matching post‑processed images if you include ablation)
- **Short summary:** data split, model config, and the above post‑processing outcome

---

## 💾 Weights & Large Files

If `weights/best.pt` exceeds GitHub’s 100 MB limit, upload to Drive and place a link in your README. Otherwise, commit it under `weights/` or attach it as a GitHub Release asset.

---

## 🛠️ Troubleshooting

- **Windows line continuation:** use `^` (CMD) or backtick `` ` `` (PowerShell). Avoid trailing `\`—that’s for Linux/macOS.
- **Empty `val_metrics.txt`:** capture stderr too: `> val_metrics.txt 2>&1`.
- **No prediction labels:** re‑run `yolo predict` with `save_txt=True` and lower `conf` (e.g., `0.15`).
- **GPU OOM:** reduce `batch` (e.g., `batch=8` or `4`).
- **Mapping mismatch:** `postprocess_fdi_v4.py` reads FDI directly from `data.yaml`—keep names consistent (e.g., “Central Incisor (11)”).

---

## 📦 Requirements (minimal)

```txt
ultralytics
opencv-python
pyyaml
torch            # GPU build recommended (see CUDA install above)
torchvision
torchaudio
```

Freeze your exact versions for reproducibility:
```bat
pip freeze > requirements.txt
```

---

## 🔖 Citation / Acknowledgements

- Detector: **Ultralytics YOLOv8** (https://github.com/ultralytics/ultralytics)  
- Dataset: Provided OralVis task dataset (FDI tooth numbering)

---

## 📫 Contact

For questions or reproducibility issues, please open an issue or reach out to the maintainer of this repository.

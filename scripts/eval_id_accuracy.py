import os, math
from pathlib import Path

GT_IMG_DIR   = Path(r"oralvis_dataset\images\test")
GT_LBL_DIR   = Path(r"oralvis_dataset\labels\test")
PRED_BEFORE  = Path(r"runs_oralvis\yv8s_preds\labels")
PRED_AFTER   = Path(r"runs_oralvis\detect\yv8s_post_v4_rad\labels")
IOU_TH = 0.5

def read_yolo(lbl_path):
    objs=[]
    if not lbl_path.exists(): return objs
    for line in lbl_path.read_text().strip().splitlines():
        p=line.strip().split()
        if len(p)<5: continue
        cls=int(float(p[0])); xc, yc, w, h = map(float, p[1:5])
        objs.append((cls, xc, yc, w, h))
    return objs

def iou(a,b):
    # a,b: (xc,yc,w,h) in normalized coords; assume same image size
    ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2
    ax2, ay2 = a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2
    bx2, by2 = b[0]+b[2]/2, b[1]+b[3]/2
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua>0 else 0.0

def match_and_score(gt, pr):
    used=set(); correct=0; total_matches=0
    for gcls, gxc, gyc, gw, gh in gt:
        best_iou, best_j = 0.0, -1
        for j,(pcls, pxc, pyc, pw, ph) in enumerate(pr):
            if j in used: continue
            i = iou((gxc,gyc,gw,gh),(pxc,pyc,pw,ph))
            if i>best_iou:
                best_iou, best_j = i, j
        if best_iou>=IOU_TH and best_j>=0:
            total_matches += 1
            used.add(best_j)
            if pr[best_j][0]==gcls:
                correct += 1
    return correct, total_matches

def evaluate(pred_dir):
    total_correct=0; total_matched=0; imgs=0
    for img_path in GT_IMG_DIR.glob("*.*"):
        stem = img_path.stem
        gt = read_yolo(GT_LBL_DIR / f"{stem}.txt")
        pr = read_yolo(pred_dir / f"{stem}.txt")
        if not gt or not pr:
            imgs += 1
            continue
        c,m = match_and_score(gt, pr)
        total_correct += c
        total_matched += m
        imgs += 1
    acc = (total_correct/total_matched)*100 if total_matched>0 else 0.0
    return acc, total_correct, total_matched, imgs

for tag,dirp in [("before", PRED_BEFORE), ("after", PRED_AFTER)]:
    acc,c,m,n = evaluate(dirp)
    print(f"{tag.upper()}  ID-Accuracy on matched boxes (IoU≥{IOU_TH}): {acc:.2f}%  ({c}/{m}) over {n} images")


# postprocess_fdi_v4.py
# Conservative snapping: fixes only obvious quadrant errors, keeps good IDs.
# Usage examples (Windows CMD):
#   python postprocess_fdi_v4.py --pred_dir "runs_oralvis\\detect\\yv8s_preds" --out_dir "runs_oralvis\\detect\\yv8s_post_v4_rad" --data_yaml data.yaml --viewer radiograph --draw
#   python postprocess_fdi_v4.py --pred_dir "runs_oralvis\\detect\\yv8s_preds" --out_dir "runs_oralvis\\detect\\yv8s_post_v4_front" --data_yaml data.yaml --viewer frontal  --draw

import argparse, re, json
from pathlib import Path

def load_mapping_from_yaml(yaml_path):
    import yaml
    y = yaml.safe_load(open(yaml_path, "r"))
    names = y.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys(), key=int)]
    if not isinstance(names, list):
        raise ValueError("Could not parse 'names' from data.yaml")
    idx_to_fdi, fdi_to_idx = {}, {}
    for i, name in enumerate(names):
        m = re.search(r"\((\d{2})\)", str(name))
        if not m:
            raise ValueError(f"Class name '{name}' lacks FDI like '(11)'.")
        fdi = int(m.group(1))
        idx_to_fdi[i] = fdi
        fdi_to_idx[fdi] = i
    return idx_to_fdi, fdi_to_idx

def kmeans_1d(values, iters=10):
    c1, c2 = min(values), max(values)
    for _ in range(iters):
        g1 = [v for v in values if abs(v-c1) <= abs(v-c2)]
        g2 = [v for v in values if abs(v-c1) >  abs(v-c2)]
        if g1: c1 = sum(g1)/len(g1)
        if g2: c2 = sum(g2)/len(g2)
    assign = [0 if abs(v-c1) <= abs(v-c2) else 1 for v in values]
    return (c1, c2), assign

def quadrant_for(viewer, arch, x):
    left = (x < 0.5)
    if viewer == "radiograph":   # image-left == patient-right
        if arch == "upper":  return 1 if left else 2
        else:                 return 4 if left else 3
    else:                          # frontal (facing patient)
        if arch == "upper":  return 2 if left else 1
        else:                 return 3 if left else 4

def sort_key_from_mid(viewer, q, x, mid_x):
    # central incisor first: sort by distance from midline outward
    # direction (left/right) differs by quadrant + viewer
    # radiograph: Q1/Q4 grow to left (use mid_x - x), Q2/Q3 grow to right (use x - mid_x)
    if viewer == "radiograph":
        if q in (1,4): return abs(mid_x - x) if x <= mid_x else 1e9 + (x - mid_x)
        else:          return abs(mid_x - x) if x >= mid_x else 1e9 + (mid_x - x)
    else:
        if q in (2,3): return abs(mid_x - x) if x <= mid_x else 1e9 + (x - mid_x)
        else:          return abs(mid_x - x) if x >= mid_x else 1e9 + (mid_x - x)

def second_digit(fdi): return fdi % 10
def quadrant_of_fdi(fdi): return fdi // 10

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_yaml", required=True)
    ap.add_argument("--viewer", default="radiograph", choices=["radiograph","frontal"])
    ap.add_argument("--draw", action="store_true")
    args = ap.parse_args()

    idx_to_fdi, fdi_to_idx = load_mapping_from_yaml(args.data_yaml)

    pred_dir = Path(args.pred_dir)
    labels_dir = pred_dir / "labels"
    if not labels_dir.exists():
        print(f"[!] labels/ not found in {pred_dir}. Re-run predict with save_txt=True.")
        return

    out_dir = Path(args.out_dir)
    (out_dir/"labels").mkdir(parents=True, exist_ok=True)
    (out_dir/"images").mkdir(parents=True, exist_ok=True)
    (out_dir/"meta").mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except Exception:
        cv2 = None
        if args.draw: print("[!] OpenCV not installed; proceeding without --draw.")

    txts = sorted(labels_dir.glob("*.txt"))
    processed = 0

    for txt in txts:
        stem = txt.stem
        # find image next to predictions
        img_path = None
        for ext in (".jpg",".png",".jpeg",".bmp"):
            p = pred_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p; break

        preds = []
        for line in txt.read_text().strip().splitlines():
            p = line.strip().split()
            if len(p) < 5: continue
            cls = int(float(p[0])); xc, yc, w, h = map(float, p[1:5])
            conf = float(p[5]) if len(p) >= 6 else None
            preds.append({"cls":cls,"xc":xc,"yc":yc,"w":w,"h":h,"conf":conf})

        if not preds:
            (out_dir/"labels"/txt.name).write_text("")
            continue

        xs = [p["xc"] for p in preds]; ys = [p["yc"] for p in preds]
        (m1, m2), assign = kmeans_1d(ys)
        upper_idx = 0 if m1 < m2 else 1
        for i,a in enumerate(assign):
            preds[i]["arch"] = "upper" if a==upper_idx else "lower"

        mid_x = sorted(xs)[len(xs)//2]
        for i,p in enumerate(preds):
            preds[i]["quad"] = quadrant_for(args.viewer, p["arch"], p["xc"])
            preds[i]["orig_fdi"] = idx_to_fdi.get(p["cls"], None)

        # group indices by quadrant
        quads = {1:[],2:[],3:[],4:[]}
        for i,p in enumerate(preds):
            quads[p["quad"]].append(i)

        new_cls = [p["cls"] for p in preds]  # default: keep original

        for q in [1,2,3,4]:
            idxs = quads[q]
            if not idxs: continue
            # order teeth from midline outward (central first)
            order = sorted(idxs, key=lambda i: sort_key_from_mid(args.viewer, q, preds[i]["xc"], mid_x))
            n = len(order)
            expected = [q*10 + d for d in range(1, min(8,n)+1)]

            # Check how consistent originals already are
            orig = [preds[i]["orig_fdi"] for i in order]
            orig_quadr_ok = sum(1 for f in orig if f and quadrant_of_fdi(f)==q)
            orig_second = [second_digit(f) if f else None for f in orig]
            # monotonic check (allow equal or +1 steps)
            mono_ok = 0
            for a,b in zip(orig_second[:-1], orig_second[1:]):
                if a is None or b is None: continue
                if b >= a and (b - a) <= 2: mono_ok += 1
            # If already mostly good, keep original IDs
            if orig_quadr_ok >= int(0.7*n) and mono_ok >= int(0.6*max(0,n-1)):
                continue

            # Otherwise, conservatively fix only items with wrong quadrant OR wild jumps
            for k,i in enumerate(order):
                f0 = preds[i]["orig_fdi"]
                if f0 is None or quadrant_of_fdi(f0) != q:
                    # wrong side → assign expected
                    f_new = expected[min(k, len(expected)-1)]
                    if f_new in fdi_to_idx: new_cls[i] = fdi_to_idx[f_new]
                else:
                    # same quadrant: keep unless second digit is way off relative to position
                    d0 = second_digit(f0); d_exp = min(k+1, 8)
                    if abs(d0 - d_exp) >= 3:
                        f_new = q*10 + d_exp
                        if f_new in fdi_to_idx: new_cls[i] = fdi_to_idx[f_new]

        # write labels
        out_lines=[]
        for i,p in enumerate(preds):
            parts = [str(new_cls[i]), f"{p['xc']:.6f}", f"{p['yc']:.6f}", f"{p['w']:.6f}", f"{p['h']:.6f}"]
            if p["conf"] is not None: parts.append(f"{p['conf']:.4f}")
            out_lines.append(" ".join(parts))
        (out_dir/"labels"/txt.name).write_text("\n".join(out_lines))

        # optional draw
        if args.draw and img_path is not None and 'cv2' in globals() and cv2 is not None:
            img = cv2.imread(str(img_path))
            if img is not None:
                H,W = img.shape[:2]
                for i,p in enumerate(preds):
                    fdi = [k for k,v in fdi_to_idx.items() if v==new_cls[i]]
                    label = f"FDI {fdi[0]}" if fdi else f"cls {new_cls[i]}"
                    x = p['xc']*W; y=p['yc']*H; w=p['w']*W; h=p['h']*H
                    x1=int(x-w/2); y1=int(y-h/2); x2=int(x+w/2); y2=int(y+h/2)
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(img,label,(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                cv2.line(img,(int(mid_x*W),0),(int(mid_x*W),H),(255,0,0),1)
                cv2.imwrite(str((out_dir/"images"/f"{stem}.jpg")))
        processed += 1

        (out_dir/"meta"/f"{stem}.json").write_text(json.dumps({"viewer": args.viewer, "mid_x": float(mid_x)}, indent=2))

    print(f"Done. Processed {processed} images.")
    print(f"New labels: {out_dir/'labels'}")
    print(f"Drawn images (if --draw): {out_dir/'images'}")

if __name__ == "__main__":
    main()

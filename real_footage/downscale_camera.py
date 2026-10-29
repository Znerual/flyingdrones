#!/usr/bin/env python3
"""Downscale a prepared camera dir to a target width (default 1920), scaling the
label coordinates to match, so the flow-alignment metric stays correct. In the
paper protocol the two 4K cameras are scored at 1920 px width (prep_all.py runs
this for every camera; cameras at <= 1920 px width are left untouched).
"""
import argparse
import json
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camdir")
    ap.add_argument("--max-width", type=int, default=1920)
    a = ap.parse_args()
    cam = Path(a.camdir)
    meta = json.loads((cam / "meta.json").read_text())
    W, H = meta["W"], meta["H"]
    if W <= a.max_width:
        print(f"{cam.name}: already {W}x{H} <= {a.max_width}, skip"); return
    s = a.max_width / W
    nW, nH = int(round(W * s)), int(round(H * s))
    print(f"{cam.name}: {W}x{H} -> {nW}x{nH} (scale {s:.4f})")

    for p in sorted((cam / "images").glob("*.png")):
        img = cv2.imread(str(p))
        cv2.imwrite(str(p), cv2.resize(img, (nW, nH), interpolation=cv2.INTER_AREA))

    def scale_xy(xy):
        return [xy[0] * s, xy[1] * s]

    lab = json.loads((cam / "labels.json").read_text())
    (cam / "labels.json").write_text(json.dumps({k: scale_xy(v) for k, v in lab.items()}))

    import math
    MB = [("m1_subpx", 0, 1), ("m2_low", 1, 3), ("m3_mid", 3, 8), ("m4_high", 8, math.inf)]
    def mbin(m):
        for n, lo, hi in MB:
            if lo <= m < hi:
                return n
        return MB[-1][0]
    sel = json.loads((cam / "selected.json").read_text())
    for e in sel:
        e["label_ref"] = scale_xy(e["label_ref"])
        e["label_nxt"] = scale_xy(e["label_nxt"])
        d = math.hypot(e["label_nxt"][0] - e["label_ref"][0], e["label_nxt"][1] - e["label_ref"][1])
        e["gt_disp_mag"] = round(d, 4)
        e["motion_stratum"] = mbin(d)
    (cam / "selected.json").write_text(json.dumps(sel, indent=2))

    meta["W"], meta["H"] = nW, nH
    meta["downscaled_from"] = [W, H]
    (cam / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  done: {len(sel)} labels scaled, meta updated")


if __name__ == "__main__":
    main()

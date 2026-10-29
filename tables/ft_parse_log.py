"""Parse a train_nature3.py stdout log into a curve TSV + a one-line-per-field summary.

usage: python ft_parse_log.py <arm> <results_dir> [<arm for baseline rows>]
reads   <results_dir>/<arm>_train_log.txt
writes  <results_dir>/<arm>_train_curve.tsv    step  kind(train|val)  lr  train_epe  train_1px  val_epe  val_epe_drone  new_best
        <results_dir>/<arm>_train_summary.tsv  key<TAB>value   (never overwrites existing files)
Train lines: "[   100, lr=0.0000246] 1px=0.2961 epe=2.5154"  (running mean over the last 100 steps)
Val lines:   "[   150] nat3_val_epe=0.4321 nat3_val_epe_drone=2.5000", optionally followed by
             "  new best val_epe_drone=..."
Step-0 val (init ckpt on the val split) is taken from summary/<arm>__init__val.csv if present.
"""
import csv
import os
import re
import sys

arm, R = sys.argv[1], sys.argv[2]
log = os.path.join(R, f"{arm}_train_log.txt")
curve_p = os.path.join(R, f"{arm}_train_curve.tsv")
summ_p = os.path.join(R, f"{arm}_train_summary.tsv")
TR = re.compile(r"^\[\s*(\d+), lr=([0-9.eE+-]+)\]\s+(.*)$")
VA = re.compile(r"^\[\s*(\d+)\]\s+nat3_val_epe=([0-9.eE+-]+)\s+nat3_val_epe_drone=([0-9.eE+-]+)")

rows = []
for line in open(log, errors="replace"):
    line = line.rstrip("\n")
    m = TR.match(line)
    if m:
        kv = dict(t.split("=", 1) for t in m.group(3).split() if "=" in t)
        rows.append(dict(step=int(m.group(1)), kind="train", lr=m.group(2),
                         train_epe=kv.get("epe", ""), train_1px=kv.get("1px", "")))
        continue
    m = VA.match(line)
    if m:
        rows.append(dict(step=int(m.group(1)), kind="val", val_epe=m.group(2),
                         val_epe_drone=m.group(3), new_best="0"))
        continue
    if line.strip().startswith("new best") and rows and rows[-1]["kind"] == "val":
        rows[-1]["new_best"] = "1"

cols = ["step", "kind", "lr", "train_epe", "train_1px", "val_epe", "val_epe_drone", "new_best"]
tr = [r for r in rows if r["kind"] == "train"]
va = [r for r in rows if r["kind"] == "val"]
if not tr or not va:
    sys.exit(f"no train/val lines parsed from {log}")
best = min(va, key=lambda r: float(r["val_epe_drone"]))
# the trainer's own pick = last "new best" line (strict <, first minimum wins on ties)
picked = [r for r in va if r.get("new_best") == "1"][-1]
assert picked is best or float(picked["val_epe_drone"]) == float(best["val_epe_drone"])

s = [("arm", arm), ("log", log),
     ("train_first_step", tr[0]["step"]), ("train_epe_first", tr[0]["train_epe"]),
     ("train_last_step", tr[-1]["step"]), ("train_epe_last", tr[-1]["train_epe"]),
     ("val_first_step", va[0]["step"]), ("val_drone_first", va[0]["val_epe_drone"]),
     ("val_last_step", va[-1]["step"]), ("val_drone_last", va[-1]["val_epe_drone"]),
     ("best_step", picked["step"]), ("val_drone_best", picked["val_epe_drone"]),
     ("val_full_at_best", picked["val_epe"]), ("n_val", len(va)),
     ("val_drone_curve", " ".join(f"{r['step']}:{r['val_epe_drone']}" for r in va))]
base_arm = sys.argv[3] if len(sys.argv) > 3 else arm   # baseline rows are per arm, not per run tag
init_val = os.path.join(R, "summary", f"{base_arm}__init__val.csv")
if os.path.exists(init_val):
    with open(init_val) as fh:
        r0 = next(csv.DictReader(fh))
    s += [("val_drone_step0_init", r0["epe_drone"]), ("val_full_step0_init", r0["epe"])]

for p, write in ((curve_p, lambda fh: [fh.write("\t".join(cols) + "\n")] +
                  [fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n") for r in rows]),
                 (summ_p, lambda fh: [fh.write(f"{k}\t{v}\n") for k, v in s])):
    if os.path.exists(p):
        print(f"SKIP {p} exists")
        continue
    with open(p, "w") as fh:
        write(fh)
    print(f"wrote {p}")
for k, v in s:
    print(f"{k}\t{v}")

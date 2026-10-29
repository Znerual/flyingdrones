#!/usr/bin/env python3
"""
Stratified 80/10/10 train/val/test split of the 1000-pair FlyingDrones eval subset,
for fine-tuning experiments -- buffered variant.

Identical goal to build_ft_split.py, but additionally guarantees a >=5-frame gap
between any two selected frames that land in DIFFERENT splits. This is required
for genuine multi-frame (VideoFlow MOF, 5-frame window i-2..i+2) supervision: a
training sample's real neighbour frames must never be another split's held-out
anchor frame, or that split's raw pixel content leaks into training inputs.

Method: chain the 1000 selected idx (sorted) into blocks wherever consecutive
selected idx are <=4 apart (so any two different blocks are guaranteed >=5 apart).
Assign whole blocks -- not individual frames -- to train/val/test via the same
largest-remainder stratified procedure as build_ft_split.py, weighted by block
size, using each block's majority stratum for bucketing.

Outputs (in splits/, next to this script):
  nature3_ft_split.json                  full manifest (adds "blocks" section)
  nature3_ft_{train,val,test}_frames.txt  ref-frame ids per split
  nature3_ft_{train,val,test}_indices.txt idx per split
"""
import json, os, collections, math, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))               # splits/
SRC  = os.path.join(HERE, "nature3_split.json")
SEED = 42
FRAC = {"val": 0.10, "test": 0.10}
MIN_GAP = 5   # blocks split whenever consecutive selected idx differ by more than this-1


def lcg_shuffle(items, key, seed):
    return sorted(items, key=lambda e: hashlib.md5(f"{seed}:{key(e)}".encode()).hexdigest())


def build_blocks(selection):
    """Chain selected entries (sorted by idx) into blocks; consecutive members
    <= (MIN_GAP - 1) apart join the same block, guaranteeing >=MIN_GAP between
    blocks."""
    items = sorted(selection, key=lambda e: e["idx"])
    blocks = []
    cur = [items[0]]
    for e in items[1:]:
        if e["idx"] - cur[-1]["idx"] <= MIN_GAP - 1:
            cur.append(e)
        else:
            blocks.append(cur); cur = [e]
    blocks.append(cur)
    return blocks


def block_stratum(block):
    """Majority stratum among block members (ties broken by first-seen order)."""
    counts = collections.Counter(e["stratum"] for e in block)
    top = counts.most_common(1)[0][0]
    return top


def main():
    d = json.load(open(SRC))
    sel = d["selection"]
    blocks = build_blocks(sel)
    print(f"selected frames: {len(sel)}  ->  blocks: {len(blocks)}")

    # sanity: min inter-block gap
    starts = sorted(b[0]["idx"] for b in blocks)
    ends_by_start = {b[0]["idx"]: b[-1]["idx"] for b in blocks}
    sorted_blocks = sorted(blocks, key=lambda b: b[0]["idx"])
    min_gap = min(sorted_blocks[i + 1][0]["idx"] - sorted_blocks[i][-1]["idx"]
                  for i in range(len(sorted_blocks) - 1))
    assert min_gap >= MIN_GAP, f"block construction bug: min gap {min_gap} < {MIN_GAP}"
    print(f"min inter-block frame gap: {min_gap} (>= {MIN_GAP} required)")

    by_stratum = collections.defaultdict(list)
    for b in blocks:
        by_stratum[block_stratum(b)].append(b)

    assign = {}  # block-id (idx of first member) -> split
    frac_rem = {"val": [], "test": []}
    counts = collections.defaultdict(lambda: collections.Counter())  # per-stratum FRAME counts

    for strat, blist in by_stratum.items():
        blist = lcg_shuffle(blist, lambda b: b[0]["idx"], SEED)
        n_frames = sum(len(b) for b in blist)
        nv_frames = math.floor(FRAC["val"] * n_frames)
        nt_frames = math.floor(FRAC["test"] * n_frames)
        if n_frames >= 3 and len(blist) >= 2:
            nv_frames = max(nv_frames, 1); nt_frames = max(nt_frames, 1)

        # greedily fill test then val then train, whole blocks at a time,
        # picking each next block from whichever split most needs frames
        remaining = {"test": nt_frames, "val": nv_frames}
        order = []
        for b in blist:
            if remaining["test"] > 0:
                order.append((b, "test")); remaining["test"] -= len(b)
            elif remaining["val"] > 0:
                order.append((b, "val")); remaining["val"] -= len(b)
            else:
                order.append((b, "train"))
        for b, s in order:
            assign[b[0]["idx"]] = s
            counts[strat][s] += len(b)
        frac_rem["val"].append((FRAC["val"] * n_frames - math.floor(FRAC["val"] * n_frames), strat))
        frac_rem["test"].append((FRAC["test"] * n_frames - math.floor(FRAC["test"] * n_frames), strat))

    block_by_id = {b[0]["idx"]: b for b in blocks}

    def totals():
        c = collections.Counter()
        for bid, s in assign.items():
            c[s] += len(block_by_id[bid])
        return c

    target = {"val": 100, "test": 100}
    for split in ["test", "val"]:
        # top-up: move smallest train-block from the stratum with the largest
        # fractional remainder, repeat until within reach of target (whole
        # blocks can't hit the target exactly; get as close as possible)
        guard = 0
        while totals()[split] < target[split] and guard < 2000:
            guard += 1
            moved = False
            for _, strat in sorted(frac_rem[split], reverse=True):
                cand = [b for b in by_stratum[strat] if assign.get(b[0]["idx"]) == "train"]
                if cand:
                    cand.sort(key=len)  # move smallest block first, finer granularity
                    b = cand[0]
                    assign[b[0]["idx"]] = split
                    counts[strat][split] += len(b); counts[strat]["train"] -= len(b)
                    moved = True
                    break
            if not moved:
                break
        guard = 0
        while totals()[split] > target[split] and guard < 2000:
            guard += 1
            moved = False
            for _, strat in sorted(frac_rem[split]):
                cand = [b for b in by_stratum[strat]
                        if assign.get(b[0]["idx"]) == split and counts[strat][split] > len(b)]
                if cand:
                    cand.sort(key=len, reverse=True)
                    b = cand[0]
                    assign[b[0]["idx"]] = "train"
                    counts[strat][split] -= len(b); counts[strat]["train"] += len(b)
                    moved = True
                    break
            if not moved:
                break

    splits = {"train": [], "val": [], "test": []}
    for b in blocks:
        s = assign[b[0]["idx"]]
        splits[s].extend(b)
    for k in splits:
        splits[k].sort(key=lambda e: e["idx"])

    tot = totals()
    print("frame counts:", dict(tot))

    # final leakage sanity check: for every pair of frames in DIFFERENT splits,
    # gap must be >= MIN_GAP
    all_by_split = {k: sorted(e["idx"] for e in v) for k, v in splits.items()}
    violations = 0
    for s1 in ("train", "val", "test"):
        for s2 in ("train", "val", "test"):
            if s1 >= s2:
                continue
            for i in all_by_split[s1]:
                for j in all_by_split[s2]:
                    if abs(i - j) < MIN_GAP:
                        violations += 1
    print(f"cross-split gap violations (<{MIN_GAP} frames apart): {violations}")
    assert violations == 0, "leakage guarantee violated -- do not use this split"

    manifest = {
        "dataset": d["dataset"], "source_subset": os.path.basename(SRC),
        "native_resolution": d["native_resolution"], "seed": SEED,
        "split_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "split_type": f"block-buffered stratified (size x motion + empty), min gap {MIN_GAP}, largest-remainder over blocks",
        "num_blocks": len(blocks),
        "counts": {k: tot[k] for k in ["train", "val", "test"]},
        "per_stratum": {s: dict(counts[s]) for s in sorted(counts)},
        "splits": {k: [{"idx": e["idx"], "ref_frame": e["ref_frame"], "stratum": e["stratum"]}
                       for e in v] for k, v in splits.items()},
    }
    json.dump(manifest, open(os.path.join(HERE, "nature3_ft_split.json"), "w"), indent=2)
    for k, v in splits.items():
        with open(os.path.join(HERE, f"nature3_ft_{k}_frames.txt"), "w") as f:
            f.write("\n".join(e["ref_frame"] for e in v) + "\n")
        with open(os.path.join(HERE, f"nature3_ft_{k}_indices.txt"), "w") as f:
            f.write("\n".join(str(e["idx"]) for e in v) + "\n")

    allidx = [e["idx"] for v in splits.values() for e in v]
    assert len(allidx) == 1000 and len(set(allidx)) == 1000, "split not a clean partition"
    print("partition OK: 1000 unique, disjoint")


if __name__ == "__main__":
    main()

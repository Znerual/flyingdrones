"""Unit tests for the metrics, the resize conventions and the backward-flow pairing.

Run with ``pytest tests/`` from the repository root (needs numpy + Pillow; the
torch-equivalence tests are skipped when torch is not installed).
"""
import importlib.util
import math
import os
import os.path as osp
import sys

import numpy as np
import pytest
from PIL import Image

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO)

import flyingdrones as fd  # noqa: E402


def _load_common():
    """The single-file module shipped next to every evaluator."""
    p = osp.join(REPO, "evaluators", "_common", "fd_eval_common.py")
    spec = importlib.util.spec_from_file_location("fd_eval_common", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- metrics

def test_metrics_hand_computed():
    H, W = 4, 8
    gt = np.zeros((H, W, 2), np.float32)
    gt[..., 0] = 10.0                      # every pixel moves 10 px in x
    pred = gt.copy()
    pred[0, 0] = (10.0 + 3.0, 4.0)         # EPE 5 at one pixel (dx=3, dy=4)
    pred[1, 1] = (10.0, 1.0)               # EPE 1 exactly (not > 1)
    pred[2, 2] = (10.0 + 2.0, 0.0)         # EPE 2
    mask = np.zeros((H, W), bool)
    mask[0, 0] = mask[2, 2] = True         # drone = the EPE-5 and EPE-2 pixels
    m = fd.compute_metrics(pred, gt, mask)

    n = H * W
    assert m["epe_all"] == pytest.approx((5 + 1 + 2) / n)
    assert m["epe_drone"] == pytest.approx((5 + 2) / 2)
    assert m["epe_bg"] == pytest.approx(1 / (n - 2))
    # epe_norm scales u by W and v by H, x100
    en = [math.hypot(3 / W, 4 / H), math.hypot(0, 1 / H), math.hypot(2 / W, 0)]
    assert m["epe_norm_all"] == pytest.approx(100 * sum(en) / n)
    assert m["epe_norm_drone"] == pytest.approx(100 * (en[0] + en[2]) / 2)
    # thresholds are strict (>)
    assert m["px1_err_all"] == pytest.approx(100 * 2 / n)
    assert m["px3_err_all"] == pytest.approx(100 * 1 / n)
    assert m["px5_err_all"] == 0.0
    assert m["px1_err_drone"] == pytest.approx(100.0)
    # FL: EPE > 3 and EPE / |gt| > 5 %  -> only the EPE-5 pixel (5 / 10 = 50 %)
    assert m["fl_all"] == pytest.approx(100 * 1 / n)
    assert m["fl_drone"] == pytest.approx(50.0)
    assert m["n_drone_px"] == 2


def test_fl_relative_condition():
    gt = np.zeros((1, 2, 2), np.float32)
    gt[0, 0, 0] = 100.0                    # large motion: EPE 4 is only 4 % -> not an outlier
    gt[0, 1, 0] = 10.0                     # small motion: EPE 4 is 40 % -> outlier
    pred = gt.copy()
    pred[..., 0] += 4.0
    m = fd.compute_metrics(pred, gt, None)
    assert m["px3_err_all"] == 100.0
    assert m["fl_all"] == 50.0


def test_input_layouts_and_empty_drone_region():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(6, 9, 2)).astype(np.float32)
    pred = rng.normal(size=(6, 9, 2)).astype(np.float32)
    a = fd.compute_metrics(pred, gt, np.zeros((6, 9)))
    b = fd.compute_metrics(pred.transpose(2, 0, 1)[None], gt.transpose(2, 0, 1), None)
    for k in a:
        if isinstance(a[k], float) and math.isnan(a[k]):
            assert math.isnan(b[k])
        else:
            assert a[k] == pytest.approx(b[k])
    assert all(math.isnan(a[f"{k}_drone"]) for k in fd.METRIC_NAMES)
    assert a["epe_all"] == pytest.approx(a["epe_bg"])


def test_aggregate_skips_nan_and_is_frame_weighted():
    rows = [{f"{k}_{r}": float("nan") for k in fd.METRIC_NAMES for r in fd.REGIONS} for _ in range(3)]
    rows[0]["epe_drone"], rows[1]["epe_drone"] = 1.0, 3.0      # third pair has no drone
    rows[0]["epe_all"], rows[1]["epe_all"], rows[2]["epe_all"] = 1.0, 2.0, 6.0
    agg = fd.aggregate(rows)
    assert agg["epe_drone"] == pytest.approx(2.0)
    assert agg["epe_all"] == pytest.approx(3.0)
    assert math.isnan(agg["fl_bg"])


def test_bootstrap_ci():
    v = list(np.linspace(0, 1, 101)) + [float("nan")]
    mean, lo, hi = fd.bootstrap_ci(v, n_boot=2000, seed=0)
    assert mean == pytest.approx(0.5)
    assert lo < mean < hi
    assert fd.bootstrap_ci([1.0] * 10, n_boot=100) == (1.0, 1.0, 1.0)


def test_package_matches_shipped_evaluator_module():
    """flyingdrones.* and evaluators/_common/fd_eval_common.py must give identical numbers."""
    common = _load_common()
    rng = np.random.default_rng(1)
    for _ in range(5):
        gt = rng.normal(scale=4, size=(27, 48, 2)).astype(np.float32)
        pred = gt + rng.normal(scale=2, size=gt.shape).astype(np.float32)
        mask = rng.random((27, 48)) > 0.8
        a = fd.compute_metrics(pred, gt, mask)
        b = common.compute_metrics(pred, gt, mask)
        assert a == b
    rows = [fd.compute_metrics(rng.normal(size=(5, 7, 2)), rng.normal(size=(5, 7, 2)),
                               rng.random((5, 7)) > 0.5) for _ in range(20)]
    assert fd.aggregate(rows) == common.aggregate(rows)
    vals = [r["epe_all"] for r in rows]
    assert fd.bootstrap_ci(vals, n_boot=500) == common.bootstrap_ci(vals, n_boot=500)
    assert fd.TWO_FRAME == common.TWO_FRAME and fd.MOF_WINDOW == common.MOF_WINDOW
    assert fd.BOF_WINDOW == common.BOF_WINDOW and fd.MEMFLOW_CLIP == common.MEMFLOW_CLIP


# --------------------------------------------------------------------------- pairing convention

def _make_scene(root, n=6, size=(16, 24), shift=(2, 1)):
    """Tiny synthetic sequence obeying the dataset convention.

    The camera window slides by ``+shift`` per frame (scene content moves by
    ``-shift`` in the image), so the content at pixel x of frame t sits at
    ``x + shift`` in frame t-1: the backward flow stored in ``<t>_flow.npy`` is the
    constant ``shift`` = (u, v).
    """
    H, W = size
    rng = np.random.default_rng(0)
    u, v = shift
    big = (rng.random((H + v * n + 1, W + u * n + 1, 3)) * 255).astype(np.uint8)
    for d in ("images", "flow", "labels"):
        os.makedirs(osp.join(root, d))
    for t in range(n):
        y0, x0 = v * t, u * t                            # window origin of frame t
        Image.fromarray(big[y0:y0 + H, x0:x0 + W]).save(osp.join(root, "images", f"{t:04d}.png"))
        flow = np.zeros((H, W, 2), np.float32)
        flow[..., 0], flow[..., 1] = u, v
        np.save(osp.join(root, "flow", f"{t:04d}_flow.npy"), flow)
        lab = np.zeros((H, W), np.uint8)
        lab[3:6, 4:9] = 1 + t % 6
        Image.fromarray(lab).save(osp.join(root, "labels", f"{t:04d}_label.png"))
    return root


def test_backward_pairing(tmp_path):
    root = _make_scene(str(tmp_path / "Nature3"))
    samples = fd.list_samples(root)
    refs = [fd.ref_base(s) for s in samples]
    assert refs == ["0001", "0002", "0003", "0004", "0005"]      # 0000 has no t-1
    for s in samples:
        t = int(fd.ref_base(s))
        assert osp.basename(s[0]) == f"{t:04d}.png"                 # reference frame I_t first
        assert osp.basename(s[1]) == f"{t - 1:04d}.png"             # then I_{t-1}
        assert osp.basename(s[2]) == f"{t:04d}_flow.npy"
        assert osp.basename(s[3]) == f"{t:04d}_label.png"

    # the stored flow is backward: I_t(x) == I_{t-1}(x + F_t(x))
    smp = fd.load_sample(samples[2])
    I_t, I_tm1 = smp["frames"][0].astype(int), smp["frames"][1].astype(int)
    u, v = (int(round(c)) for c in smp["flow"][0, 0])
    H, W = I_t.shape[:2]
    ys, xs = np.mgrid[0:H - v, 0:W - u]
    assert np.array_equal(I_t[ys, xs], I_tm1[ys + v, xs + u])
    assert not np.array_equal(I_t[ys, xs], I_tm1[np.clip(ys - v, 0, None), np.clip(xs - u, 0, None)])

    # drone mask = label > 0 (any pass index 1..6)
    assert smp["mask"].sum() == 15 and smp["label"].max() == 4


def test_temporal_windows_are_time_reversed_and_clamped(tmp_path):
    root = _make_scene(str(tmp_path / "Nature3"))
    mof = fd.list_samples(root, fd.MOF_WINDOW)
    first = mof[0]                                   # t = 1: t-2 is clamped to frame 0
    assert [osp.basename(p) for p in first[:5]] == ["0003.png", "0002.png", "0001.png", "0000.png", "0000.png"]
    assert osp.basename(first[5]) == "0001_flow.npy"
    last = mof[-1]                                   # t = 5 (last frame): t+1, t+2 clamped
    assert [osp.basename(p) for p in last[:5]] == ["0005.png", "0005.png", "0005.png", "0004.png", "0003.png"]
    bof = fd.list_samples(root, fd.BOF_WINDOW)
    assert [osp.basename(p) for p in bof[1][:3]] == ["0003.png", "0002.png", "0001.png"]


def test_split_filter_uses_reference_frame(tmp_path):
    root = _make_scene(str(tmp_path / "Nature3"))
    split = tmp_path / "split.txt"
    split.write_text("0000\n0002\n0004\n")
    kept = fd.filter_samples(fd.list_samples(root, fd.MOF_WINDOW), fd.load_split_frames(str(split)))
    assert [fd.ref_base(s) for s in kept] == ["0002", "0004"]       # 0000 can never be evaluated
    assert len(fd.load_split_frames("stratified")) == 1000
    assert "0000" in fd.load_split_frames("stratified")
    assert fd.load_split_frames("window500") == {f"{i:04d}" for i in range(1, 501)}


# --------------------------------------------------------------------------- resize conventions

def test_read_flow_rescales_vectors(tmp_path):
    flow = np.zeros((1080, 1920, 2), np.float32)
    flow[..., 0], flow[..., 1] = 8.0, -4.0
    p = str(tmp_path / "0001_flow.npy")
    np.save(p, flow)
    for (H, W) in fd.RESOLUTIONS.values():
        f = fd.read_flow(p, (H, W))
        assert f.shape == (H, W, 2)
        assert np.allclose(f[..., 0], 8.0 * W / 1920) and np.allclose(f[..., 1], -4.0 * H / 1080)


@pytest.mark.parametrize("size", [(270, 480), (432, 960)])
def test_resize_matches_torch(size):
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    rng = np.random.default_rng(2)
    flow = rng.normal(scale=5, size=(1080, 1920, 2)).astype(np.float32)
    ref = F.interpolate(torch.from_numpy(flow).permute(2, 0, 1)[None], size=size,
                        mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    assert np.allclose(fd.resize_bilinear(flow, size), ref, atol=1e-4)
    lab = rng.integers(0, 7, size=(1080, 1920)).astype(np.uint8)
    refm = F.interpolate(torch.from_numpy((lab > 0).astype(np.float32))[None, None], size=size,
                         mode="nearest")[0, 0].numpy()
    assert np.array_equal(fd.resize_nearest(lab, size) > 0, refm > 0.5)

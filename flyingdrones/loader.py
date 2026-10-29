"""FlyingDrones dataset loader.

Dataset layout (one scene directory, e.g. ``<FD_DATA>/Nature3``)::

    Nature3/
        images/<t>.png          RGB frame t, 1920x1080, t = 0000 ... 9000
        flow/<t>_flow.npy       float32 (1080, 1920, 2), (u, v) in pixels
        labels/<t>_label.png    uint8 (1080, 1920); 0 = background,
                                1..6 = render pass index of the drone covering the pixel

Ground-truth convention (IMPORTANT)
-----------------------------------
``flow/<t>_flow.npy`` is the **backward** flow of frame t: for every pixel x of
frame t, ``x + F_t(x)`` is its position in frame t-1, i.e.

    I_t(x)  ~=  I_{t-1}(x + F_t(x))

A (forward) optical-flow estimator must therefore be run on the pair
``(I_t, I_{t-1})`` with ``I_t`` as the reference (first) frame, and its output is
compared directly with ``<t>_flow.npy`` (no sign flip). Frame 0000 has no
predecessor and is never a reference frame. Temporal models receive the
time-reversed window, e.g. ``[t+2, t+1, t, t-1, t-2]`` for VideoFlow MOF (see the
``*_WINDOW`` constants below).

The label of reference frame t (``<t>_label.png``) defines the drone region of the
pair; the drone mask used by every metric is ``label > 0`` (union of all drones).

Resolutions and resize conventions
----------------------------------
All three benchmark resolutions are produced on the fly from the native
1920x1080 files (``RESOLUTIONS``). The evaluators used for the paper do this as
follows (``size=(H, W)``):

* ground-truth flow: bilinear interpolation with ``align_corners=False`` and no
  antialiasing (``torch.nn.functional.interpolate`` semantics), then
  ``u *= W / 1920`` and ``v *= H / 1080``. This is a plain resize of the full frame;
  for 960x432 the aspect ratio changes (u scaled by 0.5, v by 0.4), no crop.
* drone mask: nearest neighbour (``torch`` legacy ``"nearest"`` semantics,
  ``src = floor(dst * in / out)``).
* images: depends on the upstream code base of each model (kept as in the paper
  runs, see ``IMAGE_RESIZE_BY_FAMILY``): PIL bicubic, PIL bilinear, or OpenCV
  ``INTER_AREA``. At native 1920x1080 no resize happens.

``read_flow`` / ``read_label`` reproduce the flow and mask conventions exactly in
NumPy (no torch needed); ``read_image`` supports the three image backends.
"""

import glob
import os
import os.path as osp

import numpy as np

NATIVE_SIZE = (1080, 1920)  # (H, W)

# Benchmark resolutions, (H, W). Names follow the paper.
RESOLUTIONS = {
    "small": (270, 480),       # Drones-Small, 480x270
    "standard": (432, 960),    # Drones-Standard, 960x432
    "large": (1080, 1920),     # Drones-Large, native 1920x1080
}

# Frame offsets relative to the reference frame t (offset 0 must be present).
TWO_FRAME = (0, -1)                  # (I_t, I_{t-1})
BOF_WINDOW = (1, 0, -1)              # VideoFlow BOF, read forward slot 0  (t -> t-1)
MOF_WINDOW = (2, 1, 0, -1, -2)       # VideoFlow MOF, read forward slot 1  (t -> t-1)
ARFLOW_WINDOW = (1, 0, -1)           # ARFlow 3-frame, read flows_fw[0]    (t -> t-1)
MEMFLOW_CLIP = (3, 2, 1, 0, -1)      # MemFlow: step through the clip, keep the last flow

# Image resize backend used by each evaluator in the paper runs.
IMAGE_RESIZE_BY_FAMILY = {
    "raft": "pil_bicubic", "searaft": "pil_bicubic", "waft": "pil_bicubic",
    "memflow": "pil_bicubic", "videoflow": "pil_bicubic", "pwcnet": "pil_bicubic",
    "vcn": "pil_bicubic", "liteflownet3": "pil_bicubic",
    "arflow": "pil_bilinear", "flownet2": "pil_bilinear",
    "gmflow": "cv2_area", "unimatch": "cv2_area", "flowformerplusplus": "cv2_area",
    "fastflownet": "cv2_area", "pyflow": "cv2_area",
    "flowdiffusion": "cv2_area",  # + internal bicubic resize to 320x448 inside the model wrapper
}


# --------------------------------------------------------------------------- sample lists

def _scene_dirs(scene_root):
    """Accept either a flat scene dir (images/flow/labels) or a parent of scenes."""
    if osp.isdir(osp.join(scene_root, "images")) and osp.isdir(osp.join(scene_root, "flow")):
        return [scene_root]
    out = []
    for d in sorted(os.listdir(scene_root)):
        p = osp.join(scene_root, d)
        if osp.isdir(osp.join(p, "images")) and osp.isdir(osp.join(p, "flow")):
            out.append(p)
    if not out:
        raise FileNotFoundError(f"no images/ + flow/ under {scene_root}")
    return out


def list_samples(scene_root, offsets=TWO_FRAME, require_labels=True):
    """Return ``[(*frame_paths, flow_path, label_path), ...]``.

    ``frame_paths[k]`` is the image at offset ``offsets[k]`` from the reference frame
    t; with the default ``TWO_FRAME`` a sample is ``(I_t, I_{t-1}, <t>_flow.npy,
    <t>_label.png)``. Reference frames are all frames t >= 1 (by sorted position)
    that have ``<t>_flow.npy`` (and a label if ``require_labels``). Offsets outside
    the sequence are clamped to the first/last frame (edge replication). Frames are
    addressed by sorted position, which equals the frame number for the contiguous
    Nature3 sequence.
    """
    if 0 not in offsets:
        raise ValueError("offsets must contain 0 (the reference frame)")
    samples = []
    for scene in _scene_dirs(scene_root):
        imgs = sorted(glob.glob(osp.join(scene, "images", "*.png")))
        n = len(imgs)
        for i in range(1, n):  # frame 0 has no t-1 -> never a reference
            base = osp.splitext(osp.basename(imgs[i]))[0]
            ff = osp.join(scene, "flow", f"{base}_flow.npy")
            lf = osp.join(scene, "labels", f"{base}_label.png")
            if not osp.exists(ff):
                continue
            if require_labels and not osp.exists(lf):
                continue
            frames = [imgs[min(max(i + o, 0), n - 1)] for o in offsets]
            samples.append((*frames, ff, lf if osp.exists(lf) else None))
    return samples


def ref_base(sample):
    """Reference-frame basename (e.g. ``"0100"``) of a sample tuple."""
    for el in sample:
        if isinstance(el, str) and el.endswith("_flow.npy"):
            return osp.basename(el)[: -len("_flow.npy")]
    raise ValueError("sample has no *_flow.npy element")


# --------------------------------------------------------------------------- resizing

def _bilinear_coords(n_in, n_out):
    """Source indices/weights of torch bilinear interpolation, align_corners=False."""
    scale = n_in / n_out
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5
    src = np.maximum(src, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w1 = src - i0
    return i0, i1, 1.0 - w1, w1


def resize_bilinear(arr, size):
    """Resize an (H, W[, C]) array to ``size=(H', W')`` like
    ``F.interpolate(mode="bilinear", align_corners=False)`` (no antialiasing)."""
    H, W = size
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[:2] == (H, W):
        return a.astype(np.float32)
    y0, y1, wy0, wy1 = _bilinear_coords(a.shape[0], H)
    x0, x1, wx0, wx1 = _bilinear_coords(a.shape[1], W)
    ex = (slice(None),) + (None,) * (a.ndim - 1)
    rows = a[y0] * wy0[ex] + a[y1] * wy1[ex]
    exw = (None, slice(None)) + (None,) * (a.ndim - 2)
    out = rows[:, x0] * wx0[exw] + rows[:, x1] * wx1[exw]
    return out.astype(np.float32)


def resize_nearest(arr, size):
    """Resize like torch legacy ``mode="nearest"``: ``src = floor(dst * in / out)``."""
    H, W = size
    a = np.asarray(arr)
    if a.shape[:2] == (H, W):
        return a
    yi = np.minimum(np.floor(np.arange(H) * (a.shape[0] / H)).astype(np.int64), a.shape[0] - 1)
    xi = np.minimum(np.floor(np.arange(W) * (a.shape[1] / W)).astype(np.int64), a.shape[1] - 1)
    return a[yi][:, xi]


# --------------------------------------------------------------------------- readers

def read_flow(path, size=None):
    """Ground-truth backward flow ``(H, W, 2)`` float32 in pixels at ``size=(H, W)``."""
    flow = np.load(path).astype(np.float32)
    if size is None or tuple(size) == flow.shape[:2]:
        return flow
    oh, ow = flow.shape[:2]
    H, W = size
    out = resize_bilinear(flow, (H, W))
    out[..., 0] *= W / ow
    out[..., 1] *= H / oh
    return out


def read_label(path, size=None):
    """Drone label map ``(H, W)`` uint8 (0 = background, 1..6 = drone pass index)."""
    from PIL import Image

    lab = np.array(Image.open(path))
    if lab.ndim == 3:  # defensive: take the first channel of an RGB(A) export
        lab = lab[..., 0]
    if size is not None:
        lab = resize_nearest(lab, size)
    return lab


def drone_mask(label):
    """Boolean drone mask (union of all drones): ``label > 0``."""
    return np.asarray(label) > 0


def read_image(path, size=None, resize="pil_bicubic"):
    """RGB image ``(H, W, 3)`` uint8 at ``size=(H, W)``.

    ``resize`` selects the backend used by the paper evaluators:
    ``"pil_bicubic"`` (``Image.resize`` default), ``"pil_bilinear"`` or
    ``"cv2_area"`` (``cv2.resize(..., interpolation=cv2.INTER_AREA)``, needs OpenCV).
    """
    if resize == "cv2_area":
        import cv2

        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if size is not None and img.shape[:2] != tuple(size):
            img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        return img
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if size is not None and (im.height, im.width) != tuple(size):
        if resize == "pil_bicubic":
            im = im.resize((size[1], size[0]), Image.BICUBIC)
        elif resize == "pil_bilinear":
            im = im.resize((size[1], size[0]), Image.BILINEAR)
        else:
            raise ValueError(f"unknown resize backend {resize!r}")
    return np.array(im)


def load_sample(sample, size=None, n_frames=2, resize="pil_bicubic", load_images=True):
    """Load one sample tuple from ``list_samples``.

    Returns a dict with ``frames`` (list of HxWx3 uint8, in ``offsets`` order; empty if
    ``load_images=False``), ``flow`` (HxWx2 float32, backward GT of the reference
    frame), ``label`` (HxW uint8), ``mask`` (HxW bool) and ``ref`` (basename).
    """
    frames = list(sample[:n_frames])
    flow_path, label_path = sample[n_frames], sample[n_frames + 1]
    out = {
        "ref": ref_base(sample),
        "frames": [read_image(p, size, resize) for p in frames] if load_images else [],
        "flow": read_flow(flow_path, size),
    }
    if label_path is not None:
        out["label"] = read_label(label_path, size)
    else:
        shp = out["flow"].shape[:2]
        out["label"] = np.zeros(shp, np.uint8)
    out["mask"] = drone_mask(out["label"])
    return out

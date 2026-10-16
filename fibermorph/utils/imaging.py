"""Image utilities — resolution-aware downsampling.

Cross-section scans are often 20 MP / tens of MB, far more pixels than a shape
measurement needs. Downsampling to a target working resolution (and scaling the
resolution together with the pixel count) keeps the µm measurements essentially
unchanged (<0.5% on area/eccentricity in testing) while making processing much
lighter.
"""

from __future__ import annotations

import cv2
import numpy as np

# Target working resolution for cross-section analysis, in PIXELS PER µm. A
# ~100 µm section is ~200 px across at 2.0 px/µm — ample for area/eccentricity/EFD.
TARGET_SECTION_RES_MU = 2.0


# Never shrink an image below this longest-side (px); protects already-small
# images from being over-reduced.
MIN_LONG_SIDE = 400


def downsample_to_resolution(
    gray: np.ndarray,
    resolution_mu: float | None,
    target: float = TARGET_SECTION_RES_MU,
    min_long_side: int = MIN_LONG_SIDE,
) -> tuple[np.ndarray, float | None]:
    """Downsample ``gray`` so its working resolution is ~= ``target`` px/µm.

    Never upsamples, and never shrinks the longest side below ``min_long_side``.
    Returns ``(gray_out, effective_resolution_mu)`` where the effective
    resolution reflects the actual (rounded) resize ratio, so µm measurements
    computed at that resolution match the full-resolution values.

    If ``resolution_mu`` is None, already <= target, or the image is already
    small, it is returned unchanged.
    """
    if resolution_mu is None or resolution_mu <= target:
        return gray, resolution_mu

    factor = resolution_mu / target
    if max(gray.shape[:2]) / factor < min_long_side:
        return gray, resolution_mu

    new_w  = max(1, round(gray.shape[1] / factor))
    new_h  = max(1, round(gray.shape[0] / factor))
    gray_ds = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    effective_res = resolution_mu * (new_w / gray.shape[1])
    return gray_ds, effective_res


# ---------------------------------------------------------------------------
# Memory guard for curvature analysis
# ---------------------------------------------------------------------------
# The ridge filter (step 1 of curvature_seq) dominates peak memory. Measured on
# Great Lakes with the published settings: 4,440 MB for a 20.3 MP image and
# 1,258 MB for a 5.1 MP image, i.e. about 200 MB + 210 MB per megapixel.
CURV_MB_BASE = 200.0
CURV_MB_PER_MP = 210.0

# Share of the memory that is actually free that one analysis may plan to use;
# the rest is left for Streamlit, the uploaded file and the other widgets.
MEMORY_SHARE = 0.6

# Below this linear scale (a quarter of the width and height) the fibers are only
# a pixel or two wide and the curvature is no longer meaningful, so an image that
# would need a smaller scale is refused with a message instead of being reduced.
MIN_CURV_SCALE = 0.25


def curvature_peak_memory_mb(height: int, width: int) -> float:
    """Estimated peak memory (MB) of curvature analysis on a height x width image."""
    return CURV_MB_BASE + CURV_MB_PER_MP * (height * width) / 1e6


def _read_int(path: str):
    try:
        with open(path) as fh:
            text = fh.read().strip()
        return None if text in ("max", "") else int(text)
    except (OSError, ValueError):
        return None


def available_memory_mb() -> float | None:
    """Memory (MB) this process can still use, or None if it cannot be determined.

    The smallest of: free memory on the machine (``MemAvailable``) and, for every
    cgroup level from this process's group up to the root, the limit minus the
    current usage. cgroups are what cap login-node sessions, Slurm jobs and
    containers such as Streamlit Cloud, so a machine with plenty of RAM but a
    small per-user cap is reported correctly.
    """
    candidates = []
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) / 1024.0)
                    break
    except (OSError, ValueError, IndexError):
        pass

    try:
        with open("/proc/self/cgroup") as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = []
    v2 = next((ln.split(":", 2)[2] for ln in lines if ln.startswith("0::")), None)
    if v2 is not None:
        parts = [p for p in v2.split("/") if p]
        for depth in range(len(parts) + 1):
            base = "/sys/fs/cgroup/" + "/".join(parts[:depth])
            limit = _read_int(base.rstrip("/") + "/memory.max")
            used = _read_int(base.rstrip("/") + "/memory.current")
            if limit is not None and used is not None:
                candidates.append((limit - used) / 1048576.0)
    else:  # cgroup v1
        limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        used = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if limit is not None and used is not None and limit < 1 << 60:
            candidates.append((limit - used) / 1048576.0)

    return max(0.0, min(candidates)) if candidates else None


def memory_budget_mb() -> float | None:
    """Memory (MB) one curvature analysis may plan to use; None means no limit known.

    The environment variable FIBERMORPH_MAX_MEMORY_MB overrides the detection
    (for example on a host that hides its limit).
    """
    import os

    override = os.environ.get("FIBERMORPH_MAX_MEMORY_MB")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    avail = available_memory_mb()
    return None if avail is None else MEMORY_SHARE * avail


def curvature_scale_for_budget(height: int, width: int, budget_mb: float | None) -> float:
    """Linear scale (<= 1) at which curvature analysis of this image fits the budget.

    1.0 when it already fits, or when no budget is known. A result below
    ``MIN_CURV_SCALE`` means the image cannot be analyzed within the budget.
    """
    if budget_mb is None or curvature_peak_memory_mb(height, width) <= budget_mb:
        return 1.0
    megapixels = max(budget_mb - CURV_MB_BASE, 0.0) / CURV_MB_PER_MP
    scale = (megapixels * 1e6 / (height * width)) ** 0.5
    return min(1.0, scale)


def write_scaled_copy(src_path: str, dst_path: str, scale: float):
    """Save a grayscale copy of ``src_path`` shrunk by ``scale``; return (effective_scale, (w, h)).

    The effective scale is the exact ratio of the new width to the old one, so a
    resolution given in px per unit can be multiplied by it.
    """
    from PIL import Image

    with Image.open(src_path) as im:
        gray = np.array(im.convert("L"))
    h, w = gray.shape
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    Image.fromarray(small).save(dst_path)
    return new_w / w, (new_w, new_h)

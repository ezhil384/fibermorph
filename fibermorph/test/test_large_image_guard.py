"""Memory guard for curvature analysis in the GUI.

A 20-megapixel curvature scan needs about 4.4 GB at its peak, which gets the
process killed on a login node (4 GiB cap) or a hosted app. Images that do not
fit the memory this process can use are shrunk, with the resolution and window
scaled alongside, and the output says so; images that fit are untouched.
"""

import io
import os

import numpy as np
import pytest
from PIL import Image

from fibermorph.utils import imaging

# ---------------------------------------------------------------------------
# Estimate and scale
# ---------------------------------------------------------------------------


def test_memory_estimate_matches_the_measurements():
    # Measured on Great Lakes: 4,440 MB at 5208 x 3904 and 1,258 MB at 2604 x 1952.
    assert imaging.curvature_peak_memory_mb(3904, 5208) == pytest.approx(4440, rel=0.03)
    assert imaging.curvature_peak_memory_mb(1952, 2604) == pytest.approx(1258, rel=0.03)


def test_image_that_fits_is_not_scaled():
    assert imaging.curvature_scale_for_budget(3904, 5208, 8000) == 1.0
    assert imaging.curvature_scale_for_budget(3904, 5208, None) == 1.0   # no limit known


def test_scale_makes_the_estimate_fit_the_budget():
    for budget in (3000, 2000, 1200, 800):
        s = imaging.curvature_scale_for_budget(3904, 5208, budget)
        assert 0 < s < 1
        assert imaging.curvature_peak_memory_mb(3904 * s, 5208 * s) == pytest.approx(budget, rel=0.01)


def test_hopeless_budget_gives_a_scale_below_the_minimum():
    assert imaging.curvature_scale_for_budget(3904, 5208, 150) < imaging.MIN_CURV_SCALE


def test_budget_env_override(monkeypatch):
    monkeypatch.setenv("FIBERMORPH_MAX_MEMORY_MB", "1234")
    assert imaging.memory_budget_mb() == 1234
    monkeypatch.setenv("FIBERMORPH_MAX_MEMORY_MB", "not a number")
    assert imaging.memory_budget_mb() != "not a number"      # falls back to detection


def test_available_memory_is_positive_or_unknown():
    avail = imaging.available_memory_mb()
    assert avail is None or avail >= 0


def test_write_scaled_copy_returns_the_exact_scale(tmp_path):
    src = tmp_path / "big.png"
    Image.fromarray(np.full((301, 403), 120, np.uint8)).save(src)
    eff, (w, h) = imaging.write_scaled_copy(str(src), str(tmp_path / "small.tif"), 0.5)
    assert (w, h) == (202, 150)
    assert eff == pytest.approx(202 / 403)
    assert Image.open(tmp_path / "small.tif").size == (202, 150)


# ---------------------------------------------------------------------------
# In the GUI
# ---------------------------------------------------------------------------

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import fibermorph.gui  # noqa: E402
from pathlib import Path  # noqa: E402

APP_PATH = str(Path(fibermorph.gui.__file__).parent / "app.py")


def _fiber_png(size=1000):
    img = np.full((size, size), 230, dtype=np.uint8)
    xs = np.arange(40, size - 40)
    ys = (size // 2 + 60 * np.sin(2 * np.pi * 2 * xs / size)).astype(int)
    for x, y in zip(xs, ys):
        img[y - 3:y + 4, x] = 20
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _run_curvature(monkeypatch, budget_mb, auto_reduce=True):
    monkeypatch.delenv("FIBERMORPH_LOCAL", raising=False)
    if budget_mb is None:
        monkeypatch.delenv("FIBERMORPH_MAX_MEMORY_MB", raising=False)
    else:
        monkeypatch.setenv("FIBERMORPH_MAX_MEMORY_MB", str(budget_mb))
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["active_view"] = "curvature"
    at.run()
    at.checkbox(key="curv_auto_reduce").set_value(auto_reduce)
    at.file_uploader[0].set_value(("fiber.png", _fiber_png(), "image/png"))
    at.run()
    [b for b in at.button if b.key == "curv_run"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


needs_upload_support = pytest.mark.skipif(
    not hasattr(AppTest, "file_uploader"), reason="AppTest cannot simulate uploads")


@needs_upload_support
def test_image_that_fits_is_analyzed_at_full_size_with_unchanged_columns(monkeypatch):
    at = _run_curvature(monkeypatch, budget_mb=50000)
    summ = at.session_state["curvature_summary"]
    assert list(summ.columns) == ["source_file", "n_fragments", "curv_mean", "curv_median",
                                  "length_mean", "length_median", "length_total"]
    assert "analysis_scale" not in at.session_state["curvature_fragments"].columns
    assert not [w for w in at.warning if "Reduced to fit in memory" in w.value]


@needs_upload_support
def test_image_that_does_not_fit_is_reduced_and_says_so(monkeypatch):
    # 1 MP needs about 410 MB; a 320 MB budget forces a reduction to ~0.76.
    at = _run_curvature(monkeypatch, budget_mb=320)
    summ = at.session_state["curvature_summary"]
    frags = at.session_state["curvature_fragments"]
    assert 0.4 < summ["analysis_scale"].iloc[0] < 0.95
    assert list(frags["analysis_scale"].unique()) == list(summ["analysis_scale"])
    [msg] = [w.value for w in at.warning if "Reduced to fit in memory" in w.value]
    assert "fiber.png" in msg and "1000 × 1000" in msg and "analysis_scale" in msg


@needs_upload_support
def test_reduced_result_stays_close_to_full_size(monkeypatch):
    """Lengths are in mm either way, so a modest reduction changes them only a little."""
    full = _run_curvature(monkeypatch, budget_mb=50000).session_state["curvature_summary"]
    small = _run_curvature(monkeypatch, budget_mb=320).session_state["curvature_summary"]
    assert small["length_total"].iloc[0] == pytest.approx(full["length_total"].iloc[0], rel=0.15)


@needs_upload_support
def test_image_that_cannot_fit_even_at_a_quarter_is_refused_with_a_message(monkeypatch):
    at = _run_curvature(monkeypatch, budget_mb=205)    # 1 MP at 205 MB would need a scale of 0.15
    assert any("too small to measure curvature" in w.value for w in at.warning)
    assert any(e.value == "No curvature fragments were measured." for e in at.error)


@needs_upload_support
def test_reduction_can_be_turned_off(monkeypatch):
    at = _run_curvature(monkeypatch, budget_mb=320, auto_reduce=False)
    assert any("will probably be killed" in w.value for w in at.warning)
    assert "analysis_scale" not in at.session_state["curvature_summary"].columns

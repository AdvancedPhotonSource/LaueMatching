"""Two reader/writer mismatches fixed in 0.7.2.

* laue_visualization writes ``<output_path>.unique_spot_counts.txt``, but
  output.store_txt_files_in_h5 looked for ``<output_path>.bin.unique_spot_counts.txt``,
  so the counts never reached the HDF5.
* GaussSigmaMax was applied by laue_index.preprocess (streaming) but ignored by
  RunImage's own blur.
"""
import inspect

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def test_unique_spot_counts_text_reaches_the_h5(tmp_path):
    from laue_index.output import store_txt_files_in_h5
    import laue_visualization as lv
    base = str(tmp_path / "img_001")
    # the writer's own file name, taken from its source
    assert 'f"{output_path}.unique_spot_counts.txt"' in inspect.getsource(
        lv.create_simulation_comparison_visualization)
    (tmp_path / "img_001.unique_spot_counts.txt").write_text(
        "Grain_Nr\tUnique_Experimental_Spots\n3\t17\n")
    with h5py.File(tmp_path / "o.h5", "w") as hf:
        store_txt_files_in_h5(base, hf)
        assert "/entry/results/unique_spot_counts_text" in hf
        txt = hf["/entry/results/unique_spot_counts_text"][()].decode()
    assert "3\t17" in txt


class _Progress:
    def update(self, *a, **k):
        pass

    def complete(self, *a, **k):
        pass


def _segment_blur(tmp_path, smax):
    """Run RunImage's real _segment (components -> filter -> blur) on a small
    synthetic frame and return (blurred, filtered, centers, processor)."""
    import RunImage
    from laue_index.pipeline.laue_config import ConfigurationManager
    p = tmp_path / f"params_{smax}.txt"
    p.write_text("SpaceGroup 225\nNrPxX 128\nNrPxY 128\nPxX 0.0002\nPxY 0.0002\n"
                 "P_Array 0.02 0.002 0.513\nMinArea 2\nWatershedImage 0\n"
                 + (f"GaussSigmaMax {smax}\n" if smax else ""))
    cm = ConfigurationManager(str(p))
    proc = object.__new__(RunImage.EnhancedImageProcessor)   # no background load
    proc.config = cm
    img = np.zeros((128, 128), dtype=np.uint16)
    for y, x in ((20, 30), (60, 90), (100, 40), (40, 110)):
        img[y - 1:y + 2, x - 1:x + 2] = 500
    with h5py.File(tmp_path / f"seg_{smax}.h5", "w") as hf:
        out = proc._segment(img, str(tmp_path / f"o_{smax}"), "x.h5",
                            hf.require_group("/entry/data"), _Progress())
    return out, proc


def test_runimage_applies_gauss_sigma_max(tmp_path):
    """Behavioural: the blur RunImage feeds the indexer uses
    min(auto sigma, GaussSigmaMax), and no cap when the key is absent."""
    from scipy import ndimage as ndimg
    free, proc = _segment_blur(tmp_path, 0)
    auto = proc.calculate_gaussian_width(free["centers"], 0.0002, 0.513, 0.4)
    smax = auto / 3.0
    assert smax > 0
    capped, _ = _segment_blur(tmp_path, round(smax, 6))
    ftd = free["filtered_thresholded_image"].astype(np.double)
    np.testing.assert_allclose(free["blurred_image"],
                               ndimg.gaussian_filter(ftd, auto))
    np.testing.assert_allclose(capped["blurred_image"],
                               ndimg.gaussian_filter(ftd, round(smax, 6)))
    assert not np.allclose(capped["blurred_image"], free["blurred_image"])


def test_gauss_sigma_max_reaches_runimage_config(tmp_path):
    from laue_index.pipeline.laue_config import ConfigurationManager
    p = tmp_path / "params.txt"
    p.write_text("SpaceGroup 225\nGaussSigmaMax 1.5\n")
    assert ConfigurationManager(str(p)).get("gauss_sigma_max") == 1.5

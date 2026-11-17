import numpy as np
import pytest
from created_signal import compare, processing as P, reference as R

FS = 1000


@pytest.mark.parametrize("kind", P.WAVE_KINDS)
@pytest.mark.parametrize("freq,phase", [(5, 0.0), (13.7, 1.1), (50, -2.0)])
def test_generator_matches_scipy(kind, freq, phase):
    _, a = P.generate_signal(kind, 1.3, freq, phase, FS, 1.0)
    _, b = R.ref_generate(kind, 1.3, freq, phase, FS, 1.0)
    assert np.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("n", [1000, 999, 777, 1024])
def test_analyze_matches_numpy(n):
    x = np.random.default_rng(0).standard_normal(n)
    fa, ma, pa = P.analyze(x, FS)
    fb, mb, pb = R.ref_analyze(x, FS)
    assert np.allclose(fa, fb) and np.allclose(ma, mb, atol=1e-12)
    big = mb > 1e-6
    diff = np.angle(np.exp(1j * (pa[big] - pb[big])))        # -pi and +pi are the same angle
    assert np.allclose(diff, 0, atol=1e-9)


@pytest.mark.parametrize("spec", [
    {"type": "moving_average", "size": 7}, {"type": "gaussian", "size": 11},
    {"type": "difference"}, {"type": "custom", "kernel": [1, 2, 3, 2, 1]}])
def test_conv_filters_match_scipy(spec):
    x = np.random.default_rng(1).standard_normal(500)
    assert np.allclose(P.apply_filter_spec(x, FS, spec), R.ref_apply_filter_spec(x, FS, spec),
                       atol=1e-12)


@pytest.mark.parametrize("spec", [
    {"type": "lowpass", "cutoff": 50}, {"type": "highpass", "cutoff": 50, "order": 3},
    {"type": "bandpass", "cutoff": 20, "cutoff_high": 80},
    {"type": "bandpass", "cutoff": 20, "cutoff_high": 80, "order": 2},
    {"type": "notch", "cutoff": 50, "width": 6}])
@pytest.mark.parametrize("n", [1000, 777])
def test_freq_filters_match_numpy(spec, n):
    x = np.random.default_rng(2).standard_normal(n)
    assert np.allclose(P.apply_filter_spec(x, FS, spec), R.ref_apply_filter_spec(x, FS, spec),
                       atol=1e-10)


def test_error_metrics():
    m, e = compare.error_metrics([1, 2, 3], [1, 2, 3.5])
    assert m["max_abs"] == pytest.approx(0.5) and not m["ok"]
    m, _ = compare.error_metrics([1, 2, 3], [1, 2, 3])
    assert m["ok"] and m["rel"] == 0
    m, _ = compare.error_metrics([np.pi - 1e-9], [-np.pi + 1e-9], circular=True)
    assert m["max_abs"] < 1e-8                                   # wraps around correctly
    with pytest.raises(ValueError):
        compare.error_metrics([1, 2], [1, 2, 3])


def test_compare_all_reports_match_and_errors():
    comps = [dict(kind="square", amp=1, freq=5, phase=0.3), dict(kind="triangle", amp=1, freq=13, phase=1)]
    _, clean = P.generate_signal("sine", 1, 7, 0, FS, 1)
    x = P.add_noise(clean, 10, 1)
    res = compare.compare_all(comps, FS, 1.0, x, {"type": "gaussian", "size": 9},
                              {"type": "bandpass", "cutoff": 10, "cutoff_high": 60, "order": 2})
    assert len(res["rows"]) == 9 and res["all_ok"]
    assert all(r["max_abs"] < 1e-9 for r in res["rows"])
    assert all(r["ours_ms"] > 0 and r["ref_ms"] > 0 for r in res["rows"])
    assert set(res["errors"]) == {"freqs", "magnitude", "conv", "freq"}


def test_compare_detects_a_wrong_implementation(monkeypatch):
    """The comparison must FAIL when our code is wrong (otherwise it proves nothing)."""
    comps = [dict(kind="sine", amp=1, freq=5, phase=0)]
    _, x = P.generate_signal("sine", 1, 5, 0, FS, 1)
    monkeypatch.setattr(P, "apply_conv_filter", lambda x, *a, **k: np.asarray(x) * 1.001)
    res = compare.compare_all(comps, FS, 1.0, x, {"type": "moving_average", "size": 5},
                              {"type": "lowpass", "cutoff": 20})
    assert not res["all_ok"]
    assert not [r for r in res["rows"] if r["name"].startswith("Convolution filter")][0]["ok"]

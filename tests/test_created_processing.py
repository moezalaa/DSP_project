import numpy as np
import pytest
from created_signal import processing as P

FS = 1000


@pytest.mark.parametrize("kind", P.WAVE_KINDS)
def test_waveform_basics(kind):
    t, x = P.generate_signal(kind, 2.0, 5, 0.0, FS, 1.0)
    assert len(t) == len(x) == 1000
    assert np.isclose(t[1] - t[0], 1 / FS)
    assert np.max(np.abs(x)) <= 2.0 + 1e-12
    assert np.max(np.abs(x)) > 1.9


def test_sine_cosine_phase():
    _, s = P.generate_signal("sine", 1, 10, np.pi / 2, FS, 1)
    _, c = P.generate_signal("cosine", 1, 10, 0, FS, 1)
    assert np.allclose(s, c)


def test_square_values_and_period():
    _, x = P.generate_signal("square", 1, 10, 0, FS, 1)
    assert set(np.unique(x)) == {-1.0, 1.0}
    assert np.allclose(x[:100], x[100:200])           # period = 100 samples


def test_triangle_and_sawtooth_shape():
    _, tri = P.generate_signal("triangle", 1, 1, 0, FS, 1)
    assert np.isclose(tri[250], 1.0, atol=1e-6) and np.isclose(tri[750], -1.0, atol=1e-6)
    _, saw = P.generate_signal("sawtooth", 1, 1, 0, FS, 1)
    assert np.isclose(saw[0], -1.0) and saw[999] > 0.99 and np.all(np.diff(saw) > 0)


@pytest.mark.parametrize("bad", [dict(fs=0), dict(duration=0), dict(freq=-1), dict(kind="zigzag")])
def test_generate_bad_input(bad):
    args = dict(kind="sine", amp=1, freq=5, phase_rad=0, fs=FS, duration=1)
    args.update(bad)
    with pytest.raises(ValueError):
        P.generate_signal(**args)


def test_combine():
    a, b = np.ones(5), np.arange(5.0)
    assert np.allclose(P.combine_signals([a, b]), a + b)
    with pytest.raises(ValueError):
        P.combine_signals([])
    with pytest.raises(ValueError):
        P.combine_signals([np.ones(3), np.ones(4)])


@pytest.mark.parametrize("snr", [0, 10, 20])
def test_noise_snr(snr):
    _, x = P.generate_signal("sine", 1, 5, 0, 10000, 5)
    noise = P.add_noise(x, snr, seed=1) - x
    measured = 10 * np.log10(np.mean(x ** 2) / np.mean(noise ** 2))
    assert abs(measured - snr) < 0.5


def test_noise_is_reproducible_and_validates():
    x = np.sin(np.arange(100) / 5)
    assert np.allclose(P.add_noise(x, 10, seed=3), P.add_noise(x, 10, seed=3))
    assert not np.allclose(P.add_noise(x, 10, seed=3), P.add_noise(x, 10, seed=4))
    with pytest.raises(ValueError):
        P.add_noise(np.zeros(10), 10)


def test_analyze_peak_amplitude_and_frequency():
    _, x = P.generate_signal("sine", 1.5, 50, 0, FS, 1)
    f, m, _ = P.analyze(x, FS)
    assert f[np.argmax(m)] == 50
    assert np.isclose(m.max(), 1.5, atol=1e-9)
    assert len(f) == 501


def test_analyze_phase_of_cosine_is_its_phase():
    _, x = P.generate_signal("cosine", 1, 20, 0.7, FS, 1)
    f, m, ph = P.analyze(x, FS)
    assert np.isclose(ph[np.argmax(m)], 0.7, atol=1e-9)


def test_moving_average_smooths_noise():
    _, x = P.generate_signal("sine", 1, 5, 0, FS, 1)
    noisy = P.add_noise(x, 5, seed=0)
    y = P.apply_conv_filter(noisy, "moving_average", 9)
    assert len(y) == len(noisy)
    assert np.mean((y - x) ** 2) < np.mean((noisy - x) ** 2)


def test_difference_of_ramp_is_constant():
    y = P.apply_conv_filter(np.arange(10.0), "difference")
    assert np.allclose(y[1:], 1.0)


def test_kernels_sum_to_one_and_custom():
    assert np.isclose(P.make_kernel("gaussian", 9).sum(), 1)
    assert np.isclose(P.make_kernel("moving_average", 7).sum(), 1)
    assert np.allclose(P.make_kernel("custom", kernel=[1, 2, 1]), [1, 2, 1])
    with pytest.raises(ValueError):
        P.make_kernel("custom", kernel=[])


def test_filter_spec_dispatch():
    _, x = P.generate_signal("sine", 1, 5, 0, FS, 1)
    assert np.allclose(P.apply_filter_spec(x, FS, {"type": "none"}), x)
    y = P.apply_filter_spec(x + 0.5 * np.sin(2 * np.pi * 200 * np.arange(1000) / FS), FS,
                            {"type": "lowpass", "cutoff": 50})
    assert np.allclose(y, x, atol=1e-6)
    y = P.apply_filter_spec(x, FS, {"type": "bandpass", "cutoff": 2, "cutoff_high": 10})
    assert np.allclose(y, x, atol=1e-6)
    with pytest.raises(ValueError):
        P.apply_filter_spec(x, FS, {"type": "wavelet"})


def test_nyquist_warning():
    comps = [{"freq": 100}, {"freq": 600}]
    assert len(P.nyquist_warnings(comps, 1000)) == 1


def test_aliasing_really_happens():
    _, x = P.generate_signal("sine", 1, 700, 0, FS, 1)      # 700 Hz at fs=1000 -> 300 Hz
    f, m, _ = P.analyze(x, FS)
    assert f[np.argmax(m)] == 300


def test_pipeline_end_to_end():
    r = P.run_pipeline({
        "fs": FS, "duration": 1.0,
        "components": [dict(kind="sine", amp=1, freq=5, phase=0),
                       dict(kind="sine", amp=0.5, freq=100, phase=0)],
        "noise_snr": None, "filter": {"type": "lowpass", "cutoff": 20}})
    assert np.allclose(r["input"], r["clean"])
    f, m = r["freqs"], r["mag_out"]
    assert f[np.argmax(m)] == 5 and m[100] < 1e-9           # 100 Hz removed
    assert r["mag_in"][100] == pytest.approx(0.5, abs=1e-9)

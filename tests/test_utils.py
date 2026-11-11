import numpy as np
from core import utils


def test_magnitude_phase():
    X = np.array([3 + 4j, -1 + 0j, 0 + 2j])
    assert np.allclose(utils.magnitude(X), np.abs(X))
    assert np.allclose(utils.phase(X), np.angle(X))


def test_to_db():
    assert np.isclose(utils.to_db(np.array([1.0]))[0], 0.0, atol=1e-6)
    assert np.isclose(utils.to_db(np.array([10.0]))[0], 20.0, atol=1e-6)


def test_normalize():
    y = utils.normalize(np.array([2.0, 4.0, 6.0]))
    assert np.allclose(y, [0, 0.5, 1])
    assert np.allclose(utils.normalize(np.array([5.0, 5.0])), [0, 0])


def test_freq_axis():
    f = utils.freq_axis(8, 80)
    assert np.allclose(f, np.arange(8) * 10)
    assert np.allclose(utils.freq_axis(8, 80, shift=True), np.arange(-4, 4) * 10)


def test_fftshift_matches_numpy():
    for shape in [(8,), (7,), (6, 8), (5, 7)]:
        a = np.random.rand(*shape)
        assert np.allclose(utils.fftshift(a), np.fft.fftshift(a))
        assert np.allclose(utils.ifftshift(a), np.fft.ifftshift(a))
        assert np.allclose(utils.ifftshift(utils.fftshift(a)), a)


def test_one_sided():
    f = np.arange(8.0)
    s = np.arange(8.0) * 2
    f2, s2 = utils.one_sided(f, s)
    assert len(f2) == len(s2) == 5

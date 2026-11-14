"""Accuracy + speed comparison: OUR from-scratch code vs numpy / scipy.

compare_all(...) returns table rows (what the GUI shows) and the error signals
(what the GUI plots).
"""
import time

import numpy as np

from core.fourier import fft, ifft
from created_signal import processing as P
from created_signal import reference as R

TOLERANCE = 1e-8          # relative to the size of the values: "matches numpy"
DTFT_CHECK_SAMPLES = 128  # the DTFT loops are slow, so the table checks a short piece


def _best_time_ms(fn, repeat=3):
    best = np.inf
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0, result


def error_metrics(ours, ref, circular=False):
    """max |error|, RMSE and relative error (||e|| / ||ref||) between two arrays.

    Works for real and complex arrays.
    circular=True wraps differences into [-pi, pi] (use it for phase angles).
    """
    ours = np.asarray(ours)
    ref = np.asarray(ref)
    if ours.shape != ref.shape:
        raise ValueError(f"shape mismatch: {ours.shape} vs {ref.shape}")
    e = ours - ref
    if circular:
        e = np.angle(np.exp(1j * e))
    if e.size == 0:
        return {"max_abs": 0.0, "rmse": 0.0, "rel": 0.0, "ok": True}, e
    ae = np.abs(e)
    max_abs = float(np.max(ae))
    rmse = float(np.sqrt(np.mean(ae ** 2)))
    ref_norm = float(np.linalg.norm(ref))
    rel = float(np.linalg.norm(e) / ref_norm) if ref_norm > 0 else 0.0
    ok = max_abs <= TOLERANCE * max(1.0, float(np.max(np.abs(ref))))
    return {"max_abs": max_abs, "rmse": rmse, "rel": rel, "ok": bool(ok)}, e


def _row(name, ref_name, t_ours, t_ref, metrics):
    return {"name": name, "reference": ref_name, "ours_ms": t_ours, "ref_ms": t_ref, **metrics}


def compare_all(components, fs, duration, x, conv_spec, freq_spec, de_spec=None):
    """Compare every operation of the Created Signal pipeline (9 rows).

    components: list of {"kind","amp","freq","phase"} (phase in radians)
    x         : the actual input signal (after noise) used for the analysis/filters
    conv_spec : filter spec dict with a time-domain type (moving_average, ...)
    freq_spec : filter spec dict with a frequency-domain type (lowpass, ...)
    de_spec   : {"type": "first_order", "a": .., "b": ..}  (default a=0.9, b=0.1)
    """
    de_spec = de_spec or {"type": "first_order", "a": 0.9, "b": 0.1}
    rows = []

    # 1. waveform generation ------------------------------------------------
    def ours_gen():
        return P.combine_signals([P.generate_signal(c["kind"], c["amp"], c["freq"],
                                                    c["phase"], fs, duration)[1] for c in components])

    def ref_gen():
        return np.sum([R.ref_generate(c["kind"], c["amp"], c["freq"],
                                      c["phase"], fs, duration)[1] for c in components], axis=0)

    t1, a = _best_time_ms(ours_gen)
    t2, b = _best_time_ms(ref_gen)
    m, _ = error_metrics(a, b)
    rows.append(_row("Waveform generation", "numpy / scipy.signal", t1, t2, m))

    # 2. + 3. spectrum ----------------------------------------------------------
    t1, (fa, ma, pa) = _best_time_ms(lambda: P.analyze(x, fs))
    t2, (fb, mb, pb) = _best_time_ms(lambda: R.ref_analyze(x, fs))
    m_mag, e_mag = error_metrics(ma, mb)
    rows.append(_row("FFT: magnitude spectrum", "numpy.fft.rfft", t1, t2, m_mag))
    sig = mb > 1e-6 * max(mb.max(), 1e-300)            # only bins that really carry energy
    m_ph, _ = error_metrics(pa[sig], pb[sig], circular=True)
    rows.append(_row("FFT: phase spectrum", "numpy.angle(rfft)", t1, t2, m_ph))

    # 4. inverse FFT round trip -------------------------------------------------
    t1, rt_a = _best_time_ms(lambda: ifft(fft(x)).real)
    t2, rt_b = _best_time_ms(lambda: R.ref_roundtrip(x))
    m, _ = error_metrics(rt_a, rt_b)
    rows.append(_row("IFFT(FFT(x)) round trip", "numpy.fft.ifft", t1, t2, m))

    # 5. DTFT from the definition (loops) at the DFT frequencies w_k = 2*pi*k/N --
    xs = np.asarray(x, dtype=float)[:DTFT_CHECK_SAMPLES]
    n_d = len(xs)
    w_k = 2 * np.pi * np.arange(n_d) / n_d
    t1, Xa = _best_time_ms(lambda: P.dtft(xs, w_k), repeat=1)
    t2, Xb = _best_time_ms(lambda: R.ref_dtft_on_dft_grid(xs))
    m, _ = error_metrics(Xa, Xb)
    rows.append(_row(f"DTFT (loops, {n_d} samples)", "numpy.fft.fft", t1, t2, m))

    # 6. time-domain filter (convolution) --------------------------------------
    t1, ya = _best_time_ms(lambda: P.apply_filter_spec(x, fs, conv_spec))
    t2, yb = _best_time_ms(lambda: R.ref_apply_filter_spec(x, fs, conv_spec))
    m, e_conv = error_metrics(ya, yb)
    rows.append(_row(f"Convolution filter: {conv_spec['type']}", "scipy.signal.convolve", t1, t2, m))

    # 7. linear convolution through the frequency domain (zero-padded) -------
    h = P.make_kernel("moving_average", int(conv_spec.get("size", 5)))
    t1, ca = _best_time_ms(lambda: P.convolution_vs_frequency(x, h)["padded"])
    t2, cb = _best_time_ms(lambda: R.ref_linear_convolution(x, h))
    m, _ = error_metrics(ca, cb)
    rows.append(_row("Linear conv via FFT (zero-padded)", "numpy.convolve", t1, t2, m))

    # 8. first-order difference equation ------------------------------------------
    t1, da = _best_time_ms(lambda: P.apply_filter_spec(x, fs, de_spec))
    t2, db = _best_time_ms(lambda: R.ref_apply_filter_spec(x, fs, de_spec))
    m, _ = error_metrics(da, db)
    rows.append(_row("Difference equation y=a*y[n-1]+b*x", "scipy.signal.lfilter", t1, t2, m))

    # 9. frequency-domain filter ---------------------------------------------------
    t1, za = _best_time_ms(lambda: P.apply_filter_spec(x, fs, freq_spec))
    t2, zb = _best_time_ms(lambda: R.ref_apply_filter_spec(x, fs, freq_spec))
    m, e_freq = error_metrics(za, zb)
    rows.append(_row(f"Frequency filter: {freq_spec['type']}", "numpy.fft", t1, t2, m))

    return {"rows": rows,
            "errors": {"freqs": fa, "magnitude": e_mag, "conv": e_conv, "freq": e_freq},
            "all_ok": all(r["ok"] for r in rows)}

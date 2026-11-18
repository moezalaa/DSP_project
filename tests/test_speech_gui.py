"""Smoke tests for the Speech tab (offscreen, no display or microphone needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402

from speech_signal.gui import SpeechSignalTab  # noqa: E402

app = QApplication.instance() or QApplication([])
SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "assets", "speech_sample.wav")


@pytest.fixture
def errors(monkeypatch):
    """Collect error popups instead of opening them."""
    got = []
    monkeypatch.setattr("speech_signal.gui.show_error", lambda parent, msg, title="Error": got.append(str(msg)))
    return got


@pytest.fixture
def tab(errors):
    t = SpeechSignalTab()
    assert t.load_path(SAMPLE)
    return t


def test_tab_starts_empty_and_disabled():
    t = SpeechSignalTab()
    assert t.current is None
    assert not t.process_widgets[0].isEnabled()


def test_load_enables_controls_and_plots(tab):
    assert tab.fs == 44100 and tab.process_widgets[0].isEnabled()
    tab.plot_tabs.setCurrentIndex(1)       # spectrogram draws without error
    tab.plot_tabs.setCurrentIndex(0)


def test_filter_echo_noise_reset(tab, errors):
    n = len(tab.original)
    tab.filter_type.setCurrentText("lowpass"); tab.cutoff.setValue(1000)
    tab.apply_filter()
    assert len(tab.current) == n and len(tab.history) == 2
    tab.butterworth_check.setChecked(True)
    tab._telephone_preset(); tab.apply_filter()
    tab.apply_echo()
    assert len(tab.current) > n                       # echo tail makes it longer
    tab.apply_noise_reduction()
    assert not errors
    tab.reset_audio()
    assert np.array_equal(tab.current, tab.original) and len(tab.history) == 1


def test_bad_input_shows_message_not_crash(tab, errors):
    tab.filter_type.setCurrentText("bandpass")
    tab.cutoff.setValue(3000); tab.cutoff_high.setValue(500)
    tab.apply_filter()
    tab.filter_type.setCurrentText("lowpass"); tab.cutoff.setValue(30000)   # above Nyquist
    tab.apply_filter()
    assert len(errors) == 2


def test_missing_file_shows_message(errors):
    t = SpeechSignalTab()
    assert not t.load_path("does_not_exist.wav")
    assert errors


def test_apply_without_audio_shows_message(errors):
    SpeechSignalTab().apply_echo()
    assert errors


# ---------------------------------------------------------- hum / notch workflow
def test_add_tone_find_peak_and_iir_notch(tab, errors):
    tab.tone_freq.setValue(50); tab.tone_amp.setValue(0.1)
    tab.apply_tone()
    tab.detect_peak()
    assert abs(tab._peak_hz - 50) < 0.5 and "50" in tab.peak_label.text()

    tab.filter_type.setCurrentText("notch (IIR, 2nd order)")
    tab.cutoff.setValue(tab._peak_hz); tab.iir_bw.setValue(10)
    tab.apply_filter()
    assert not errors
    assert tab.coef_label.text().startswith("b = [") and "a = [" in tab.coef_label.text()
    assert len(tab.history) == 3
    # the 50 Hz bin is far smaller after the notch than in the noisy signal
    def hum(x):
        X = np.abs(np.fft.rfft(x))
        return X[int(round(50 * len(x) / tab.fs))]
    assert hum(tab.current) < 0.01 * hum(tab.previous)


def test_compare_modes_and_filter_controls_draw(tab, errors):
    tab.apply_tone()
    tab.filter_type.setCurrentText("notch (IIR, 2nd order)")
    assert tab.iir_bw.isVisibleTo(tab) and not tab.butterworth_check.isEnabled()
    tab.apply_filter()
    for mode in (0, 1, 2):
        tab.compare_combo.setCurrentIndex(mode)
    tab.reset_audio()
    assert tab.coef_label.text() == ""
    assert not errors


def test_new_controls_report_errors_instead_of_crashing(tab, errors):
    tab.tone_freq.setValue(30000)                    # above Nyquist (22050)
    tab.apply_tone()
    tab.filter_type.setCurrentText("notch (IIR, 2nd order)")
    tab.cutoff.setValue(30000)
    tab.apply_filter()
    assert len(errors) == 2


def test_hum_tools_without_audio_show_message(errors):
    t = SpeechSignalTab()
    t.detect_peak(); t.apply_tone(); t.apply_filter()
    assert len(errors) == 3

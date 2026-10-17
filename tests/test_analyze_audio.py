import numpy as np
import pytest

from analyze_audio import LABELS, _pitch_summary, find_inputs


def test_find_inputs_matches_tradition_names_case_insensitively(tmp_path):
    for name in ("hinduism.WAV", "Christianity.ogg", "ISLAM.mp3"):
        (tmp_path / name).touch()

    found = find_inputs(tmp_path)

    assert tuple(found) == LABELS
    assert found["Hinduism"].name == "hinduism.WAV"


def test_find_inputs_requires_all_three_recordings(tmp_path):
    (tmp_path / "hinduism.wav").touch()

    with pytest.raises(ValueError, match="Christianity"):
        find_inputs(tmp_path)


def test_find_inputs_rejects_extra_audio(tmp_path):
    for name in ("hinduism.wav", "christianity.wav", "islam.wav", "extra.wav"):
        (tmp_path / name).touch()

    with pytest.raises(ValueError, match="extra.wav"):
        find_inputs(tmp_path)


def test_pitch_summary_ignores_unvoiced_frames():
    pitch = np.full(21, 100.0)
    pitch[0] = np.nan
    median, pitch_range, voiced_fraction = _pitch_summary(
        pitch
    )

    assert median == pytest.approx(100.0)
    assert pitch_range == pytest.approx(0.0)
    assert voiced_fraction == pytest.approx(20 / 21)


def test_pitch_summary_marks_sparse_estimates_unavailable():
    median, pitch_range, _ = _pitch_summary(np.array([100.0, np.nan, np.nan]))

    assert np.isnan(median)
    assert np.isnan(pitch_range)

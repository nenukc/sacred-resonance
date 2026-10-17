from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import math

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
LABELS = ("Hinduism", "Christianity", "Islam")
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
SAMPLE_RATE = 22050
FFT_SIZE = 2048
HOP_LENGTH = 512


@dataclass
class Recording:
    tradition: str
    path: Path
    duration_seconds: float
    audio: np.ndarray
    rms_mean: float
    mel_db: np.ndarray
    pitch_hz: np.ndarray
    pitch_median_hz: float
    pitch_range_semitones: float
    voiced_fraction: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_flatness: float
    onset_rate_per_second: float
    mel_profile: np.ndarray
    warnings: tuple[str, ...]


def find_inputs(input_dir: Path = INPUT_DIR) -> dict[str, Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    files = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    matched: dict[str, Path] = {}
    for tradition in LABELS:
        candidates = [path for path in files if path.stem.casefold() == tradition.casefold()]
        if len(candidates) != 1:
            names = ", ".join(path.name for path in candidates) or "none"
            raise ValueError(
                f"Expected exactly one {tradition} file named {tradition} "
                f"with a supported audio extension; found {len(candidates)} ({names})."
            )
        matched[tradition] = candidates[0]

    unexpected = [path.name for path in files if path not in matched.values()]
    if unexpected:
        raise ValueError(
            "Remove or rename extra audio files in the input folder: "
            + ", ".join(sorted(unexpected))
        )
    return matched


def _pitch_summary(pitch_hz: np.ndarray) -> tuple[float, float, float]:
    valid = pitch_hz[np.isfinite(pitch_hz) & (pitch_hz > 0)]
    voiced_fraction = float(valid.size / pitch_hz.size) if pitch_hz.size else 0.0
    if valid.size < 20 or voiced_fraction < 0.05:
        return math.nan, math.nan, voiced_fraction
    semitones = 12 * np.log2(valid)
    return (
        float(np.median(valid)),
        float(np.percentile(semitones, 95) - np.percentile(semitones, 5)),
        voiced_fraction,
    )


def analyze_recording(tradition: str, path: Path) -> Recording:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"{path.name} is empty or contains invalid samples.")
    if np.max(np.abs(audio)) < 1e-6:
        raise ValueError(f"{path.name} is silent.")

    audio, _ = librosa.effects.trim(
        audio, top_db=35, frame_length=FFT_SIZE, hop_length=HOP_LENGTH
    )
    duration = audio.size / SAMPLE_RATE
    if duration < 5:
        raise ValueError(f"{path.name} has less than five seconds of usable audio.")

    rms = librosa.feature.rms(
        y=audio, frame_length=FFT_SIZE, hop_length=HOP_LENGTH
    )[0]
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=64
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    spectrum = np.abs(
        librosa.stft(audio, n_fft=FFT_SIZE, hop_length=HOP_LENGTH)
    )
    pitch_hz, _, voiced_probability = librosa.pyin(
        audio,
        sr=SAMPLE_RATE,
        fmin=65,
        fmax=1000,
        frame_length=FFT_SIZE,
        hop_length=HOP_LENGTH,
    )
    pitch_hz[voiced_probability < 0.7] = np.nan
    pitch_median, pitch_range, voiced_fraction = _pitch_summary(pitch_hz)
    centroid = librosa.feature.spectral_centroid(
        S=spectrum, sr=SAMPLE_RATE
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        S=spectrum, sr=SAMPLE_RATE
    )
    flatness = librosa.feature.spectral_flatness(S=spectrum)
    onsets = librosa.onset.onset_detect(
        y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, units="time"
    )

    profile = np.mean(mel, axis=1)
    profile /= max(float(profile.sum()), 1e-12)
    warnings = []
    if float(np.mean(np.abs(audio) >= 0.999)) > 0.01:
        warnings.append("Potential clipping")
    if not np.isfinite(pitch_median):
        warnings.append("Reliable pitch could not be estimated")

    return Recording(
        tradition=tradition,
        path=path,
        duration_seconds=duration,
        audio=audio,
        rms_mean=float(np.mean(rms)),
        mel_db=mel_db,
        pitch_hz=pitch_hz,
        pitch_median_hz=pitch_median,
        pitch_range_semitones=pitch_range,
        voiced_fraction=voiced_fraction,
        spectral_centroid_hz=float(np.mean(centroid)),
        spectral_bandwidth_hz=float(np.mean(bandwidth)),
        spectral_flatness=float(np.mean(flatness)),
        onset_rate_per_second=float(onsets.size / duration),
        mel_profile=profile,
        warnings=tuple(warnings),
    )


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else math.nan


def save_visualization(recordings: list[Recording], path: Path) -> None:
    fig, axes = plt.subplots(len(recordings), 3, figsize=(15, 10), layout="constrained")
    for row, recording in enumerate(recordings):
        stride = max(1, recording.audio.size // 30000)
        sample_times = np.arange(0, recording.audio.size, stride) / SAMPLE_RATE
        axes[row, 0].plot(
            sample_times, recording.audio[::stride], color="#254b70", linewidth=0.5
        )
        axes[row, 0].set(title=f"{recording.tradition}: waveform", xlabel="Time (s)")

        axes[row, 1].imshow(
            recording.mel_db,
            origin="lower",
            aspect="auto",
            extent=[0, recording.duration_seconds, 0, 64],
            cmap="magma",
            vmin=-80,
            vmax=0,
        )
        axes[row, 1].set(title="Mel spectrogram", xlabel="Time (s)", ylabel="Mel band")

        pitch_times = librosa.frames_to_time(
            np.arange(recording.pitch_hz.size), sr=SAMPLE_RATE, hop_length=HOP_LENGTH
        )
        axes[row, 2].plot(
            pitch_times, recording.pitch_hz, ".", markersize=1, color="#9a4e00"
        )
        axes[row, 2].set(
            title="Estimated pitch",
            xlabel="Time after trimming (s)",
            ylabel="Frequency (Hz)",
            ylim=(0, 1000),
        )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _format(value: float, suffix: str = "") -> str:
    return f"{value:.2f}{suffix}" if np.isfinite(value) else "Not estimated"


def write_report(recordings: list[Recording], path: Path) -> None:
    rows = []
    for recording in recordings:
        rows.append(
            "| {tradition} | {duration} | {pitch} | {range} | {voiced} | "
            "{energy} | {centroid} | {flatness} | {onsets} |".format(
                tradition=recording.tradition,
                duration=_format(recording.duration_seconds, " s"),
                pitch=_format(recording.pitch_median_hz, " Hz"),
                range=_format(recording.pitch_range_semitones, " st"),
                voiced=_format(recording.voiced_fraction * 100, "%"),
                energy=_format(recording.rms_mean),
                centroid=_format(recording.spectral_centroid_hz, " Hz"),
                flatness=_format(recording.spectral_flatness),
                onsets=_format(recording.onset_rate_per_second, " /s"),
            )
        )

    pairs = sorted(
        (
            (
                cosine_similarity(first.mel_profile, second.mel_profile),
                first.tradition,
                second.tradition,
            )
            for first, second in combinations(recordings, 2)
        ),
        reverse=True,
    )
    pair_rows = [
        f"| {first} | {second} | {_format(similarity)} |"
        for similarity, first, second in pairs
    ]
    warning_lines = [
        f"- **{recording.tradition}:** {warning}"
        for recording in recordings
        for warning in recording.warnings
    ]
    if not warning_lines:
        warning_lines = ["- No automatic audio-quality warnings."]

    text = f"""# Acoustic pattern report

Three recordings were analyzed after trimming leading and trailing silence.
The mel-profile similarity compares each recording's average distribution of
energy across mel-frequency bands. Higher values indicate more similar broad
spectral profiles; they do not mean the recordings sound identical.

## Feature summary

| Tradition | Duration | Median pitch | Pitch range | Voiced frames | Mean RMS energy | Spectral centroid | Spectral flatness | Onsets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Broad spectral-profile similarity

| Recording A | Recording B | Cosine similarity |
|---|---|---:|
{chr(10).join(pair_rows)}

## Quality notes

{chr(10).join(warning_lines)}

## Visualization

![Waveforms, mel spectrograms, and estimated pitch tracks](sound_patterns.png)

## Interpretation

These measurements describe these three files only. RMS energy depends on
recording gain and is not a loudness comparison. A single recording for
each tradition cannot show that a sound pattern is characteristic of a
religion. Performer, language, melody, room acoustics, microphone, and
recording quality can all affect the measurements. Pitch estimates may also
be unreliable for polyphonic, noisy, or strongly reverberant recordings.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    files = find_inputs()
    recordings = [
        analyze_recording(tradition, files[tradition]) for tradition in LABELS
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_visualization(recordings, OUTPUT_DIR / "sound_patterns.png")
    write_report(recordings, OUTPUT_DIR / "report.md")
    print(f"Analysis complete: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()

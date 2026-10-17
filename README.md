# Sacred Resonance

Compare acoustic patterns in three vocal recordings: one each from Hinduism, Christianity, and Islam.

## Use

1. Put exactly three audio files in `input/`, named for their traditions:
   - `hinduism.wav`
   - `christianity.wav`
   - `islam.wav`
2. Run `python analyze_audio.py`.
3. Read `output/report.md` and view `output/sound_patterns.png`.

WAV, FLAC, MP3, M4A, and OGG files are supported. Names are case-insensitive, and the extension can vary. Use clear recordings with little background sound and, where possible, similar durations and recording conditions.

Only files in `input/` are analyzed. The older files in `data/incoming/` were left untouched and are not part of the new workflow.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install ".[test]"
```

## What the analysis shows

The report summarizes duration, pitch, energy, spectral shape, and onset activity, then compares average mel-frequency spectral profiles. The visualization shows each recording’s waveform, spectrogram, and pitch track.

These are descriptive sound measurements, not a classifier or a test of religious identity. With one recording per tradition, observed differences may reflect the performer, language, melody, room, microphone, or recording quality rather than the tradition. Pitch may not be measurable in every recording.

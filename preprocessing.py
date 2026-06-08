import mne
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional


NOTCH_FREQ = 50
NOTCH_HARMONICS = (2, 3, 4)
BANDPASS_LOW = 8.0
BANDPASS_HIGH = 30.0
DEFAULT_TMIN = -0.5
DEFAULT_TMAX = 3.5
DEFAULT_BASELINE = (None, 0)
EPOCH_REJECT_THRESHOLD = 150e-6


def load_raw(filepath: Union[str, Path],
             preload: bool = True,
             eog_channels: Optional[List[str]] = None) -> mne.io.BaseRaw:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"EEG data file not found: {filepath}")
    suffix = filepath.suffix.lower()
    if suffix == ".bdf":
        raw = mne.io.read_raw_bdf(str(filepath), preload=preload, verbose=False)
    elif suffix in (".fif", ".fif.gz"):
        raw = mne.io.read_raw_fif(str(filepath), preload=preload, verbose=False)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Only .bdf and .fif are supported."
        )
    if eog_channels:
        raw.set_channel_types(
            {ch: "eog" for ch in eog_channels if ch in raw.ch_names}, verbose=False
        )
    return raw


def apply_notch_filter(raw: mne.io.BaseRaw,
                       freq: float = NOTCH_FREQ,
                       harmonics: tuple = NOTCH_HARMONICS) -> mne.io.BaseRaw:
    notch_freqs = np.arange(freq, raw.info["sfreq"] / 2, freq)
    if harmonics:
        for h in harmonics:
            harmonic_freq = freq * h
            if harmonic_freq < raw.info["sfreq"] / 2:
                notch_freqs = np.append(notch_freqs, harmonic_freq)
    notch_freqs = np.sort(np.unique(notch_freqs))
    raw.notch_filter(notch_freqs, fir_design="fir_win", verbose=False)
    return raw


def apply_bandpass_filter(raw: mne.io.BaseRaw,
                          l_freq: float = BANDPASS_LOW,
                          h_freq: float = BANDPASS_HIGH) -> mne.io.BaseRaw:
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design="fir_win",
        method="fir",
        verbose=False,
    )
    return raw


def extract_events(raw: mne.io.BaseRaw,
                   event_id: Optional[Dict[str, int]] = None,
                   stim_channel: Optional[str] = None) -> tuple:
    events = mne.find_events(raw, stim_channel=stim_channel, verbose=False)
    if events.size == 0:
        raise ValueError(
            "No events found in the raw data. Check stim_channel or event markers."
        )
    if event_id is not None:
        valid_codes = set(event_id.values())
        mask = np.isin(events[:, 2], list(valid_codes))
        events = events[mask]
        if events.size == 0:
            raise ValueError(
                f"No events matching event_id {event_id} found in the data."
            )
    return events, event_id


def create_epochs(raw: mne.io.BaseRaw,
                  events: np.ndarray,
                  event_id: Optional[Dict[str, int]] = None,
                  tmin: float = DEFAULT_TMIN,
                  tmax: float = DEFAULT_TMAX,
                  baseline: tuple = DEFAULT_BASELINE,
                  reject: Optional[dict] = None,
                  flat: Optional[dict] = None) -> mne.Epochs:
    if reject is None:
        reject = dict(eeg=EPOCH_REJECT_THRESHOLD)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        flat=flat,
        preload=True,
        verbose=False,
    )
    return epochs


def preprocess_pipeline(
    filepath: Union[str, Path],
    event_id: Optional[Dict[str, int]] = None,
    stim_channel: Optional[str] = None,
    notch_freq: float = NOTCH_FREQ,
    notch_harmonics: tuple = NOTCH_HARMONICS,
    l_freq: float = BANDPASS_LOW,
    h_freq: float = BANDPASS_HIGH,
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
    baseline: tuple = DEFAULT_BASELINE,
    reject_threshold: float = EPOCH_REJECT_THRESHOLD,
    eog_channels: Optional[List[str]] = None,
) -> mne.Epochs:
    print(f"[1/5] Loading raw data from: {filepath}")
    raw = load_raw(filepath, preload=True, eog_channels=eog_channels)
    print(f"  -> {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz, "
          f"{raw.times[-1]:.1f} s")

    print(f"[2/5] Applying notch filter @ {notch_freq} Hz (harmonics: {notch_harmonics})")
    raw = apply_notch_filter(raw, freq=notch_freq, harmonics=notch_harmonics)

    print(f"[3/5] Applying bandpass filter [{l_freq}-{h_freq}] Hz")
    raw = apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)

    print("[4/5] Extracting events and creating epochs")
    events, resolved_event_id = extract_events(
        raw, event_id=event_id, stim_channel=stim_channel
    )
    print(f"  -> Found {len(events)} events")
    if resolved_event_id:
        for label, code in resolved_event_id.items():
            count = np.sum(events[:, 2] == code)
            print(f"     '{label}' (code {code}): {count} epochs")

    reject = dict(eeg=reject_threshold)
    epochs = create_epochs(
        raw,
        events,
        event_id=resolved_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
    )
    print(f"[5/5] Epochs ready: {len(epochs)} trials, "
          f"{len(epochs.ch_names)} channels, {epochs.get_data().shape[2]} samples/trial")
    return epochs

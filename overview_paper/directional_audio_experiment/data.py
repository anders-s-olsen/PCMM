"""Prepare paired clean/noisy directional-audio data without downloads.

Three distinct MiniLibriMix speakers follow one joint, continuously active
conversation schedule.  Their clean McVAMPIRE render is reused verbatim for
the no-driving-noise and car-noise conditions.  Original utterance IDs and
the measured noise samples are disjoint across train/test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import re
import uuid
import wave
import numpy as np
from scipy import signal
from scipy.io import loadmat, wavfile


MODEL_NAMES = ["complex_acg", "uniform_vmvm", "complex_gaussian",
               "complex_bingham", "complex_watson"]
CONDITION_NAMES = ("no_driving_noise", "car_noise")


@dataclass
class Config:
    minilibrimix_root: str = ""
    mcvampire_root: str = ""
    output_dir: str = "output"
    seed: int = 321
    speaker_ids: list[str] | None = None
    display_seat_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 5])
    source_positions: list[int] = field(default_factory=lambda: [1, 2, 3])
    mouth_orientation_degrees: int = 0
    microphone_ids: list[int] = field(default_factory=lambda: [1, 4, 5, 7])
    reference_channel: int = 0  # zero-based within the selected microphones
    sample_rate: int = 8000
    n_fft: int = 1024
    hop_length: int = 256
    n_frequencies: int = 6
    frequency_min_hz: float = 250.
    frequency_max_hz: float = 3500.
    amplitude_threshold_db: float = -40.
    amplitude_reference_percentile: float = 95.
    absolute_amplitude_floor: float = 1e-10
    min_channel_relative_amplitude: float = 1e-5
    min_train_observations: int = 150
    min_test_observations: int = 1
    min_frequencies_per_frame: int = 4
    train_seconds: float = 45.
    test_seconds: float = 30.
    conversation_turn_seconds: list[float] = field(default_factory=lambda: [1.8, 4.2])
    target_two_speaker_fraction: float = .08
    target_three_speaker_fraction: float = .02
    max_two_speaker_fraction: float = .10
    max_three_speaker_fraction: float = .05
    overlap_target_tolerance: float = .015
    airtime_balance_tolerance: float = .05
    received_power_balance_tolerance_db: float = .05
    driving_noise_files: list[str] = field(
        default_factory=lambda: ["80kph.wav", "90kph.wav", "100kph.wav"])
    driving_noise_snr_db: float = 10.
    # Legacy fields retained so old local configs can be read.  Preparation
    # no longer uses independent on/off schedules or amplitude gating.
    speech_on_seconds: list[float] = field(default_factory=lambda: [1.8, 4.2])
    speech_off_seconds: list[float] = field(default_factory=lambda: [.8, 2.2])
    source_gains_db: list[float] = field(default_factory=lambda: [0., 0., 0.])
    train_utterance_fraction: float = .6
    min_utterances_per_split: int = 2
    models: list[str] = field(default_factory=lambda: MODEL_NAMES.copy())
    component_counts: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    fit_restarts: int = 3
    fit_tolerance: float = 1e-5
    fit_max_iterations: int = 2000
    fit_patience: int = 25
    fit_learning_rate: float = .1
    fit_threads: int = 8
    device: str = "cpu"
    alignment_min_overlap: int = 30
    alignment_iterations: int = 30
    alignment_initializations: int = 3
    consensus_max_iterations: int = 3
    consensus_label_tolerance: float = .001
    consensus_min_cluster_frames: int = 12
    figure_seconds: float = 18.
    export_audio: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        path = Path(path).resolve()
        raw = json.loads(path.read_text())
        cfg = cls(**raw)
        for key in ["minilibrimix_root", "mcvampire_root", "output_dir"]:
            value = getattr(cfg, key)
            if value:
                p = Path(value).expanduser()
                setattr(cfg, key, str((path.parent / p).resolve() if not p.is_absolute() else p))
        cfg.validate()
        return cfg

    def validate(self) -> None:
        S = len(self.source_positions)
        if S != 3:
            raise ValueError("This one-figure scaffold fixes three true source positions.")
        if len(set(self.source_positions)) != S or any(s not in range(1, 9) for s in self.source_positions):
            raise ValueError("Use three different McVAMPIRE seat IDs in 1..8.")
        if len(self.display_seat_ids) != 4 or len(set(self.display_seat_ids)) != 4:
            raise ValueError("display_seat_ids must contain four different measured seats.")
        if any(s not in range(1, 9) for s in self.display_seat_ids):
            raise ValueError("Displayed McVAMPIRE seat IDs must be in 1..8.")
        if not set(self.source_positions).issubset(self.display_seat_ids):
            raise ValueError("All occupied source positions must be among the four displayed seats.")
        if len(set(self.display_seat_ids) - set(self.source_positions)) != 1:
            raise ValueError("Exactly one of the four displayed seats must be unoccupied.")
        if len(set(self.microphone_ids)) != len(self.microphone_ids) or any(m not in range(1, 15) for m in self.microphone_ids):
            raise ValueError("Microphone IDs are unique ONE-BASED indices in 1..14.")
        if len(self.microphone_ids) != 4:
            raise ValueError("This experiment uses exactly four measured microphones.")
        if not 0 <= self.reference_channel < len(self.microphone_ids):
            raise ValueError("reference_channel indexes the selected microphone list from zero.")
        if self.speaker_ids is not None and (len(self.speaker_ids) != S or len(set(map(str, self.speaker_ids))) != S):
            raise ValueError("speaker_ids must contain three different actual speaker IDs.")
        if len(self.source_gains_db) != S:
            raise ValueError("source_gains_db must have one value per source.")
        if not 0 < self.frequency_min_hz < self.frequency_max_hz < self.sample_rate / 2:
            raise ValueError("Frequency interval must be strictly between DC and Nyquist.")
        if not 0 < self.hop_length <= self.n_fft or self.n_fft % 2:
            raise ValueError("Use a positive hop <= an even n_fft.")
        if self.n_frequencies < 2 or not 1 <= self.min_frequencies_per_frame <= self.n_frequencies:
            raise ValueError("Need >=2 selected frequencies and a feasible frame coverage requirement.")
        if not self.component_counts or any(not isinstance(k, int) or k < 1 for k in self.component_counts) or 3 not in self.component_counts:
            raise ValueError("Positive candidate counts must include K=3 for the spatial panels.")
        if self.fit_restarts < 1 or not self.models or len(set(self.models)) != len(self.models) or not set(self.models).issubset(MODEL_NAMES):
            raise ValueError("Invalid restarts or unknown model family.")
        if not math.isfinite(self.fit_tolerance) or self.fit_tolerance <= 0:
            raise ValueError("fit_tolerance must be finite and positive.")
        if not math.isfinite(self.fit_learning_rate) or self.fit_learning_rate <= 0:
            raise ValueError("fit_learning_rate must be finite and positive.")
        if any(not isinstance(value, int) or value < 1 for value in
               [self.fit_max_iterations, self.fit_patience, self.fit_threads]):
            raise ValueError("Fit iteration, patience, and thread counts must be positive integers.")
        for interval in [self.conversation_turn_seconds, self.speech_on_seconds,
                         self.speech_off_seconds]:
            if len(interval) != 2 or not 0 < interval[0] <= interval[1]:
                raise ValueError("Activity intervals must be positive [min,max].")
        fractions = [self.target_two_speaker_fraction, self.target_three_speaker_fraction,
                     self.max_two_speaker_fraction, self.max_three_speaker_fraction]
        if any(not math.isfinite(value) or not 0 <= value < 1 for value in fractions):
            raise ValueError("Overlap targets and caps must be finite fractions in [0,1).")
        if self.target_two_speaker_fraction > self.max_two_speaker_fraction:
            raise ValueError("The two-speaker target cannot exceed its hard cap.")
        if self.target_three_speaker_fraction > self.max_three_speaker_fraction:
            raise ValueError("The three-speaker target cannot exceed its hard cap.")
        if self.target_two_speaker_fraction + self.target_three_speaker_fraction >= 1:
            raise ValueError("Overlap targets must leave some single-speaker frames.")
        if not 0 <= self.overlap_target_tolerance < 1 or not 0 < self.airtime_balance_tolerance < 1:
            raise ValueError("Schedule tolerances must be fractions in their valid range.")
        if not math.isfinite(self.received_power_balance_tolerance_db) or self.received_power_balance_tolerance_db <= 0:
            raise ValueError("received_power_balance_tolerance_db must be finite and positive.")
        if not math.isfinite(self.driving_noise_snr_db):
            raise ValueError("driving_noise_snr_db must be finite.")
        if not self.driving_noise_files or len(set(self.driving_noise_files)) != len(self.driving_noise_files):
            raise ValueError("List one or more unique McVAMPIRE driving-noise WAV files.")
        if not 0 < self.train_utterance_fraction < 1 or min(self.train_seconds, self.test_seconds) <= 0:
            raise ValueError("Invalid utterance split fraction or recording duration.")
        if min(self.train_seconds, self.test_seconds) * self.sample_rate < self.n_fft:
            raise ValueError("Recordings must be at least one complete STFT window long.")
        if not 0 < self.amplitude_reference_percentile <= 100 or self.amplitude_threshold_db > 0:
            raise ValueError("Use a percentile in (0,100] and a nonpositive relative dB threshold.")
        if self.absolute_amplitude_floor <= 0 or not 0 <= self.min_channel_relative_amplitude < 1:
            raise ValueError("Invalid amplitude floor.")
        if min(self.min_train_observations, self.min_utterances_per_split) < 1:
            raise ValueError("Observation/utterance minima must be positive.")
        if self.alignment_min_overlap < 3 or min(self.alignment_iterations, self.alignment_initializations) < 1:
            raise ValueError("Alignment needs >=3 shared frames and positive iteration/initialization counts.")
        if not isinstance(self.consensus_max_iterations, int) or self.consensus_max_iterations < 1:
            raise ValueError("consensus_max_iterations must be a positive integer.")
        if (not math.isfinite(self.consensus_label_tolerance)
                or not 0 <= self.consensus_label_tolerance < 1):
            raise ValueError("consensus_label_tolerance must be a fraction in [0,1).")
        if (not isinstance(self.consensus_min_cluster_frames, int)
                or self.consensus_min_cluster_frames < 2):
            raise ValueError("consensus_min_cluster_frames must be an integer >=2.")


@dataclass(frozen=True)
class Utterance:
    speaker_id: str
    utterance_id: str
    path: str
    seconds: float


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Return floating audio without per-channel or per-recording normalization."""
    fs, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        if audio.dtype == np.uint8:
            audio = (audio.astype(np.float64) - 128.) / 128.
        else:
            # scipy left-justifies 24-bit PCM in int32, so this is also correct
            # for the native 24-bit McVAMPIRE IR WAVs.
            audio = audio.astype(np.float64) / float(2 ** (8 * audio.dtype.itemsize - 1))
    else:
        audio = audio.astype(np.float64)
    if not np.isfinite(audio).all():
        raise ValueError(f"Nonfinite audio in {path}")
    return int(fs), audio


def _duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as stream:
            return stream.getnframes() / stream.getframerate()
    except (wave.Error, EOFError):
        fs, audio = read_wav(path)
        return len(audio) / fs


def inventory_speech(root: str | Path) -> list[Utterance]:
    """Parse native mixed filenames but load ONLY separate s1/s2 sources.

    Same original utterance can occur in different packaged mixtures. Keep
    one copy (the longest available version) BEFORE splitting recordings.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Extract MiniLibriMix and set its root: {root}")
    records = {}
    for path in sorted(root.rglob("*.wav")):
        match = re.fullmatch(r"s([1-9][0-9]*)", path.parent.name)
        if not match:
            continue  # never train from mix_clean, mix_both or noise directories
        parts = path.stem.split("_")
        slot = int(match.group(1)) - 1
        if slot >= len(parts) or not re.fullmatch(r"\d+-\d+-\d+", parts[slot]):
            raise ValueError(f"Cannot recover original utterance identity from {path}")
        uid = parts[slot]
        record = Utterance(uid.split("-")[0], uid, str(path.resolve()), _duration(path))
        if uid not in records or record.seconds > records[uid].seconds:
            records[uid] = record
    if not records:
        raise FileNotFoundError("No MiniLibriMix s1/s2 WAVs found. Point to the extracted archive, not metadata alone.")
    return sorted(records.values(), key=lambda r: r.utterance_id)


def partition_utterances(records: list[Utterance], cfg: Config) -> tuple[list[str], list[list[Utterance]], list[list[Utterance]]]:
    groups: dict[str, list[Utterance]] = {}
    for record in records:
        groups.setdefault(record.speaker_id, []).append(record)
    minimum = cfg.min_utterances_per_split
    candidates = [s for s in groups if len(groups[s]) >= 2 * minimum]
    candidates.sort(key=lambda s: (-sum(u.seconds for u in groups[s]), s))
    ids = [str(s) for s in cfg.speaker_ids] if cfg.speaker_ids else candidates[:3]
    if len(ids) < 3:
        raise ValueError("Need three speakers with enough distinct utterances for both splits.")
    train, test = [], []
    for speaker in ids:
        if speaker not in candidates:
            raise ValueError(f"Speaker {speaker} needs at least {2 * minimum} unique utterances.")
        utterances = sorted(groups[speaker], key=lambda r: r.utterance_id)
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, int(speaker)]))
        utterances = [utterances[i] for i in rng.permutation(len(utterances))]
        split = min(len(utterances) - minimum, max(minimum, round(len(utterances) * cfg.train_utterance_fraction)))
        train.append(utterances[:split])
        test.append(utterances[split:])
    train_ids = {u.utterance_id for group in train for u in group}
    test_ids = {u.utterance_id for group in test for u in group}
    if train_ids & test_ids:
        raise AssertionError("Train/test source-utterance leakage.")
    return ids, train, test


def resample_audio(audio: np.ndarray, old_fs: int, new_fs: int, *, impulse_response: bool = False) -> np.ndarray:
    if old_fs == new_fs:
        return audio.copy()
    divisor = math.gcd(old_fs, new_fs)
    result = signal.resample_poly(audio, new_fs // divisor, old_fs // divisor, axis=0)
    if impulse_response:
        # Preserve the discrete filter's low-frequency gain after decimation.
        result *= old_fs / new_fs
    return result


def load_mcvampire(cfg: Config) -> tuple[list[np.ndarray], dict]:
    root = Path(cfg.mcvampire_root)
    matches = sorted(root.rglob("geometry.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one geometry.mat below {root}; found {len(matches)}.")
    geometry = loadmat(matches[0], simplify_cells=True)
    microphones = np.stack([np.asarray(d["pos"], float) for d in geometry["mic"]])
    seats = np.stack([np.asarray(d["centerpos"], float) for d in geometry["mouth"]])
    bank = []
    for seat in range(1, 9):
        name = f"ir_mouth_pos{seat}_{cfg.mouth_orientation_degrees}.wav"
        files = sorted(root.rglob(name))
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one {name} below {root}, found {len(files)}.")
        fs, impulse = read_wav(files[0])
        if impulse.ndim != 2 or impulse.shape[1] != 14:
            raise ValueError(f"Expected native (samples,14) IR in {files[0]}.")
        selected = impulse[:, np.asarray(cfg.microphone_ids) - 1]
        selected = resample_audio(selected, fs, cfg.sample_rate, impulse_response=True)
        if len(selected) > cfg.n_fft:
            raise ValueError("n_fft must cover the measured IR; increase n_fft instead of truncating the IR.")
        bank.append(selected)
    return bank, {"microphones_xyz": microphones.tolist(), "seat_centers_xyz": seats.tolist(),
                  "note": "Measured mouth-simulator centers; acoustic mouth point is about 7 cm forward. Top views omit height."}


def _speech_stream(records: list[Utterance], cfg: Config) -> tuple[np.ndarray, list[dict]]:
    """Concatenate normalized utterances once, retaining their provenance."""
    fs = cfg.sample_rate
    clips = []
    offsets = []
    total = 0
    for record in records:
        old_fs, audio = read_wav(record.path)
        if audio.ndim != 1:
            raise ValueError(f"Expected monaural MiniLibriMix source: {record.path}")
        audio = resample_audio(audio, old_fs, fs)
        peak = np.max(np.abs(audio), initial=0.)
        if peak < 1e-10:
            continue
        # Remove only near-silent ends. No waveform is reused across split/seat.
        keep = np.flatnonzero(abs(audio) > peak * 10 ** (-45 / 20))
        audio = audio[keep[0]:keep[-1] + 1]
        rms = np.sqrt(np.mean(audio ** 2))
        audio *= .08 / max(rms, 1e-12)
        # Smooth joins between different utterances inside an on episode too.
        fade = min(round(.015 * fs), len(audio) // 2)
        if fade:
            ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
            audio[:fade] *= ramp
            audio[-fade:] *= ramp[::-1]
        clips.append(audio)
        offsets.append({"utterance_id": record.utterance_id, "start": total, "end": total + len(audio)})
        total += len(audio)
    if not clips:
        raise ValueError("No usable speech in a selected speaker/split.")
    return np.concatenate(clips), offsets


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, bool), (1, 1))
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(changes[::2], changes[1::2]))


def _frame_activity(sample_active: np.ndarray, cfg: Config) -> np.ndarray:
    """Map sample support to the exact unpadded SciPy STFT window grid."""
    sample_active = np.asarray(sample_active, bool)
    starts = np.arange(0, sample_active.shape[1] - cfg.n_fft + 1, cfg.hop_length)
    cumulative = np.pad(np.cumsum(sample_active, axis=1, dtype=np.int64), ((0, 0), (1, 0)))
    return cumulative[:, starts + cfg.n_fft] > cumulative[:, starts]


def _conversation_baseline(n_samples: int, cfg: Config,
                           rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    """Make balanced, contiguous single-speaker turns with no scheduled gaps."""
    mean_turn = float(np.mean(cfg.conversation_turn_seconds))
    cycles = max(1, round(n_samples / cfg.sample_rate / mean_turn / 3))
    n_blocks = 3 * cycles
    lower, upper = cfg.conversation_turn_seconds
    while n_blocks > 3 and n_samples / cfg.sample_rate / n_blocks < lower:
        n_blocks -= 3
    while n_samples / cfg.sample_rate / n_blocks > upper:
        n_blocks += 3
    boundaries = np.rint(np.linspace(0, n_samples, n_blocks + 1)).astype(int)
    orders: list[int] = []
    previous = -1
    for _ in range(n_blocks // 3):
        candidates = [p for p in [(0, 1, 2), (0, 2, 1), (1, 0, 2),
                                  (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                      if p[0] != previous]
        order = candidates[int(rng.integers(len(candidates)))]
        orders.extend(order)
        previous = order[-1]
    active = np.zeros((3, n_samples), bool)
    turns = []
    for i, speaker in enumerate(orders):
        first, last = int(boundaries[i]), int(boundaries[i + 1])
        active[speaker, first:last] = True
        turns.append({"speaker_index": int(speaker), "start_seconds": first / cfg.sample_rate,
                      "end_seconds": last / cfg.sample_rate})
    if not active.any(axis=0).all():
        raise AssertionError("The baseline conversation unexpectedly contains scheduled silence.")
    return active, turns


def _single_speaker_runs(source_active: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    counts = source_active.sum(axis=0)
    result = {s: [] for s in range(source_active.shape[0])}
    for first, last in _true_runs(counts == 1):
        speaker = int(np.flatnonzero(source_active[:, first])[0])
        result[speaker].append((first, last))
    return result


def _reserve_frame_interval(runs: list[tuple[int, int]], n_frames: int,
                            reserved: list[tuple[int, int]], guard: int) -> int:
    """Choose a contiguous interval wholly inside a single-speaker run."""
    choices = []
    for first, last in runs:
        for candidate in range(first, last - n_frames + 1):
            end = candidate + n_frames
            if all(end + guard <= used_first or candidate >= used_last + guard
                   for used_first, used_last in reserved):
                distance = min(candidate - first, last - end)
                choices.append((distance, candidate))
    if not choices:
        raise ValueError(
            "Conversation turns are too short to place the requested overlap regions. "
            "Increase the recording/turn duration or reduce the overlap targets."
        )
    # Prefer the deepest interior placement, which keeps window leakage away
    # from ordinary speaker transitions. The tie break is deterministic.
    _, first = max(choices, key=lambda item: (item[0], -item[1]))
    reserved.append((first, first + n_frames))
    return first


def joint_conversation_schedule(cfg: Config, seconds: float,
                                seed_offset: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Construct and audit one continuously active low-overlap schedule.

    Overlap targets are enforced after mapping sample supports through the
    exact STFT windows. Equal-size insertions for all three baseline speakers
    keep both total and single-speaker airtime balanced.
    """
    n_samples = round(seconds * cfg.sample_rate)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, seed_offset]))
    sample_active, turns = _conversation_baseline(n_samples, cfg, rng)
    baseline_frames = _frame_activity(sample_active, cfg)
    n_frames = baseline_frames.shape[1]
    baseline_counts = baseline_frames.sum(axis=0)
    if np.any(baseline_counts == 0) or np.any(baseline_counts > 2):
        raise AssertionError("Balanced baseline turns should yield only one/two-speaker STFT frames.")

    desired_two = round(cfg.target_two_speaker_fraction * n_frames)
    desired_three = round(cfg.target_three_speaker_fraction * n_frames)
    minimum_region_frames = max(1, cfg.n_fft // cfg.hop_length)
    extra_two = max(0, desired_two - int(np.count_nonzero(baseline_counts == 2)))

    # Three equal insertions (one per baseline speaker) preserve airtime. For
    # extremely short fixtures where three resolvable insertions cannot fit,
    # omit that overlap order rather than introduce a large speaker imbalance.
    two_region_frames = (max(minimum_region_frames, round(extra_two / 3))
                         if extra_two >= 3 * minimum_region_frames else 0)
    three_region_frames = (max(minimum_region_frames, round(desired_three / 3))
                           if desired_three >= 3 * minimum_region_frames else 0)
    runs = _single_speaker_runs(baseline_frames)
    reserved: dict[int, list[tuple[int, int]]] = {s: [] for s in range(3)}
    insertions = []

    def insert(order: int, n_affected: int) -> None:
        if not n_affected:
            return
        for base_speaker in range(3):
            first_frame = _reserve_frame_interval(
                runs[base_speaker], n_affected, reserved[base_speaker], minimum_region_frames)
            # An interval [a,b) with these endpoints intersects exactly frames
            # first_frame .. first_frame+n_affected-1 and no adjacent frame.
            first_sample = (first_frame - 1) * cfg.hop_length + cfg.n_fft
            last_sample = (first_frame + n_affected) * cfg.hop_length
            if not 0 <= first_sample < last_sample <= n_samples:
                raise AssertionError("Derived overlap insertion lies outside the recording.")
            if order == 2:
                added = [(base_speaker + 1) % 3]
            elif order == 3:
                added = [s for s in range(3) if s != base_speaker]
            else:
                raise AssertionError("Only two/three-speaker insertions are supported.")
            sample_active[added, first_sample:last_sample] = True
            insertions.append({"order": order, "base_speaker_index": base_speaker,
                               "added_speaker_indices": added,
                               "first_frame": first_frame,
                               "last_frame_exclusive": first_frame + n_affected,
                               "start_seconds": first_sample / cfg.sample_rate,
                               "end_seconds": last_sample / cfg.sample_rate})

    # Reserve the larger regions first to make placement robust for short data.
    if two_region_frames >= three_region_frames:
        insert(2, two_region_frames); insert(3, three_region_frames)
    else:
        insert(3, three_region_frames); insert(2, two_region_frames)

    source_active = _frame_activity(sample_active, cfg)
    active_counts = source_active.sum(axis=0)
    count_values = {str(order): int(np.count_nonzero(active_counts == order))
                    for order in (1, 2, 3)}
    fractions = {order: count_values[str(order)] / n_frames for order in (1, 2, 3)}
    if np.any(active_counts == 0):
        raise AssertionError("The final STFT schedule contains silent frames.")
    if fractions[2] > cfg.max_two_speaker_fraction + 1e-12:
        raise ValueError(f"Two-speaker fraction {fractions[2]:.3%} exceeds its hard cap.")
    if fractions[3] > cfg.max_three_speaker_fraction + 1e-12:
        raise ValueError(f"Three-speaker fraction {fractions[3]:.3%} exceeds its hard cap.")
    target_errors = {
        "2": fractions[2] - cfg.target_two_speaker_fraction,
        "3": fractions[3] - cfg.target_three_speaker_fraction,
    }
    if (abs(target_errors["2"]) > cfg.overlap_target_tolerance or
            abs(target_errors["3"]) > cfg.overlap_target_tolerance):
        raise ValueError(
            "Discrete STFT schedule could not meet overlap targets within tolerance: "
            f"errors={target_errors}. Increase duration or overlap_target_tolerance."
        )

    frame_airtime = source_active.sum(axis=1)
    single_counts = np.array([
        np.count_nonzero((active_counts == 1) & source_active[s]) for s in range(3)])
    sample_airtime = sample_active.sum(axis=1)

    def relative_spread(values: np.ndarray) -> float:
        return float(np.ptp(values) / max(np.mean(values), 1.))

    frame_spread = relative_spread(frame_airtime)
    single_spread = relative_spread(single_counts)
    sample_spread = relative_spread(sample_airtime)
    if max(frame_spread, single_spread, sample_spread) > cfg.airtime_balance_tolerance:
        raise ValueError(
            "Joint schedule does not meet the speaker-airtime balance tolerance: "
            f"frame={frame_spread:.3%}, single={single_spread:.3%}, sample={sample_spread:.3%}."
        )
    report = {
        "n_samples": n_samples, "n_frames": n_frames,
        "active_speaker_frame_counts": count_values,
        "active_speaker_frame_fractions": {key: count_values[key] / n_frames for key in count_values},
        "target_frame_fractions": {"2": cfg.target_two_speaker_fraction,
                                   "3": cfg.target_three_speaker_fraction},
        "target_fraction_errors": target_errors,
        "source_frame_airtime_counts": frame_airtime.astype(int).tolist(),
        "source_sample_airtime_counts": sample_airtime.astype(int).tolist(),
        "single_speaker_frame_counts": single_counts.astype(int).tolist(),
        "frame_airtime_relative_spread": frame_spread,
        "sample_airtime_relative_spread": sample_spread,
        "single_speaker_airtime_relative_spread": single_spread,
        "turns": turns, "overlap_insertions": insertions,
        "activity_definition": "A source is active in an STFT frame iff its scheduled sample support intersects that unpadded analysis window.",
    }
    return sample_active, source_active, report


def _scheduled_source(records: list[Utterance], sample_active: np.ndarray,
                      cfg: Config) -> tuple[np.ndarray, list[dict]]:
    """Fill scheduled regions sequentially without looping or split leakage."""
    fs = cfg.sample_rate
    stream, offsets = _speech_stream(records, cfg)
    out = np.zeros(len(sample_active), dtype=float)
    consumed = 0
    events = []
    for first, last in _true_runs(sample_active):
        n = last - first
        if consumed + n > len(stream):
            raise ValueError(
                f"Not enough independent source audio: need >{(consumed+n)/fs:.1f}s, "
                f"have {len(stream)/fs:.1f}s. Shorten train_seconds/test_seconds, "
                "reduce scheduled overlap, or choose speakers with more utterances. "
                "The code will NOT repeat recordings or leak test material."
            )
        excerpt = stream[consumed:consumed + n].copy()
        fade = min(round(.015 * fs), n // 2)
        if fade:
            ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
            excerpt[:fade] *= ramp
            excerpt[-fade:] *= ramp[::-1]
        out[first:last] = excerpt
        used = [d["utterance_id"] for d in offsets if d["start"] < consumed + n and d["end"] > consumed]
        events.append({"start_seconds": first / fs, "end_seconds": last / fs, "utterance_ids": used})
        consumed += n
    return out, events


def render_scene(records: list[list[Utterance]], bank: list[np.ndarray], cfg: Config,
                 seconds: float, seed_offset: int) -> tuple[np.ndarray, np.ndarray, list,
                                                              np.ndarray, np.ndarray, dict]:
    sample_active, source_active, schedule_report = joint_conversation_schedule(
        cfg, seconds, seed_offset)
    sources, event_lists = [], []
    for s, group in enumerate(records):
        dry, events = _scheduled_source(group, sample_active[s], cfg)
        dry *= 10 ** (cfg.source_gains_db[s] / 20)
        sources.append(dry)
        event_lists.append(events)
    images = []
    for s, dry in enumerate(sources):
        ir = bank[cfg.source_positions[s] - 1]
        images.append(np.stack([signal.fftconvolve(dry, ir[:, m])[:len(dry)] for m in range(ir.shape[1])]))
    images = np.stack(images)  # (source,microphone,sample)
    return images.sum(axis=0), images, event_lists, np.stack(sources), source_active, schedule_report


def _stft(audio: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return signal.stft(audio, fs=cfg.sample_rate, window="hann", nperseg=cfg.n_fft,
                       noverlap=cfg.n_fft - cfg.hop_length, nfft=cfg.n_fft,
                       boundary=None, padded=False, axis=-1)


def select_frequency_bins(cfg: Config) -> np.ndarray:
    all_hz = np.fft.rfftfreq(cfg.n_fft, 1 / cfg.sample_rate)
    targets = np.linspace(cfg.frequency_min_hz, cfg.frequency_max_hz, cfg.n_frequencies)
    bins = np.unique([int(np.argmin(abs(all_hz - target))) for target in targets])
    if len(bins) != cfg.n_frequencies:
        raise ValueError("Too many requested frequencies for this FFT/range.")
    return bins


def amplitude_mask(x: np.ndarray, thresholds: np.ndarray, cfg: Config) -> np.ndarray:
    """Legacy amplitude-gate helper; paired preparation deliberately skips it."""
    rms = np.sqrt(np.mean(abs(x) ** 2, axis=-1))
    # Phase-only models need each channel's phase to be defined. Use a common
    # conservative channel floor so model families see identical observations.
    channel_floor = np.maximum(cfg.absolute_amplitude_floor,
                               rms * cfg.min_channel_relative_amplitude)
    return (rms > thresholds[:, None]) & (np.min(abs(x), axis=-1) > channel_floor)


def _noise_path(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.is_file():
        return direct.resolve()
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one driving-noise file {filename!r} below {root}; found {len(matches)}.")
    return matches[0].resolve()


def load_driving_noise_pair(cfg: Config) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load matched-speed, sample-disjoint train/test noise segments.

    Every configured recording contributes equally (to one sample) to both
    splits. Training takes the beginning and test takes the end, with overlap
    forbidden. This avoids confounding the held-out split with car speed.
    """
    root = Path(cfg.mcvampire_root)
    recordings = []
    paths = []
    for filename in cfg.driving_noise_files:
        path = _noise_path(root, filename)
        fs, audio = read_wav(path)
        if audio.ndim != 2 or audio.shape[1] != 14:
            raise ValueError(f"Expected native (samples,14) driving noise in {path}.")
        selected = audio[:, np.asarray(cfg.microphone_ids) - 1]
        recordings.append(resample_audio(selected, fs, cfg.sample_rate))
        paths.append(path)

    def allocations(total: int) -> list[int]:
        quotient, remainder = divmod(total, len(recordings))
        return [quotient + (i < remainder) for i in range(len(recordings))]

    n_train = round(cfg.train_seconds * cfg.sample_rate)
    n_test = round(cfg.test_seconds * cfg.sample_rate)
    train_sizes, test_sizes = allocations(n_train), allocations(n_test)
    train_parts, test_parts, segments = [], [], []
    for path, audio, train_size, test_size in zip(paths, recordings, train_sizes, test_sizes):
        if train_size + test_size > len(audio):
            raise ValueError(
                f"Disjoint train/test requests need {(train_size + test_size) / cfg.sample_rate:.2f}s "
                f"from {path.name}, but it contains only {len(audio) / cfg.sample_rate:.2f}s. "
                "Add matched driving_noise_files or shorten the scenes."
            )
        test_first = len(audio) - test_size
        train_parts.append(audio[:train_size])
        test_parts.append(audio[test_first:])
        segments.append({
            "path": str(path),
            "train_start_seconds": 0.,
            "train_end_seconds": train_size / cfg.sample_rate,
            "test_start_seconds": test_first / cfg.sample_rate,
            "test_end_seconds": len(audio) / cfg.sample_rate,
            "separation_seconds": (test_first - train_size) / cfg.sample_rate,
        })
    train = np.concatenate(train_parts, axis=0).T
    test = np.concatenate(test_parts, axis=0).T
    if train.shape != (len(cfg.microphone_ids), n_train) or test.shape != (len(cfg.microphone_ids), n_test):
        raise AssertionError("Driving-noise assembly returned an unexpected shape.")
    return train, test, {
        "files": [str(path) for path in paths],
        "segments": segments,
        "split_policy": "Each speed contributes equally; train uses each file's beginning and test its non-overlapping end.",
    }


def _isolated_array_powers(images: np.ndarray) -> np.ndarray:
    """Time-average isolated received powers over all selected microphones."""
    return np.mean(np.square(images), axis=(1, 2))


def _power_balance_report(train_before: np.ndarray, train_after: np.ndarray,
                          test_after: np.ndarray, gains: np.ndarray) -> dict:
    def describe(powers: np.ndarray) -> dict:
        geometric = float(np.exp(np.mean(np.log(np.maximum(powers, 1e-30)))))
        relative_db = 10 * np.log10(np.maximum(powers, 1e-30) / geometric)
        return {
            "powers": powers.tolist(),
            "power_fractions": (powers / powers.sum()).tolist(),
            "relative_db": relative_db.tolist(),
            "max_minus_min_db": float(np.ptp(relative_db)),
        }
    return {
        "definition": "Mean squared isolated received waveform over all selected microphones and all samples.",
        "training_before_equalization": describe(train_before),
        "equalization_gains_linear": gains.tolist(),
        "equalization_gains_db": (20 * np.log10(gains)).tolist(),
        "training_after_equalization": describe(train_after),
        "test_after_frozen_equalization": describe(test_after),
    }


def _snr_db(images: np.ndarray, noise: np.ndarray) -> float:
    speech_power = float(np.mean(np.sum(np.square(images), axis=0)))
    noise_power = float(np.mean(np.square(noise)))
    return float(10 * np.log10(max(speech_power, 1e-30) / max(noise_power, 1e-30)))


def _split_arrays(mixture: np.ndarray, images: np.ndarray, noise: np.ndarray,
                  source_active: np.ndarray, bins: np.ndarray, cfg: Config) -> dict:
    hz, times, mixture_stft = _stft(mixture, cfg)
    _, image_times, image_stft = _stft(images, cfg)
    _, noise_times, noise_stft = _stft(noise, cfg)
    if not (np.array_equal(times, image_times) and np.array_equal(times, noise_times)):
        raise AssertionError("Paired STFT grids differ unexpectedly.")
    x = np.moveaxis(mixture_stft[:, bins, :], 0, -1).astype(np.complex64)
    image_power = np.mean(np.abs(image_stft) ** 2, axis=1)
    source_energy = image_power[:, bins, :].astype(np.float32)
    broadband_source_energy = image_power.sum(axis=1).astype(np.float32)
    noise_energy = np.mean(np.abs(noise_stft[:, bins, :]) ** 2, axis=0).astype(np.float32)
    if source_active.shape != (3, len(times)):
        raise AssertionError("Schedule and STFT frame grids do not agree.")
    speech_active = source_active.any(axis=0)
    if not speech_active.all():
        raise AssertionError("The paired experiment must contain no silent STFT frames.")
    valid = np.ones((len(bins), len(times)), dtype=bool)
    denominator = np.maximum(source_energy.sum(axis=0, keepdims=True), 1e-30)
    return {
        "x": x,
        "source_energy": source_energy,
        "source_power_fraction": (source_energy / denominator).astype(np.float32),
        "broadband_source_energy": broadband_source_energy,
        "noise_energy": noise_energy,
        "source_active": source_active.astype(bool),
        "speech_active": speech_active.astype(bool),
        "valid": valid,
        "times": times.astype(np.float64),
        "frequencies_hz": hz[bins].astype(np.float64),
    }


def prepare_dataset(cfg: Config, *, overwrite: bool = False) -> Path:
    cfg.validate()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prepared.json").exists() and not overwrite:
        raise FileExistsError(f"{out}/prepared.json exists; use --overwrite to deliberately regenerate.")
    if (out / "prepared.json").exists():
        (out / "prepared.json").unlink()
    records = inventory_speech(cfg.minilibrimix_root)
    speaker_ids, train_records, test_records = partition_utterances(records, cfg)
    bank, geometry = load_mcvampire(cfg)
    (train_mix, train_images, train_events, train_dry, train_source_active,
     train_schedule) = render_scene(train_records, bank, cfg, cfg.train_seconds, 11)
    (test_mix, test_images, test_events, test_dry, test_source_active,
     test_schedule) = render_scene(test_records, bank, cfg, cfg.test_seconds, 29)

    # Derive one scalar correction per seat from isolated TRAIN received power,
    # then freeze it for held-out speech and both acoustic conditions.
    train_power_before = _isolated_array_powers(train_images)
    if np.any(train_power_before <= 0):
        raise ValueError("A rendered training source has zero received power.")
    target_power = float(np.exp(np.mean(np.log(train_power_before))))
    source_equalization = np.sqrt(target_power / train_power_before)
    train_images *= source_equalization[:, None, None]
    test_images *= source_equalization[:, None, None]
    train_dry *= source_equalization[:, None]
    test_dry *= source_equalization[:, None]
    train_mix = train_images.sum(axis=0)
    test_mix = test_images.sum(axis=0)
    train_power_after = _isolated_array_powers(train_images)
    test_power_after = _isolated_array_powers(test_images)
    training_spread_db = float(np.ptp(10 * np.log10(train_power_after)))
    if training_spread_db > cfg.received_power_balance_tolerance_db:
        raise AssertionError(
            f"Training seat powers still differ by {training_spread_db:.4f} dB after equalization.")

    raw_train_noise, raw_test_noise, noise_provenance = load_driving_noise_pair(cfg)
    train_speech_power = float(np.mean(np.sum(np.square(train_images), axis=0)))
    raw_train_noise_power = float(np.mean(np.square(raw_train_noise)))
    if raw_train_noise_power <= 0:
        raise ValueError("Selected McVAMPIRE training noise has zero power.")
    noise_gain = math.sqrt(train_speech_power /
                           (raw_train_noise_power * 10 ** (cfg.driving_noise_snr_db / 10)))
    train_noise = raw_train_noise * noise_gain
    test_noise = raw_test_noise * noise_gain
    train_noisy_mix = train_mix + train_noise
    test_noisy_mix = test_mix + test_noise

    # One final TRAIN-derived scalar keeps exported floating WAVs convenient;
    # it is frozen across splits, sources, and clean/noisy conditions.
    output_gain = .8 / max(np.max(np.abs(train_mix)), np.max(np.abs(train_noisy_mix)), 1e-12)
    train_images *= output_gain; test_images *= output_gain
    train_dry *= output_gain; test_dry *= output_gain
    train_mix *= output_gain; test_mix *= output_gain
    train_noise *= output_gain; test_noise *= output_gain
    train_noisy_mix *= output_gain; test_noisy_mix *= output_gain

    bins = select_frequency_bins(cfg)
    conditions = {
        "no_driving_noise": {
            "train": _split_arrays(train_mix, train_images, np.zeros_like(train_noise),
                                   train_source_active, bins, cfg),
            "test": _split_arrays(test_mix, test_images, np.zeros_like(test_noise),
                                  test_source_active, bins, cfg),
        },
        "car_noise": {
            "train": _split_arrays(train_noisy_mix, train_images, train_noise,
                                   train_source_active, bins, cfg),
            "test": _split_arrays(test_noisy_mix, test_images, test_noise,
                                  test_source_active, bins, cfg),
        },
    }
    source_reference = np.maximum(
        np.percentile(conditions["no_driving_noise"]["train"]["broadband_source_energy"],
                      95, axis=1), 1e-20)
    for condition in CONDITION_NAMES:
        for split in ("train", "test"):
            item = conditions[condition][split]
            item["activity"] = np.clip(
                np.sqrt(item["broadband_source_energy"] / source_reference[:, None]), 0., 1.).astype(np.float32)
            np.savez_compressed(out / f"{split}_{condition}.npz", **item)

    display_indices = np.asarray(cfg.display_seat_ids) - 1
    steering = np.stack([
        np.fft.rfft(bank[index], n=cfg.n_fft, axis=0)[bins] for index in display_indices])
    np.savez_compressed(out / "reference.npz", steering=steering.astype(np.complex64),
                        seat_ids=np.asarray(cfg.display_seat_ids, dtype=np.int16),
                        microphone_ids=np.asarray(cfg.microphone_ids, dtype=np.int16),
                        fft_bins=bins.astype(np.int32),
                        frequencies_hz=conditions["no_driving_noise"]["train"]["frequencies_hz"])
    power_report = _power_balance_report(
        train_power_before, _isolated_array_powers(train_images),
        _isolated_array_powers(test_images), source_equalization)
    noise_report = {
        **noise_provenance,
        "target_training_snr_db": cfg.driving_noise_snr_db,
        "scaling_definition": "Sum of isolated clean-source received powers divided by measured multichannel noise power, averaged over all training samples/channels.",
        "training_derived_noise_gain": noise_gain,
        "realized_training_snr_db": _snr_db(train_images, train_noise),
        "realized_test_snr_db_with_frozen_gain": _snr_db(test_images, test_noise),
    }
    metadata = {
        "schema_version": 2, "preparation_id": uuid.uuid4().hex,
        "config": asdict(cfg), "speaker_ids": speaker_ids,
        "condition_names": list(CONDITION_NAMES),
        "display_seat_ids": cfg.display_seat_ids,
        "source_positions": cfg.source_positions, "microphone_ids": cfg.microphone_ids,
        "empty_display_seat_id": next(iter(set(cfg.display_seat_ids) - set(cfg.source_positions))),
        "geometry": geometry, "common_training_output_gain": output_gain,
        "source_power_balance": power_report, "driving_noise": noise_report,
        "schedule": {"train": train_schedule, "test": test_schedule},
        "train_utterances": [[asdict(u) for u in group] for group in train_records],
        "test_utterances": [[asdict(u) for u in group] for group in test_records],
        "train_events": train_events, "test_events": test_events,
        "train_used_utterance_ids": [sorted({uid for event in events for uid in event["utterance_ids"]}) for events in train_events],
        "test_used_utterance_ids": [sorted({uid for event in events for uid in event["utterance_ids"]}) for events in test_events],
        "truth_power_definition": "Mean isolated clean-source STFT power over selected microphones; identical across paired conditions.",
        "validity_definition": "No amplitude gate. All train and test STFT frames are valid because the joint schedule is continuously speech-active.",
        "observations_train": [int(conditions["no_driving_noise"]["train"]["valid"].shape[1])] * len(bins),
        "observations_test": [int(conditions["no_driving_noise"]["test"]["valid"].shape[1])] * len(bins),
        "frequencies_hz": conditions["no_driving_noise"]["train"]["frequencies_hz"].tolist(),
        "prepared_files": {condition: {split: f"{split}_{condition}.npz"
                                        for split in ("train", "test")}
                           for condition in CONDITION_NAMES},
        "scope": "Paired clean/car-noise conditions from one fixed three-speaker/four-seat measured configuration; custom utterance-disjoint split, not the official MiniLibriMix benchmark split."
    }
    temporary_metadata = out / f".prepared-{uuid.uuid4().hex}.json"
    temporary_metadata.write_text(json.dumps(metadata, indent=2))
    temporary_metadata.replace(out / "prepared.json")
    if cfg.export_audio:
        audio = {
            "no_driving_noise": {"train": train_mix, "test": test_mix},
            "car_noise": {"train": train_noisy_mix, "test": test_noisy_mix},
        }
        for condition in CONDITION_NAMES:
            for split in ("train", "test"):
                wavfile.write(out / f"{split}_{condition}_mixture.wav", cfg.sample_rate,
                              audio[condition][split].T.astype(np.float32))
        for name, images in [("train", train_images), ("test", test_images)]:
            for s in range(3):
                wavfile.write(out / f"{name}_source{s+1}_reference.wav", cfg.sample_rate,
                              images[s, cfg.reference_channel].astype(np.float32))
    return out


def load_prepared(directory: str | Path) -> tuple[dict, dict[str, dict[str, dict]], dict]:
    directory = Path(directory)
    metadata = json.loads((directory / "prepared.json").read_text())
    if metadata.get("schema_version") != 2:
        raise ValueError("This runner requires paired prepared-data schema version 2; regenerate the data.")
    conditions: dict[str, dict[str, dict]] = {}
    for condition in CONDITION_NAMES:
        conditions[condition] = {}
        for split in ("train", "test"):
            path = directory / metadata["prepared_files"][condition][split]
            with np.load(path, allow_pickle=False) as archive:
                conditions[condition][split] = {key: archive[key] for key in archive.files}
    with np.load(directory / "reference.npz", allow_pickle=False) as archive:
        reference = {key: archive[key] for key in archive.files}
    return metadata, conditions, reference

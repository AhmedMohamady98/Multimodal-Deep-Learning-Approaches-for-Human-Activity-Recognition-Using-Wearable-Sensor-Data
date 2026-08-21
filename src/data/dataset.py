"""
HARMES dataset loader and windowing pipeline (v3 — bug-fixed).

Each item returned: (imu, audio, humidity, label)
  imu:      torch.float32  (window_size, 12)
  audio:    torch.float32  (N_audio,)   -- variable length, AST handles internally
  humidity: torch.float32  (window_size, 1)
  label:    torch.long     scalar in [0, NUM_CLASSES-1]

CHANGES vs v2 (May 2026 bug fix — supervisor-flagged):
  1. Windows are now created over the ENTIRE recording timeline, not only over
     labeled activity segments. This is required for the model to learn a
     continuous-stream classifier rather than a closed-set highlight-reel one.
  2. A "Null" class (ID 15, NUM_CLASSES = 16) represents periods between
     labeled activities.
  3. Each window is assigned the MAJORITY label by sample overlap. Edge windows
     spanning an activity boundary get whichever side dominates.
"""

import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

IMU_SR   = 50       # Hz
AUDIO_SR = 44_100   # Hz

WINDOW_SIZE = 100   # samples at 50 Hz = 2 seconds (default; overridden by scripts)
STEP        = 50    # 50% overlap

# ── Activity name registry ────────────────────────────────────────────────────

ACTIVITY_TO_ID = {
    "Washing hands":        0,
    "Floor cleaning":       1,
    "Window cleaning":      2,
    "Vacuum cleaning":      3,
    "Brushing teeth":       4,
    "Washing dishes":       5,
    "Putting away dishes":  6,
    "Making tea":           7,
    "Cutting vegetables":   8,
    "Creaming hands":       9,
    "Drinking":             10,
    "Watering plants":      11,
    "Cleaning table":       12,
    "Disinfecting hands":   13,
    "Cleaning dishwasher":  14,
    "Null":                 15,    # ← NEW: "no activity" / transitions between activities
}

ID_TO_ACTIVITY = {v: k for k, v in ACTIVITY_TO_ID.items()}
NUM_CLASSES    = len(ACTIVITY_TO_ID)   # = 16
NULL_LABEL_ID  = ACTIVITY_TO_ID["Null"]

# All known label variants → canonical name (lowercase key for case-insensitive lookup)
_ALIASES = {
    "washing hands":              "Washing hands",
    "floor cleaning":             "Floor cleaning",
    "window cleaning":            "Window cleaning",
    "vacuum cleaning":            "Vacuum cleaning",
    "brushing teeth":             "Brushing teeth",
    "washing dishes":             "Washing dishes",
    "putting away dishes":        "Putting away dishes",
    "putting away the dishes":    "Putting away dishes",
    "making tea":                 "Making tea",
    "cutting vegetables":         "Cutting vegetables",
    "creaming hands":             "Creaming hands",
    "cream hands":                "Creaming hands",
    "drinking":                   "Drinking",
    "watering plants":            "Watering plants",
    "watering plant":             "Watering plants",
    "cleaning table":             "Cleaning table",
    "disinfecting hands":         "Disinfecting hands",
    "cleaning dishwasher":        "Cleaning dishwasher",
    "cleaning out the dishwasher": "Cleaning dishwasher",
}

def _normalize(name: str) -> str | None:
    """Map any label variant to canonical name; None if unknown."""
    return _ALIASES.get(name.strip().lower())

# ── Label parsing ─────────────────────────────────────────────────────────────

def parse_labels(labels_df: pd.DataFrame) -> list[tuple[str, float, float]]:
    """
    Convert a Labels DataFrame into a list of (canonical_name, start_ms, end_ms).
    Skips unknown activities and any unpaired start/end rows.
    """
    instances = []
    pending: dict[str, float] = {}

    for _, row in labels_df.iterrows():
        if row["Type"] not in ("start", "end"):
            continue
        canonical = _normalize(str(row["Description"]))
        if canonical is None:
            continue
        ts = float(row["ts_sync"])
        if row["Type"] == "start":
            pending[canonical] = ts
        elif row["Type"] == "end" and canonical in pending:
            instances.append((canonical, pending.pop(canonical), ts))

    return instances

# ── Sample-level label assignment ─────────────────────────────────────────────

def build_sample_labels(
    ts_imu: np.ndarray,
    instances: list[tuple[str, float, float]],
) -> np.ndarray:
    """
    Return a per-sample label array, same length as ts_imu.
    Samples falling inside [start_ms, end_ms] of a labeled activity get that
    activity's ID. All other samples get NULL_LABEL_ID.

    If a sample falls inside two activities (shouldn't happen in HARMES but is
    handled defensively), the later-listed instance wins.
    """
    labels = np.full(len(ts_imu), NULL_LABEL_ID, dtype=np.int64)
    for activity, start_ms, end_ms in instances:
        mask = (ts_imu >= start_ms) & (ts_imu <= end_ms)
        labels[mask] = ACTIVITY_TO_ID[activity]
    return labels

# ── Window extraction ─────────────────────────────────────────────────────────

_IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

def extract_windows(
    rec: dict,
    window_size: int = WINDOW_SIZE,
    step: int = STEP,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Slide a window over the ENTIRE recording timeline (not just labeled segments).

    For each window, the label is the majority-overlap class — i.e. the class
    that occupies the most IMU samples in that window. The Null class
    (ID = 15) is included as a candidate, so windows lying between activities
    are correctly labeled "Null".

    Returns list of (imu, audio, humidity, label_id):
      imu:      (window_size, 12)
      audio:    (N_audio,)   -- length varies slightly with window position
      humidity: (window_size, 1)
      label_id: int in [0, NUM_CLASSES-1]
    """
    imu_l_df = rec["IMU_L"]
    imu_r_df = rec["IMU_R"]
    bme_df   = rec["BME280"]
    audio    = rec["Audio"]          # numpy (M,) float32

    ts_imu   = imu_l_df["ts_sync"].values        # (N_l,) ms
    imu_l    = imu_l_df[_IMU_COLS].values        # (N_l, 6)
    imu_r    = imu_r_df[_IMU_COLS].values        # (N_r, 6)  may differ from N_l
    humidity = bme_df["humid"].values            # (N_h,)    may differ from N_l

    # IMU_L / IMU_R / BME280 can have slightly different sample counts due to
    # drift or dropped samples (observed in HARMES). Trim everything to the
    # common length before windowing — otherwise imu_r[idx] / humidity[idx]
    # IndexError on the last window when ts_imu is the longest stream.
    N = min(len(ts_imu), len(imu_l), len(imu_r), len(humidity))
    if N < window_size:
        return []
    ts_imu   = ts_imu[:N]
    imu_l    = imu_l[:N]
    imu_r    = imu_r[:N]
    humidity = humidity[:N]

    instances     = parse_labels(rec["Labels"])
    sample_labels = build_sample_labels(ts_imu, instances)

    windows  = []

    # Audio timing: convert ts_imu (ms) → audio sample index, relative to start of
    # this recording. ts_imu[0] is the recording's first IMU timestamp; we treat
    # audio[0] as aligned with ts_imu[0]. (For concatenated multi-part recordings,
    # _concat_recordings has already shifted ts_imu to a single timeline and audio
    # is the concatenation of the parts in matching order.)
    ts0_ms = float(ts_imu[0])

    for w_start in range(0, N - window_size + 1, step):
        idx = np.arange(w_start, w_start + window_size)

        # Majority-overlap label — includes Null as a candidate
        label_id = int(np.bincount(sample_labels[idx], minlength=NUM_CLASSES).argmax())

        imu_win = np.concatenate([imu_l[idx], imu_r[idx]], axis=1)  # (W, 12)
        hum_win = humidity[idx].reshape(-1, 1)                       # (W, 1)

        # Audio aligned to window timestamps relative to recording start
        a_start = int((ts_imu[idx[0]]  - ts0_ms) * AUDIO_SR / 1000)
        a_end   = int((ts_imu[idx[-1]] - ts0_ms) * AUDIO_SR / 1000)
        audio_win = audio[a_start:a_end]

        if len(audio_win) < 100:   # guard against degenerate edge cases
            continue

        windows.append((imu_win, audio_win, hum_win, label_id))

    return windows

# ── Split-recording concatenation ─────────────────────────────────────────────

def _concat_recordings(recs: list[dict]) -> dict:
    """
    Concatenate split recording parts into one recording dict.
    Offsets ts_sync in each part so timestamps are monotonically increasing.

    NOTE: audio is concatenated in matching order. Within each part the audio
    is assumed sample-aligned with the IMU starting at the part's first IMU
    timestamp. `extract_windows` computes audio offsets RELATIVE to ts_imu[0]
    of the concatenated recording, which works because the cumulative IMU
    timeline matches the cumulative audio sample count (both proportional to
    elapsed real time).
    """
    recs = sorted(recs, key=lambda r: str(r["rec_id"]))

    def _concat_df(key: str) -> pd.DataFrame:
        parts, offset = [], 0.0
        for r in recs:
            df = r[key].copy()
            df["ts_sync"] = df["ts_sync"] + offset
            offset = float(df["ts_sync"].iloc[-1]) + 1000.0 / IMU_SR
            parts.append(df)
        return pd.concat(parts, ignore_index=True)

    def _concat_labels() -> pd.DataFrame:
        parts, offset = [], 0.0
        for r in recs:
            df = r["Labels"].copy()
            df["ts_sync"] = df["ts_sync"] + offset
            last = float(r["IMU_L"]["ts_sync"].iloc[-1])
            offset += last + 1000.0 / IMU_SR
            parts.append(df)
        return pd.concat(parts, ignore_index=True)

    return {
        "IMU_L":  _concat_df("IMU_L"),
        "IMU_R":  _concat_df("IMU_R"),
        "BME280": _concat_df("BME280"),
        "Audio":  np.concatenate([r["Audio"] for r in recs]),
        "Labels": _concat_labels(),
        "rec_id": str(recs[0]["rec_id"]),
    }

# ── Dataset ───────────────────────────────────────────────────────────────────

class HARMESDataset(Dataset):
    """
    PyTorch Dataset for HARMES.

    Args:
        participant_ids: list of int, e.g. [1, 2, 3]
        data_dir:        path to folder containing participant_XX.pkl files
        window_size:     IMU samples per window (default 100 = 2 s at 50 Hz)
        step:            stride between windows (default 50 = 50% overlap)
    """

    def __init__(
        self,
        participant_ids: list[int],
        data_dir: str | Path,
        window_size: int = WINDOW_SIZE,
        step: int = STEP,
    ):
        data_dir = Path(data_dir)
        self._windows: list[tuple] = []

        for pid in participant_ids:
            pkl_path = data_dir / f"participant_{pid:02d}.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"Missing participant file: {pkl_path}")

            with open(pkl_path, "rb") as f:
                recordings: list[dict] = pickle.load(f)

            # Group by 4-char base ID (e.g. "0101"); exclude free-form (xx04)
            grouped: dict[str, list[dict]] = {}
            for rec in recordings:
                base_id = str(rec["rec_id"])[:4]
                if base_id.endswith("04"):
                    continue
                grouped.setdefault(base_id, []).append(rec)

            for base_id, parts in sorted(grouped.items()):
                rec = parts[0] if len(parts) == 1 else _concat_recordings(parts)
                self._windows.extend(
                    extract_windows(rec, window_size=window_size, step=step)
                )

        print(f"HARMESDataset: {len(self._windows)} windows "
              f"from {len(participant_ids)} participants")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        imu, audio, humidity, label = self._windows[idx]
        return (
            torch.from_numpy(imu).float(),       # (W, 12)
            torch.from_numpy(audio).float(),     # (N_audio,)
            torch.from_numpy(humidity).float(),  # (W, 1)
            torch.tensor(label, dtype=torch.long),
        )

    def class_counts(self) -> dict[str, int]:
        """Return {activity_name: window_count} for inspection."""
        counts: dict[str, int] = {k: 0 for k in ACTIVITY_TO_ID}
        for _, _, _, label in self._windows:
            counts[ID_TO_ACTIVITY[label]] += 1
        return counts

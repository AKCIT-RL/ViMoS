"""
Music feature extraction utility for GENMO music-driven dance generation.

Extracts 35-dimensional music features at 30fps from raw audio files.
The feature vector is designed to match the AIST++ music feature format
used during training (encoded_music_dim=35 per configs/pipeline/dual_mode.yaml).

Feature layout (total 35 dims):
  - Beat strength envelope      :  1 dim   (onset strength / tempogram at beat freq)
  - MFCC (first 20 coefficients): 20 dims
  - Chroma (12 pitch classes)   : 12 dims
  - Spectral centroid           :  1 dim
  - Spectral bandwidth          :  1 dim
"""

from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_music_features(
    audio_path: Union[str, Path],
    target_fps: int = 30,
    duration: Optional[float] = None,
    sr: Optional[int] = None,
) -> np.ndarray:
    """Extract 35-dimensional music features from an audio file.

    Each output frame corresponds to one motion frame at ``target_fps``.

    Args:
        audio_path: Path to an audio file (WAV, MP3, FLAC, …).
        target_fps: Target frames per second for the output features.
                    Must match the model's motion fps (default 30).
        duration:   Optional clip duration in seconds. ``None`` uses the
                    full file.
        sr:         Sample rate to load at. ``None`` keeps the native rate.

    Returns:
        numpy array of shape ``(L, 35)`` and dtype ``float32``, where ``L``
        is the number of frames at ``target_fps``.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # ------------------------------------------------------------------
    # 1. Load audio
    # ------------------------------------------------------------------
    y, sr = librosa.load(str(audio_path), sr=sr, duration=duration, mono=True)

    hop_length = int(sr / target_fps)  # samples per output frame

    # ------------------------------------------------------------------
    # 2. Beat / onset strength  (1 dim)
    # ------------------------------------------------------------------
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Normalise to [0, 1]
    env_max = onset_env.max()
    if env_max > 0:
        onset_env = onset_env / env_max
    beat_feat = onset_env[:, None]  # (T, 1)

    # ------------------------------------------------------------------
    # 3. MFCC  (20 dims)
    # ------------------------------------------------------------------
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=20, hop_length=hop_length
    ).T  # (T, 20)

    # ------------------------------------------------------------------
    # 4. Chroma  (12 dims)
    # ------------------------------------------------------------------
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=hop_length, n_chroma=12
    ).T  # (T, 12)

    # ------------------------------------------------------------------
    # 5. Spectral centroid + bandwidth  (2 dims)
    # ------------------------------------------------------------------
    spec_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    ).T  # (T, 1)
    spec_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, hop_length=hop_length
    ).T  # (T, 1)

    # Normalise spectral features to [0, 1] using a fixed scale (robust to
    # differences across songs)
    nyquist = sr / 2.0
    spec_centroid = spec_centroid / nyquist   # centroid in [0, 1]
    spec_bandwidth = spec_bandwidth / nyquist  # bandwidth in [0, 1]

    # ------------------------------------------------------------------
    # 6. Align lengths  (librosa may produce T or T±1 across features)
    # ------------------------------------------------------------------
    L = min(
        beat_feat.shape[0],
        mfcc.shape[0],
        chroma.shape[0],
        spec_centroid.shape[0],
        spec_bandwidth.shape[0],
    )
    features = np.concatenate(
        [
            beat_feat[:L],       # (L,  1)
            mfcc[:L],            # (L, 20)
            chroma[:L],          # (L, 12)
            spec_centroid[:L],   # (L,  1)
            spec_bandwidth[:L],  # (L,  1)
        ],
        axis=1,
    ).astype(np.float32)        # (L, 35)

    assert features.shape[1] == 35, (
        f"Unexpected feature dimension: {features.shape[1]} (expected 35)"
    )
    return features


def load_music_embed(
    audio_path: Union[str, Path],
    target_fps: int = 30,
    duration: Optional[float] = None,
) -> torch.Tensor:
    """Convenience wrapper that returns a float32 torch tensor.

    Args:
        audio_path: Path to the audio file.
        target_fps: Frames per second for the returned tensor.
        duration:   Clip duration in seconds (None = full file).

    Returns:
        ``torch.FloatTensor`` of shape ``(L, 35)``.
    """
    features = extract_music_features(
        audio_path, target_fps=target_fps, duration=duration
    )
    return torch.from_numpy(features)

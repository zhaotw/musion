from typing import Tuple, Optional, List, Dict
from scipy.signal import find_peaks, lfilter, savgol_filter
import torch
import numpy as np
import pretty_midi

# MIDI note numbers for drum instruments
BD = 35
SD = 38
HH = 42
HH_OPEN = 46
TOMS = 47
CRASH = 49
RIDE = 51

FRAMES_PER_SECOND = 100

# Cache device availability to avoid repeated checks
_DEVICE_CACHE: Optional[str] = None


def _get_device() -> str:
    """Get and cache the device (cuda or cpu)."""
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        _DEVICE_CACHE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return _DEVICE_CACHE

def equal_loudness(audio: np.ndarray) -> np.ndarray:
    """
    等响度滤波器，等价于 essentia.standard.EqualLoudness() (44.1kHz 采样率)
    基于 ReplayGain 规范的 Yulewalk + Butterworth 滤波器
    """
    # Yulewalk filter (10th order IIR)
    by = np.array([
        0.05418656406430, -0.02911007808948, -0.00848709379851,
        -0.00851165645469, -0.00834990904936, 0.02245293253339,
        -0.02596338512915, 0.01624864962975, -0.00240879051584,
        0.00674613682247, -0.00187763777362
    ], dtype=np.float64)
    
    ay = np.array([
        1.0, -3.47845948550071, 6.36317777566148, -8.54751527471874,
        9.47693607801280, -8.81498681370155, 6.85401540936998,
        -4.39470996079559, 2.19611684890774, -0.75104302451432,
        0.13149317958808
    ], dtype=np.float64)
    
    # Butterworth highpass filter (2nd order, fc ≈ 38 Hz)
    bb = np.array([0.98500175787242, -1.97000351574484, 0.98500175787242], dtype=np.float64)
    ab = np.array([1.0, -1.96977855582618, 0.97022847566350], dtype=np.float64)
    
    # 级联应用两个滤波器
    y = lfilter(by, ay, audio.astype(np.float64))
    y = lfilter(bb, ab, y)
    
    return y.astype(np.float32)

def map_energy_to_velocity(
    energy: np.ndarray,
    max_db: float,
    min_velocity: int = 0,
    max_velocity: int = 127,
    use_log: bool = False
) -> np.ndarray:
    # Normalize energy to range [0, 1]
    energy = np.clip(np.array(energy), -90, max_db) + 90
    energy_min = 0
    energy_max = max_db + 90

    normalized_energy = (energy - energy_min) / (energy_max - energy_min)

    if use_log:
        normalized_energy = np.clip(normalized_energy, 0.33, 1)
        normalized_energy = normalized_energy ** 1.75

    return np.interp(normalized_energy, [0, 1], [min_velocity, max_velocity]).astype(int)

def get_peak_energies(loudness_values: List[float]) -> Tuple[np.ndarray, List[float]]:
    loudness_values = np.array(loudness_values) + 90
    loudness_values = np.clip(loudness_values, 0, 100)

    # use savitsky-golay smoothing
    loudness_values = savgol_filter(loudness_values, window_length=35, polyorder=1)

    peaks, data = find_peaks(loudness_values, distance=100, height=28, prominence=10)
    peak_energies = [loudness_values[p] for p in peaks]

    return peaks, peak_energies

def _get_loudness_torch(audio: np.ndarray) -> torch.Tensor:
    """
    GPU/torch path to compute per-frame loudness power.

    This approximates Essentia's Loudness as mean square power per windowed frame.
    
    Args:
        audio: Input audio signal as 1D numpy array
        
    Returns:
        Tensor of per-frame loudness power values (stays on GPU if available)
    """
    FRAME_SIZE = 1024
    HOP_SIZE = 441
    device = _get_device()
    
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    if audio.shape[0] < FRAME_SIZE:
        # Audio is too short, return empty tensor    
        return torch.empty(0, dtype=torch.float32, device=device)
    
    x = torch.as_tensor(audio, dtype=torch.float32, device=device)
    frames = x.unfold(0, FRAME_SIZE, HOP_SIZE)
    window = torch.hann_window(FRAME_SIZE, periodic=True, device=device)
    windowed = frames * window

    # Essentia Loudness is closer to frame energy (sum of squares) than mean power.
    # Using sum helps align dB scale with the CPU path.
    power = (windowed * windowed).sum(dim=1)
    return power  # Keep as tensor, don't detach yet

def get_loudness(
    audio: np.ndarray,
    threshold: Optional[float] = None,
    activity_threshold: Optional[float] = None
) -> Tuple[float, List[float]]:
    """
    Compute loudness values for audio input.
    
    Args:
        audio: Input audio signal (can be multi-channel)
        threshold: Minimum loudness threshold in dB
        activity_threshold: Activity detection threshold in dB
        
    Returns:
        Tuple of (max_db, loudness_list) where loudness_list contains dB values per frame
    """
    # Convert to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    audio = equal_loudness(audio)

    loudness_power = _get_loudness_torch(audio)
    
    if loudness_power.numel() == 0:
        return -90.0, []

    # Convert loudness values to decibels, ensuring no non-positive values
    device = loudness_power.device
    loudness_db = torch.where(
        loudness_power > 0,
        10 * torch.log10(loudness_power),
        torch.tensor(-90.0, device=device)
    )

    max_db = loudness_db.max().item()

    # Set anything below threshold to -90
    if threshold is not None:
        loudness_db = torch.where(
            loudness_db < threshold,
            torch.tensor(-90.0, device=device),
            loudness_db
        )

    if activity_threshold is not None:
        if not (loudness_db > activity_threshold).any():
            return -90.0, [-90.0] * len(loudness_db)

    return max_db, loudness_db.cpu().tolist()

def is_monotonic_neighbour(x, n, neighbour):
    for i in range(neighbour):
        if x[n - i] < x[n - i - 1]:
            return False
        if x[n + i] < x[n + i + 1]:
            return False

    return True

def nearest_onset_time(
    onsets: List[float],
    onset_time: float
) -> Tuple[Optional[int], Optional[float]]:
    """
    Find the nearest onset time in the loudness array and refine it.
    
    See Section III-D in [1] for the refinement algorithm.
    [1] Q. Kong, et al., High-resolution Piano Transcription 
        with Pedals by Regressing Onsets and Offsets Times, 2020.
    
    Args:
        onsets: List of loudness values per frame
        onset_time: Target onset time in seconds
        
    Returns:
        Tuple of (onset_idx, refined_onset_time) or (None, None) if not found
    """
    NEIGHBOUR_SIZE = 2
    
    idx = int(onset_time * FRAMES_PER_SECOND)
    
    # Limit window to neighbour_size samples either side of idx
    start_idx = max(0, idx - NEIGHBOUR_SIZE)
    end_idx = min(len(onsets), idx + NEIGHBOUR_SIZE)
    window = onsets[start_idx:end_idx]
    
    if len(window) == 0:
        return None, None
    
    window_onset_idx = np.argmax(window)
    onset_idx = start_idx + window_onset_idx

    if onsets[onset_idx] <= -90:
        return None, None

    # Refine onset time using parabolic interpolation
    shift = 0.0
    if onsets[onset_idx] > -55 and is_monotonic_neighbour(onsets, onset_idx, 1):
        if onset_idx > 0 and onset_idx < len(onsets) - 1:
            if onsets[onset_idx - 1] > onsets[onset_idx + 1]:
                denominator = onsets[onset_idx] - onsets[onset_idx + 1]
            else:
                denominator = onsets[onset_idx] - onsets[onset_idx - 1]
            if abs(denominator) > 1e-6:  # Avoid division by zero
                shift = (onsets[onset_idx + 1] - onsets[onset_idx - 1]) / denominator / 2
    
    refined_onset_time = (onset_idx + shift) / FRAMES_PER_SECOND
    
    return onset_idx, refined_onset_time

def estimate_velocity(
    combined_mid: pretty_midi.PrettyMIDI,
    res: Dict[str, np.ndarray],
    snare_sensitivity: bool = False
) -> None:
    """
    Estimate and refine velocity values for drum notes based on separated drum parts.
    
    Args:
        combined_mid: PrettyMIDI object containing drum notes
        res: Dictionary with separated drum parts ('kick', 'snare', 'toms', 'hh', 'crash', 'ride')
        snare_sensitivity: If True, set minimum velocity to 40 for snare notes
    """
    bd, sd, toms, hh, crash, ride = res['kick'], res['snare'], res['toms'], res['hh'], res['crash'], res['ride']

    default_threshold = None
    sd_max_db, sd_loudness = get_loudness(sd, threshold=default_threshold)
    toms_max_db, toms_loudness = get_loudness(toms, threshold=default_threshold)
    crash_max_db, crash_loudness = get_loudness(crash, threshold=default_threshold)
    ride_max_db, ride_loudness = get_loudness(ride, threshold=default_threshold)
    bd_max_db, bd_loudness = get_loudness(bd, threshold=default_threshold)
    hh_max_db, hihat_loudness = get_loudness(hh, threshold=default_threshold)
    max_db = max([bd_max_db, sd_max_db, toms_max_db, hh_max_db, crash_max_db, ride_max_db])

    bd_velocities = map_energy_to_velocity(bd_loudness, max_db, use_log=True)
    sd_velocities = map_energy_to_velocity(sd_loudness, max_db, use_log=True)
    hihat_velocities = map_energy_to_velocity(hihat_loudness, max_db)
    toms_velocities = map_energy_to_velocity(toms_loudness, max_db)
    ride_velocities = map_energy_to_velocity(ride_loudness, max_db)
    crash_velocities = map_energy_to_velocity(crash_loudness, max_db)

    crash_refraction_time = 0
    crash_refraction_period = 0.2
    crash_peaks, crash_peak_energies = get_peak_energies(crash_loudness)
    crash_peaks = np.array(crash_peaks)
    crash_peak_times = crash_peaks / 100

    def refine_note_by_loudness(loudness_list, velocities, note, snare_sensitivity=False):
        onset_idx, onset_time = nearest_onset_time(loudness_list, note.start)
        if onset_time is not None:
            note.start = onset_time
            note.end = note.start + 0.01
            note.velocity = max(velocities[max(0, onset_idx-1):min(len(velocities), onset_idx+2)])
            if note.velocity == 0 and snare_sensitivity:
                note.velocity = 40

    for n in combined_mid.instruments[0].notes:
        if n.pitch == BD:
            refine_note_by_loudness(bd_loudness, bd_velocities, n)
        if n.pitch == SD:
            refine_note_by_loudness(sd_loudness, sd_velocities, n, snare_sensitivity)
        if n.pitch == HH:
            refine_note_by_loudness(hihat_loudness, hihat_velocities, n)
        if n.pitch == TOMS:
            refine_note_by_loudness(toms_loudness, toms_velocities, n)
        if n.pitch == CRASH:
            ride_onset_idx, ride_onset_time = nearest_onset_time(ride_loudness, n.start)
            crash_onset_idx, crash_onset_time = nearest_onset_time(crash_loudness, n.start)

            if ride_onset_idx is not None:
                ride_velocity = max(ride_velocities[max(0, ride_onset_idx-1):min(len(ride_velocities), ride_onset_idx+2)])
            else:
                ride_velocity = None

            if crash_onset_idx is not None:
                crash_velocity = max(crash_velocities[max(0, crash_onset_idx-1):min(len(crash_velocities), crash_onset_idx+2)])
            else:
                crash_velocity = None

            if crash_velocity is not None and crash_onset_time > crash_refraction_time:
                n.velocity = crash_velocity
                if n.velocity == 0:
                    n.velocity = 40

                if len(crash_peaks) > 0:
                    # find closest crash peak to current crash_onset_idx
                    current_crash_peak_idx = np.argmin(np.abs(crash_peaks - crash_onset_idx))
                    if current_crash_peak_idx < len(crash_peaks) - 1:
                        # look ahead to the next crash peak and subtract 1 to allow leeway
                        next_crash_peak_time = crash_peak_times[current_crash_peak_idx + 1] - 1
                    else:
                        next_crash_peak_time = n.start + 2 # we've reached the end of the crash peaks

                    crash_refraction_time = next_crash_peak_time
                else:
                    crash_refraction_time = n.start + crash_refraction_period
            elif ride_velocity is not None:
                n.pitch = RIDE
                n.velocity = ride_velocity
                if n.velocity == 0:
                    n.velocity = 40
            else:
                n.velocity = 0

    # Post-process hihats to detect open vs closed hihats
    # Open hihats have continuously high loudness
    hihat_notes = [n for n in combined_mid.instruments[0].notes if n.pitch == HH]
    for i in range(len(hihat_notes) - 1):
        interval_start = hihat_notes[i].start
        interval_end = min(interval_start + 0.15, hihat_notes[i + 1].start)

        start_frame = int(interval_start * FRAMES_PER_SECOND)
        end_frame = int(interval_end * FRAMES_PER_SECOND)
        interval_loudness = hihat_loudness[start_frame:end_frame]
        
        if len(interval_loudness) == 0:
            continue
        
        # Normalize loudness values (add 90 to convert from dB scale)
        interval_loudness_normalized = [l + 90.0 for l in interval_loudness]
        min_loudness = min(interval_loudness_normalized)
        max_loudness = max(interval_loudness_normalized)
        
        # If minimum loudness is > 75% of maximum, it's likely an open hihat
        if max_loudness > 0 and min_loudness > (0.75 * max_loudness):
            hihat_notes[i].pitch = HH_OPEN

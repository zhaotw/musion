import os
from typing import Optional, Dict, List

import numpy as np
import pretty_midi
import madmom
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.signal import FramedSignalProcessor, SignalProcessor
from madmom.audio.spectrogram import LogarithmicFilteredSpectrogramProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor

from musion.utils.base import FeatConfig, MusionPCM
from musion.utils.ort_musion_base import OrtMusionBase
from musion.transcribe.base import *
from musion.transcribe.drums.velocity import estimate_velocity
from musion.separate.drums import _DrumsSeparate
from musion.transcribe.midi_utils import MIDI_RESOLUTION

MODULE_PATH = os.path.dirname(__file__)

# Drum MIDI note numbers: ["BD", "SD", "TT", "HH", "CY+RD"]
LABELS_5: List[int] = [35, 38, 47, 42, 49]
FPS: int = 100
DEFAULT_NOTE_DURATION: float = 0.01  # seconds
DEFAULT_VELOCITY: int = 100  # Will be refined by estimate_velocity

class _DrumsTranscribe(TranscribeBase, OrtMusionBase):
    def __init__(self, device: str = None) -> None:
        TranscribeBase.__init__(self, device)
        OrtMusionBase.__init__(self,
            os.path.join(MODULE_PATH, 'transcribe_drums.onnx'),
            device)

        frameSize = self._feat_cfg.n_fft
        audio_sample_rate = self._feat_cfg.sample_rate

        sig = SignalProcessor(num_channels=1, sample_rate=audio_sample_rate)
        frames = FramedSignalProcessor(frame_size=frameSize, fps=FPS)
        stft = ShortTimeFourierTransformProcessor()
        spec = LogarithmicFilteredSpectrogramProcessor(
            num_channels=1,
            sample_rate=audio_sample_rate,
            filterbank=LogarithmicFilterbank,
            frame_size=frameSize,
            fps=FPS,
            num_bands=12,
            fmin=20,
            fmax=20000,
            norm_filters=True,
        )

        self.pre_processor = SequentialProcessor((sig, frames, stft, spec))

        # Peak picking thresholds for each drum type
        peak_thresholds: List[float] = [0.22, 0.24, 0.32, 0.22, 0.2]
        self.processors = [
            madmom.features.notes.NoteOnsetPeakPickingProcessor(
                threshold=t, smooth=0, pre_avg=0.1, post_avg=0.01, pre_max=0.02, post_max=0.01, combine=0.02, fps=FPS)
            for t in peak_thresholds
        ]

        self.separate_drums = _DrumsSeparate(self.device)

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            mono=True,
            sample_rate=44100,
            n_fft=2048,
            hop_length=441,
            f_min=20,
            f_max=20000,
        )

    def _process(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> Dict[str, mido.MidiFile]:
        audio = self.pre_processor(audio_path)
        # Reshape to add channel dimension: [num_frames, num_bands] -> [num_frames, num_bands, 1]
        audio = audio.reshape((audio.shape[0], audio.shape[1], 1))

        raw_probs = self.predict(audio)[0]  # [num_frames, num_labels]

        midi = pretty_midi.PrettyMIDI(resolution=MIDI_RESOLUTION)
        instrument = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        midi.instruments.append(instrument)

        for i, processor in enumerate(self.processors):
            # Reshape probability array for peak picking: [num_frames] -> [num_frames, 1]
            prob_array = raw_probs[:, i].reshape(-1, 1)
            peaks = processor.process(prob_array)  # Shape: (num_peaks, 2) with (time, pitch)

            for onset_time in peaks[:, 0]:
                note = pretty_midi.Note(
                    velocity=DEFAULT_VELOCITY,
                    pitch=LABELS_5[i],
                    start=onset_time,
                    end=onset_time + DEFAULT_NOTE_DURATION
                )
                instrument.notes.append(note)

        drum_parts_separation_res = self.separate_drums(audio_path=audio_path)
        estimate_velocity(midi, drum_parts_separation_res)
        midi = self._align_midi_with_beats(midi, audio_path)

        return {'mid': midi}

    def predict(self, audio: np.ndarray, limit_input_size: int = 60000) -> np.ndarray:
        """
        Predict drum transcription probabilities for audio input.
        
        For RNN models, uses overlapping windows with warmup sequences to handle long audio.
        
        Args:
            audio: Input audio features with shape [num_frames, num_bands, num_channels]
            limit_input_size: Maximum window size before potential segmentation fault.
                             If input is larger, it's split into overlapping windows.
        
        Returns:
            Probability array with shape [num_frames, num_labels]
        
        TODO: Identify the error "Computed output size would be negative: -2 
              [input_size: 0, effective_filter_size: 3, stride: 1]"
        """
        window_size = limit_input_size
        warmup_size = 412
        step_size = window_size - 2 * warmup_size

        predictions: List[np.ndarray] = []
        # Split into overlapping windows if input is too large
        window_indices = range(0, len(audio) - warmup_size, step_size)
        
        for sample_idx in window_indices:
            window_audio = audio[sample_idx : sample_idx + window_size]
            # Add batch dimension: [num_frames, num_bands, num_channels] -> [1, num_frames, num_bands, num_channels]
            window_audio_batch = window_audio.reshape((1,) + window_audio.shape)
            prediction = self._predict([window_audio_batch])[0]
            
            if len(window_indices) == 1:
                # Single window: no warmup needed
                predictions.append(prediction)
            elif sample_idx == 0:
                # First window: no warmup at beginning, only at end
                predictions.append(prediction[:window_size - warmup_size])
            elif sample_idx == window_indices[-1]:
                # Last window: warmup at beginning, none at end
                predictions.append(prediction[warmup_size:])
            else:
                # Middle windows: warmup at both beginning and end
                predictions.append(prediction[warmup_size:][:step_size])

        return np.concatenate(predictions)

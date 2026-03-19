import abc
from typing import Optional

import numpy as np
import mido
import pretty_midi

import musion
from musion.transcribe.midi_utils import align_midi_with_beats, create_mido_midifile_stream
from musion.utils.base import MusionPCM
from musion.utils.ort_musion_base import OrtMusionBase


class TranscribeBase(OrtMusionBase):
    def __init__(self, model_path: str, device: str = None) -> None:
        super().__init__(model_path, device)
        self.beat = musion.beat.inference._Beat(self.device)
        self.separate = musion.separate._Separate(self.device)

    def _align_midi_with_beats(self, midi, audio_path) -> mido.MidiFile:
        beats = self.beat(audio_path=audio_path)['beats']
        beats = np.asarray(beats)
        if isinstance(midi, mido.MidiFile):
            midi = create_mido_midifile_stream(midi)
            midi = pretty_midi.PrettyMIDI(midi)
        midi = align_midi_with_beats(midi, beats)
        return midi

    @abc.abstractmethod
    def _process_wo_beat_align(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> dict:
        raise NotImplementedError("Subclass must implement this method.")

    def _process(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None,
                 mix_audio_path: Optional[str] = None) -> dict:
        if audio_path is None and pcm is None and mix_audio_path is not None:
            if self.accept_input_stem is not None:
                stem_samples = self.separate(audio_path=mix_audio_path, pcm=pcm)[self.accept_input_stem + '.wav']
                pcm = MusionPCM(stem_samples, self.separate._feat_cfg.sample_rate)
            else:
                raise ValueError("The target instrument to transcribe is not supported by separation from a mix audio, you should provide a pure solo/separated recording for just ONE corresponding instrument by passing auio_path.")

        midi = self._process_wo_beat_align(audio_path, pcm)
        if mix_audio_path is not None:  # use mix audio path for better beat tracking result
            audio_path = mix_audio_path
        midi = self._align_midi_with_beats(midi, audio_path)
        return {'mid': midi}

    def _save(self, key, save_path, res):
        if 'mid' in key:
            res[key].write(save_path) if isinstance(res[key], pretty_midi.PrettyMIDI) else res[key].save(save_path)

    @property
    @abc.abstractmethod
    def accept_input_stem(self) -> str:
        raise NotImplementedError("Subclass must implement this method.")

    @property
    def result_keys(self):
        return ['mid']
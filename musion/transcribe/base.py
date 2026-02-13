import numpy as np
import mido
import pretty_midi

import musion
from musion.transcribe.midi_utils import align_midi_with_beats, create_mido_midifile_stream



class TranscribeBase:
    def __init__(self, device: str = None) -> None:
        self.beat = musion.beat.inference._Beat(device)

    def _align_midi_with_beats(self, midi, audio_path) -> mido.MidiFile:
        beats = self.beat(audio_path=audio_path)['beats']
        beats = np.asarray(beats)
        if isinstance(midi, mido.MidiFile):
            midi = create_mido_midifile_stream(midi)
            midi = pretty_midi.PrettyMIDI(midi)
        midi = align_midi_with_beats(midi, beats)
        return midi

    def _save(self, key, save_path, res):
        if 'mid' in key:
            res[key].write(save_path) if isinstance(res[key], pretty_midi.PrettyMIDI) else res[key].save(save_path)

    @property
    def result_keys(self):
        return ['mid']
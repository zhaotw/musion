"""
mido: tick based time stamp, pretty_midi: second based time stamp
"""

from typing import Optional, List
import io
from dataclasses import dataclass

import numpy as np
import mido
import pretty_midi

MIDI_RESOLUTION = 960 # ticks per beat

@dataclass
class MidiEvent:
    """
    intermediate data type for midi related operations across various midi libraries
    """
    time: float # in seconds
    type: str   # ['note_on', 'note_off', 'tempo', 'tempo_fdb', 'control_change'], mido defined
    value: int
    velocity: Optional[int] = None

def preprocess_beats(beats):
    if beats[0][0] != 0:
        beats = np.insert(beats, 0, [0, 0], 0)

    # fill blank in the front
    BEAT_DUR_THRESHLOD = 12 # sec
    fill_cnt = 0
    for i in range(1, len(beats)):
        if beats[i][0] - beats[i-1][0] > BEAT_DUR_THRESHLOD:
            beats = np.insert(beats, i, [beats[i-1][0] + BEAT_DUR_THRESHLOD, 0], 0)
            fill_cnt += 1

    return beats, fill_cnt

def get_tempo_changes(beats, first_db_time):
    tempo_changes = []

    for i in range(1, len(beats)):
        beat_dur = beats[i][0] - beats[i - 1][0]
        bpm = 60 / beat_dur
        tempo = mido.bpm2tempo(bpm)
        is_first_db = beats[i - 1][0] == first_db_time
        tempo_changes.append(MidiEvent(beats[i - 1][0], "tempo_fdb" if is_first_db else "tempo", tempo))
    return tempo_changes

def split_on_offset(note_seq) -> List[MidiEvent]:
    note_on_off_events = []
    for note in note_seq:
        onset_note = MidiEvent(note.start, "note_on", note.pitch, note.velocity)
        offset_note = MidiEvent(note.end, "note_off", note.pitch, 0)
        note_on_off_events.append(onset_note)
        note_on_off_events.append(offset_note)
    return note_on_off_events

def cc2event(control_changes):
    cc_events = []
    for cc in control_changes:
        cc_event = MidiEvent(cc.time, "control_change", cc.number, cc.value)
        cc_events.append(cc_event)
    return cc_events

def get_meta_track(beats, numerator, ticks_per_beat):
    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage('track_name', name="Tempo & Time Signature", time=0))
    meta_track.append(mido.MetaMessage('time_signature', numerator=1, denominator=4, time=0))

    past_ticks = 0
    first_db_set = False
    for i in range(1, len(beats)):
        beat_dur = beats[i][0] - beats[i - 1][0]
        bpm = 60 / beat_dur
        tempo = mido.bpm2tempo(bpm)

        if not first_db_set and int(beats[i-1][1]) == 1:
            # time_sig msg must be ahead of tempo msg for correct tempo calculation
            meta_track.append(mido.MetaMessage('time_signature', numerator=numerator, denominator=4, time=past_ticks))
            first_db_set = True
            meta_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        else:
            meta_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=past_ticks))
        
        past_ticks = mido.second2tick(beat_dur, ticks_per_beat, tempo)

    return meta_track

def get_beat_aligned_track(midi_events, init_tempo, ticks_per_beat, inst_name, inst_program, channel):
    track = mido.MidiTrack()

    track.append(mido.MetaMessage('track_name', name=inst_name, time=0))
    track.append(mido.Message('program_change', program=inst_program, time=0, channel=channel))

    last_tempo = init_tempo

    end_sec = 0
    dt = 0
    for event in midi_events[1:]:
        time = mido.second2tick(event.time - end_sec, ticks_per_beat, last_tempo)
        if 'tempo' in event.type:
            if event.value == last_tempo and event.type != "tempo_fdb":
                continue
            last_tempo = event.value
            dt += time
        elif event.type == 'control_change':
            dt += time
            track.append(mido.Message('control_change', control=event.value, value=event.velocity, time=dt, channel=channel))
            dt = 0
        else:
            dt += time
            track.append(mido.Message(event.type, note=event.value, velocity=event.velocity, time=dt, channel=channel))
            dt = 0
        end_sec = event.time

    return track

def get_beats_meta(beats):
    numerator = max(beats[:, 1])
    for t, n in beats:
        if int(n) == 1:
            first_db_time = t
            break
        
    return int(numerator), first_db_time

def create_mido_midifile_stream(midi: mido.MidiFile):
    midi_bytes = io.BytesIO()
    midi.save(file=midi_bytes)
    midi_bytes.seek(0)
    return midi_bytes

def align_midi_with_beats(midi: pretty_midi.PrettyMIDI, beats: np.ndarray) -> mido.MidiFile:
    ticks_per_beat = midi.resolution
    beats, fill_cnt = preprocess_beats(beats)
    numerator, first_db_time = get_beats_meta(beats)

    midi_with_tempo = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    midi_with_tempo.tracks.append(get_meta_track(beats, numerator, ticks_per_beat))

    tempo_changes = get_tempo_changes(beats, first_db_time)
    
    channel = 0
    for inst in midi.instruments:
        track_name, program = inst.name, inst.program
        midi_events = split_on_offset(inst.notes)
        midi_events.extend(tempo_changes)
        midi_events.extend(cc2event(inst.control_changes))
        midi_events.sort(key=lambda x: x.time)

        track_with_tempo = get_beat_aligned_track(midi_events, tempo_changes[0].value, ticks_per_beat, track_name, program, 9 if inst.is_drum else channel)
        midi_with_tempo.tracks.append(track_with_tempo)

        if channel == 8: # skip drums channel
            channel = 9
        channel = (channel + 1) % 16
    
    return midi_with_tempo

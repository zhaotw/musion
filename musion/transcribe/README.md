1. Piano Transcription
Encapsulation for [Piano Transcription](https://github.com/bytedance/piano_transcription)

Transcribe a piano piece to score in midi format.

Input:
    audio file path or audio pcm

Return:  
{  
    'mid':  mido.MidiFile  
}

2. Vocal Transcription
Encapsulation for [SVT_SpeechBrain](https://github.com/guxm2021/SVT_SpeechBrain)

Input:
    audio file path or audio pcm

Return:  
{  
    'vocals':  [ [note start time, note end time, note pitch] x n ] 
}
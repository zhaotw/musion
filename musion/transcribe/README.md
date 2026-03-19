# Supported Instruments
1. Piano Transcription
Encapsulation for [Transkun](https://github.com/Yujia-Yan/Transkun)


2. Vocal Transcription
Encapsulation for [SVT_SpeechBrain](https://github.com/guxm2021/SVT_SpeechBrain)


3. Drums Transcription
Encapsulation for [DrumTranscription](https://github.com/xavriley/DrumTranscription)

# Input:  
    audio file path or audio pcm

There are 3 types of file path input:
### 1. audio_path
    audio_path is supposed to be a pure solo/separated recording for just ONE corresponding instrument.
### 2. mix_audio_path
    mix_audio_path is supposed to be a mixed recording of all instruments. and will be separated to the target instrument internally.
### 3. audio_path and mix_audio_path
    audio_path and mix_audio_path are both provided, and mix_audio_path is just used for better beat tracking result.

```python
from musion.transcribe import Transcribe

trans = Transcribe('Vocal')

# 1
midi1 = trans(audio_path="/mnt/nas1/users/tianwei.zhao/repo/musion/test_data/separate/プラチナ/プラチナ.vocals.wav")

# 2
midi2 = trans(mix_audio_path="/mnt/nas_tianwei/music/プラチナ.mp3")

# 3
midi3 = trans(audio_path="/mnt/nas1/users/tianwei.zhao/repo/musion/test_data/separate/プラチナ/プラチナ.vocals.wav",
              mix_audio_path="/mnt/nas_tianwei/music/プラチナ.mp3")
```

# Return:  
{  
    'mid':  mido.MidiFile  
}
Encapsulation for [Demucs](https://github.com/facebookresearch/demucs) HTDemucs

Separate songs into "drums", "bass", "other", "vocals"

Input:
    audio file path or audio pcm

Return:  
{  
    'drums':  drum_wavform_data,  
    'bass':   bass_wavform_data,  
    'other':  other_wavform_data,  
    'vocals': vocals_wavform_data,  
}
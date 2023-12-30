# MUSION, a toolbox for music information retrieval and music signal processing.
A collection of MIR open-source inference tools, with a unified and concise { Python interface | CLI | GUI }.

## Current tools
[separate](musion/separate/README.md) music source separation  
Separate songs into "drums", "bass", "other", "vocals"

[struct](musion/struct/README.md) music structure analysis / music segmentation
Detect chorus part for pop songs

[transcribe](musion/transcribe/README.md) automatic music transcription  
Only support for piano transcription now

## Installation
1. Download ONNX model files from https://zenodo.org/record/8183131, and put them in the corresponding folders.
2. Install locally.
```shell
pip install .
```
3. Support cuda acceleration. Install it if you would like optimal runtime performance.

## Example Usage
All the tools use the same procedure and method, only different at tool name.  
Here is a example for the `struct` tool.
```python
from musion.struct import Struct

struct = Struct()

# Select one of the two ways for input
## 1. Process a single file, given its file path
struct_res = struct(audio_path='dir/audio.wav')
## audio_path could also be a directory that contains audio files, or a list of audio paths.
struct_res = struct(audio_path='dir/')
struct_res = struct(audio_path=['audio1.wav', 'audio2.wav'])

## 2. Process a single audio data, given its pcm
from musion import MusionPCM
pcm = MusionPCM(samples, sample_rate)
struct_res = struct(pcm=pcm)

""" Optional Parameters """
# Save the result to a file
from musion import SaveConfig
save_cfg = SaveConfig(dir_path='dir/to/save/in', keys=['struct'])
# Where the keys can be obtained by
struct.result_keys
# Save the result by passing save_cfg, to the file: dir/to/save/in/audio.strcut
struct(audio_path='dir/audio.wav', save_cfg=save_cfg)

# Enable parallel processing when input contains multiple files, just set a proper number for num_threads
struct(audio_path='dir/', num_threads=5)

```
## Command-Line Interface
Almost the same parameters as above. Type '$ musion -h' for more details. Here's a comprehensive example.
```shell
$ musion separate test_wavs/ --save_dir results_dir/ --save_keys vocals.wav bass.wav --num_threads 5
```

## Contributing
If you would like to participate in the development of Musion you are more than welcome to do so. Don't hesitate to throw us a pull request and we'll do our best to examine it quickly.

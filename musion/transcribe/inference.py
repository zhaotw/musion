from musion.utils.base import TaskDispatcher
from musion.transcribe.piano.piano import _PianoTranscribe
from musion.transcribe.drums.drums import _DrumsTranscribe
from musion.transcribe.vocal.vocal import _VocalTranscribe


class Transcribe(TaskDispatcher): 
    def __init__(self, target_instrument: str):
        task_class = globals().get(f'_{target_instrument}Transcribe', None)
        if task_class:
            super().__init__(task_class)
        else:
            raise ValueError(f"Unsupported instrument: {target_instrument}")

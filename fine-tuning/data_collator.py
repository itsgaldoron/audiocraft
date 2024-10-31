import torch
import torchaudio
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MusicGenCollator:
    processor: any
    max_length: int = 30  # seconds
    sampling_rate: int = 48000
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        audio_paths = [feature["audio_path"] for feature in features]
        text_inputs = [feature["text"] for feature in features]
        
        # Process audio
        waveforms = []
        for audio_path in audio_paths:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sampling_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sampling_rate)(waveform)
            # Ensure stereo
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            waveforms.append(waveform)
        
        # Process text
        text_inputs = self.processor(
            text_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "audio": torch.stack(waveforms)
        } 
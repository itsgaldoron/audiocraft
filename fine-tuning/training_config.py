from dataclasses import dataclass
from transformers import TrainingArguments

@dataclass
class MusicGenConfig:
    model_name = "facebook/musicgen-stereo-small"
    max_audio_length = 30  # seconds
    sampling_rate = 48000
    
    training_args = TrainingArguments(
        output_dir="./musicgen-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=500,
        save_steps=1000,
        logging_steps=100,
        save_total_limit=2,
        fp16=True,
        remove_unused_columns=False,
    ) 
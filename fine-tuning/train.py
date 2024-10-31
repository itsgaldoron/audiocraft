import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from prepare_dataset import create_dataset
from training_config import MusicGenConfig
from data_collator import MusicGenCollator
from transformers import Trainer

def main():
    # Load model and processor
    config = MusicGenConfig()
    model = MusicgenForConditionalGeneration.from_pretrained(config.model_name)
    processor = AutoProcessor.from_pretrained(config.model_name)
    
    # Prepare dataset
    dataset = create_dataset()
    
    # Create data collator
    data_collator = MusicGenCollator(
        processor=processor,
        max_length=config.max_audio_length,
        sampling_rate=config.sampling_rate
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=config.training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model("./musicgen-finetuned-final")

if __name__ == "__main__":
    main() 
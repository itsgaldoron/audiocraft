import json
import os
from datasets import Dataset

def load_json_files(dataset_dir):
    data = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json'):
            with open(os.path.join(dataset_dir, filename), 'r') as f:
                item = json.load(f)
                # Prepare text description
                description = f"{item['description']} Genre: {item['genre']}."
                if item['moods']:
                    description += f" Moods: {', '.join(item['moods'])}."
                if item['keywords']:
                    description += f" Keywords: {item['keywords']}."
                
                data.append({
                    'audio_path': f"dataset/example/{item['name']}.{item['file_extension']}",
                    'text': description
                })
    return data

def create_dataset():
    dataset_items = load_json_files('dataset/example')
    return Dataset.from_list(dataset_items) 
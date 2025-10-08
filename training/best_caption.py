import torch
import json
import os
import random
import pathlib
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
import re
import re

def clean_repetitions(text):
    end_token = "<|endoftext|>"
    if end_token in text:
        text = text.split(end_token)[0]

    match = re.search(r'(.*[.!?])', text)
    if match:
        text = match.group(1)

    text = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\'\"()\-\[\]]+', ' ', text)
    
    garbage_words = [
        'pestic', 'pesticidal', 'pesticid', 'pesticides', 'pesticider', 'pesticidr',
        'strutConnector', 'attRot', 'guiActiveUnfocused', 'guiActive', 'guiName', 'guiIcon',
        'sbm', 'wcs', 'tnc', 'dfx', 'srfAttach', 'TheNitromeFan', 'TheNitrome',
        'Canaver', 'Canavero', 'Kinnikuman', 'Magikarp', 'SolidGoldMagikarp',
        'ModLoader', 'ForgeModLoader', 'Reloaded', 'Initialized', 'Unloaded',
        'Interstitial', 'embedreportprint', 'Cosponsors', 'EStream', 'CrossRef',
        'milomilo', 'madeupword0002', 'externalToEVAOnly', 'externalToEVA',
        'Downloadha', 'Azerbijani', 'Azerbaidjan', 'Azerbajani', 'contraceptive',
        'antioxid', 'Pengu', 'Leilan', 'TAMADRA', 'CLASSIFIED', 'Vaults', 'Depths',
        'NVIDIA', 'PsyNetMessage', 'entrepreneu', 'EntityItem', 'regor', 'userc',
        'UCHIJ', 'fmt', 'enthusi', 'misunderstaning', 'contraceptive', '4090',
        'milo', 'unk', 'guiNamemilo', 'TheNitromemiloembedreportprintmilo',
        'strutConnectorModLoader', 'strutConnectorReloaded', 'attRotReloaded',
        'concludere', 'opioid', 'opioider', 'opioidergi', 'opioidergicos',
        'opioidergen', 'opioidergenes', 'opioiderm', 'opioidermon', 'opioidermones',
        'opioiderme', 'opioidermes', 'opioidermo', 'opioidermos', 'opioidermona',
        'opioidermonas', 'opioiderma', 'opioidermas', 'opioiderno', 'opioidernos',
        'opioidernio', 'opioidernios', 'opioiderna', 'opioidernas', 'opioiderpa',
        'opioiderpas', 'opioiderpe', 'opioiderpes', 'opioiderpetro', 'opioiderpetros',
        'opioiderpi', 'opioiderpis', 'opioiderre', 'opioiderres', 'opioiders',
        'opioiderte', 'opioidertes', 'opioiderde', 'opioiderdes', 'opioiderdad',
        'opioiderdades', 'opioidertada', 'opioidertadas', 'opioider', 'Archdemon',
        'antioxidaci', 'antioxidacin', 'contraceptivos', 'pesticidas'
    ]
    for word in garbage_words:
        pattern = re.escape(word)
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    words = text.split()
    cleaned_words = []
    
    for word in words:
        if not cleaned_words or word.lower() != cleaned_words[-1].lower():
            cleaned_words.append(word)
    
    result = ' '.join(cleaned_words)
    
    patterns_to_clean = [
        r'\b(\w+)\s+\1\b',
        r'\b(\w+\s+\w+)\s+\1\b',
    ]
    
    for pattern in patterns_to_clean:
        result = re.sub(pattern, r'\1', result, flags=re.IGNORECASE)

    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()

def generate_caption_safe(model, processor, image_path, device):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    param_sets = [
        {"repetition_penalty": 2.0, "temperature": 0.8, "top_k": 50},
        {"repetition_penalty": 3.0, "temperature": 0.7, "top_k": 30},
        {"repetition_penalty": 4.0, "temperature": 0.6, "top_k": 20},
    ]
    
    best_caption = ""
    min_repetitions = float('inf')
    
    for params in param_sets:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=200,
                min_length=15,
                num_beams=5,
                no_repeat_ngram_size=3,
                do_sample=True,
                early_stopping=True,
                **params
            )
        
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        words = caption.split()
        word_counts = {}
        max_count = 0
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            max_count = max(max_count, word_counts[word])
        
        if max_count < min_repetitions:
            min_repetitions = max_count
            best_caption = caption
            
        if max_count <= 2:
            break
    
    return clean_repetitions(best_caption)

def test_model(checkpoint_path, test_json=None, image_base_path=None, num_samples=50):
    random.seed(42)
    
    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent
    
    if test_json is None:
        test_json = project_root / "data" / "processed" / "test_224.json"
    if image_base_path is None:
        image_base_path = project_root / "data" / "images"
    
    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / "results" / checkpoint_path
    
    print(f"Loading model from: {checkpoint_path}")
    
    processor = Blip2Processor.from_pretrained(checkpoint_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    model.eval()
    device = next(model.parameters()).device
    
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    results = []
    
    print(f"\nGenerating original captions for {num_samples} images...")
    
    for i in tqdm(range(min(num_samples, len(test_data)))):
        item = test_data[i]
        img_path = os.path.join(image_base_path, item['image'])
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        try:
            generated_caption = generate_caption_safe(model, processor, img_path, device)
            
            result = {
                'index': i,
                'image': item['image'],
                'original_caption': item['caption'],
                'generated_caption': generated_caption,
                'context': item.get('context', ''),
            }
            results.append(result)
            
            if i < 5:
                print(f"\n--- Example {i+1} ---")
                print(f"Image: {item['image']}")
                print(f"Original: {item['caption'][:100]}...")
                print(f"Generated: {generated_caption}")
                if item.get('context'):
                    print(f"Context: {item.get('context', '')}")
                
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    
    output_file = project_root / "results" / f"test_results_{checkpoint_path.name}_original.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    avg_length_original = sum(len(r['original_caption'].split()) for r in results) / len(results)
    avg_length_generated = sum(len(r['generated_caption'].split()) for r in results) / len(results)
    
    name_count = sum(1 for r in results if any(word[0].isupper() for word in r['generated_caption'].split() if word))
    date_count = sum(1 for r in results if any(char.isdigit() for char in r['generated_caption']))
    
    print(f"\nStatistics:")
    print(f"Total samples processed: {len(results)}")
    print(f"Average original caption length: {avg_length_original:.1f} words")
    print(f"Average generated caption length: {avg_length_generated:.1f} words")
    print(f"Captions with proper nouns: {name_count}/{len(results)} ({name_count/len(results)*100:.1f}%)")
    print(f"Captions with numbers/dates: {date_count}/{len(results)} ({date_count/len(results)*100:.1f}%)")
    
    return output_file

if __name__ == "__main__":
    model_checkpoint = "/home/labs25captioning/blip2/results/multi_gpu_training/multi_gpu_20250830_004609/best_model"
    
    results_file = test_model(
        checkpoint_path=model_checkpoint,
        num_samples=50
    )
    
    print(f"\nInference completed! Results saved to: {results_file}")
    print("You can now run evaluation using:")
    print(f"python evaluate_results.py --results_file {results_file}")
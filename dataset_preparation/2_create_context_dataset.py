import json
import random
import pathlib
import re
import html
import unicodedata
import spacy

script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent

IMG_DIR = project_root / "data" / "images" / "original"
META_FILE = project_root / "data" / "raw" / "captioning_dataset.json"
URLS_FILE = project_root / "data" / "raw" / "img_urls_super.json"
OUTPUT_DIR = project_root / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 20_000
EVENT_KEYWORDS = ["Olympics", "World Cup", "Super Bowl", "FIFA", "World Series"]

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
url_date_pattern = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")

def clean_text(text):
    return " ".join(html.unescape(unicodedata.normalize("NFKC", text)).split())

def get_entities_by_label(doc, *labels):
    return [e.text for e in doc.ents if e.label_ in labels]

def get_best_entity(entity_list):
    return max(entity_list, key=len) if entity_list else ""

def extract_date_from_url(url, doc):
    match = url_date_pattern.search(url)
    if match:
        return "-".join(match.groups())
    for entity in doc.ents:
        if entity.label_ == "DATE" and re.fullmatch(r"\d{4}", entity.text):
            return entity.text
    return ""

def extract_event(doc):
    events = get_entities_by_label(doc, "EVENT")
    if events:
        return get_best_entity(events)
    text_lower = doc.text.lower()
    for keyword in EVENT_KEYWORDS:
        if keyword.lower() in text_lower:
            return keyword
    return ""

def extract_section_from_url(url_parts):
    for segment in url_parts[7:]:
        if segment.isalpha():
            return segment
    return ""

def create_context_dataset():
    print("Loading metadata and URLs...")
    with open(META_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open(URLS_FILE, 'r', encoding='utf-8') as f:
        urls = json.load(f)
    
    dataset = []
    print(f"Processing {len(metadata)} articles...")
    
    for article_id, article_data in metadata.items():
        image_slots = article_data.get("images", {})
        url_slots = urls.get(article_id, {})
        
        for slot, raw_caption in image_slots.items():
            image_name = f"{article_id}_{slot}.jpg"
            image_path = IMG_DIR / image_name
            
            if not image_path.exists():
                continue
            
            caption = clean_text(raw_caption)
            if len(caption.split()) < 5:
                continue
            
            url = url_slots.get(str(slot), "")
            url_parts = url.split("/")
            
            doc = nlp(caption)
            
            section = extract_section_from_url(url_parts)
            place = get_best_entity(get_entities_by_label(doc, "GPE", "LOC"))
            date = extract_date_from_url(url, doc)
            person = get_best_entity(get_entities_by_label(doc, "PERSON", "ORG"))
            event = extract_event(doc)
            
            context_parts = []
            if section:
                context_parts.append(section)
            if place:
                context_parts.append(place)
            if date:
                context_parts.append(date)
            if person:
                context_parts.append(f"PERSON={person}")
            if event:
                context_parts.append(f"EVENT={event}")
            
            context_string = f"[{' | '.join(context_parts)}] " if context_parts else ""
            
            dataset.append({
                "image": str(image_path),
                "context": context_string,
                "caption": caption
            })
    
    print(f"Created dataset with {len(dataset)} samples")
    
    random.seed(42)
    random.shuffle(dataset)
    dataset = dataset[:TARGET_SIZE]
    
    print(f"Limited to {len(dataset)} samples")
    
    n = len(dataset)
    splits = {
        "train": dataset[:int(0.8 * n)],
        "val": dataset[int(0.8 * n):int(0.9 * n)],
        "test": dataset[int(0.9 * n):]
    }
    
    for split_name, split_data in splits.items():
        output_path = OUTPUT_DIR / f"{split_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"{split_name:5} {len(split_data):,} samples -> {output_path}")
    
    print("Context dataset created successfully")

if __name__ == "__main__":
    create_context_dataset()
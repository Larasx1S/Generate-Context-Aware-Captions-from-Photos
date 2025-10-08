import json
import pathlib
import shutil
import hashlib
from tqdm import tqdm

script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent

SOURCE_DIR = project_root / "data" / "processed"
IMG_DIR = project_root / "data" / "images" / "images224"
JSON_FILES = ["train_224.json", "val_224.json", "test_224.json"]

def get_subdirectory_name(filename: str) -> str:
    hash_hex = hashlib.sha1(filename.encode()).hexdigest()[:2]
    return hash_hex

def organize_images():
    print("Organizing images into subdirectories...")
    image_files = list(IMG_DIR.glob("*.jpg"))
    
    for image_path in tqdm(image_files, desc="Moving images"):
        subdir_name = get_subdirectory_name(image_path.name)
        subdir_path = IMG_DIR / subdir_name
        subdir_path.mkdir(exist_ok=True)
        
        dest_path = subdir_path / image_path.name
        if not dest_path.exists():
            shutil.move(str(image_path), str(dest_path))

def update_json_files():
    print("Updating JSON file paths...")
    
    for json_filename in JSON_FILES:
        json_path = SOURCE_DIR / json_filename
        
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping")
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for sample in data:
            filename = pathlib.Path(sample["image"]).name
            subdir_name = get_subdirectory_name(filename)
            sample["image"] = f"images224/{subdir_name}/{filename}"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Updated {json_filename}")

def main():
    organize_images()
    update_json_files()
    print("Folder organization completed")

if __name__ == "__main__":
    main()
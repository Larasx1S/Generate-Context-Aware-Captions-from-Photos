import json
import pathlib
from multiprocessing.pool import ThreadPool
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent

SOURCE_DIR = project_root / "data" / "processed"
IMG_SOURCE_ROOT = project_root / "data" / "images" / "original"
IMG_224_DIR = project_root / "data" / "images" / "images224"
IMG_224_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 224
QUALITY = 90

def resize_single_image(args):
    image_path, dest_path = args
    if dest_path.exists():
        return
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
            
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BICUBIC)
            
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            
            img.save(dest_path, format="JPEG", quality=QUALITY, optimize=True)
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")

def main():
    print("Processing JSON files and resizing images...")
    
    for split_name in ["train", "val", "test"]:
        input_path = SOURCE_DIR / f"{split_name}.json"
        output_path = SOURCE_DIR / f"{split_name}_224.json"
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        resize_tasks = []
        for sample in data:
            original_path = pathlib.Path(sample["image"])
            image_name = original_path.name
            
            source_img_path = IMG_SOURCE_ROOT / image_name
            dest_img_path = IMG_224_DIR / image_name
            
            if source_img_path.exists():
                resize_tasks.append((source_img_path, dest_img_path))
                sample["image"] = f"images224/{image_name}"
        
        with ThreadPool(8) as pool:
            pool.map(resize_single_image, resize_tasks)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"{split_name}: {len(data)} samples -> {output_path}")
    
    print("All images resized to 224px and RGBA converted to RGB")

if __name__ == "__main__":
    main()
import json
import re
import pathlib
import sys

def to_super_jumbo(url: str) -> str:
    return re.sub(r'-(articleLarge|jumbo|popup|articleInline)([^/]*?)\.jpg',
                  r'-superJumbo.jpg', url, flags=re.IGNORECASE)

def transform_urls_recursive(node):
    if isinstance(node, str):
        return to_super_jumbo(node)
    if isinstance(node, dict):
        return {k: transform_urls_recursive(v) for k, v in node.items()}
    if isinstance(node, list):
        return [transform_urls_recursive(x) for x in node]
    return node

def main():
    script_dir = pathlib.Path(__file__).parent
    input_path = script_dir.parent / 'data' / 'raw' / 'img_urls_all.json'
    output_path = script_dir.parent / 'data' / 'raw' / 'img_urls_super.json'
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Reading {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Converting URLs to superJumbo variants...")
    transformed_data = transform_urls_recursive(data)
    
    print(f"Writing {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()
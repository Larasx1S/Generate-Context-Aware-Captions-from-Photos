import json
import os
import sys
import requests
import argparse
import urllib.parse as urlparse
import posixpath
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path

def resolve_url_components(url):
    parsed = urlparse.urlparse(url)
    new_path = posixpath.normpath(parsed.path)
    if parsed.path.endswith('/'):
        new_path += '/'
    cleaned = parsed._replace(path=new_path)
    return cleaned.geturl()

def download_images_for_thread(args):
    thread_num, keys, urls_data, output_dir = args
    failed_downloads = []
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'
    })

    for key in tqdm(keys, desc=f"Thread {thread_num}", position=thread_num):
        if key not in urls_data:
            continue
            
        for slot_id, url in urls_data[key].items():
            output_path = os.path.join(output_dir, f"{key}_{slot_id}.jpg")
            
            if os.path.exists(output_path):
                continue
                
            try:
                clean_url = resolve_url_components(url)
                response = session.get(clean_url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
            except Exception as e:
                failed_downloads.append({
                    'key': key,
                    'slot_id': slot_id,
                    'url': url,
                    'error': str(e)
                })
    
    if failed_downloads:
        script_dir = Path(__file__).parent
        failed_file = script_dir / f"failed_downloads_thread_{thread_num}.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_downloads, f, indent=2)
        print(f"Thread {thread_num}: {len(failed_downloads)} failed downloads saved to {failed_file}")
    
    return len(failed_downloads)

def main():
    parser = argparse.ArgumentParser(description='Download images from GoodNews dataset URLs')
    parser.add_argument('--num_threads', type=int, default=8, 
                       help='Number of threads to use for downloading')
    parser.add_argument('--urls_file', type=str, default=None,
                       help='Path to the URLs JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save downloaded images')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if args.urls_file is None:
        args.urls_file = project_root / "data" / "raw" / "img_urls_super.json"
    if args.output_dir is None:
        args.output_dir = project_root / "data" / "images" / "original"
    
    urls_file = Path(args.urls_file)
    output_dir = Path(args.output_dir)
    
    if not urls_file.exists():
        print(f"Error: URLs file not found: {urls_file}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading URLs from {urls_file}")
    with open(urls_file, 'r') as f:
        all_img_urls = json.load(f)
    
    keys = list(all_img_urls.keys())
    total_images = sum(len(urls) for urls in all_img_urls.values())
    
    print(f"Found {len(keys)} articles with {total_images} total images")
    print(f"Starting download with {args.num_threads} threads...")
    
    thread_range = len(keys) // args.num_threads + 1
    thread_args = [
        (i + 1, keys[thread_range * i: thread_range * (i + 1)], all_img_urls, str(output_dir))
        for i in range(args.num_threads)
    ]
    
    failed_counts = Parallel(n_jobs=args.num_threads, backend="loky")(
        map(delayed(download_images_for_thread), thread_args)
    )
    
    total_failed = sum(failed_counts)
    print(f"\nDownload completed!")
    print(f"Total failed downloads: {total_failed}")
    print(f"Images saved to: {output_dir}")

if __name__ == '__main__':
    main()
import os
import json
from pathlib import Path
from tqdm import tqdm

# Update this path to your cache folder
cache_dir = Path("/home/middeljans/GastroKey/cache")

def cleanup_cache(directory):
    files = list(directory.glob("*.json"))
    removed_count = 0
    
    print(f"Scanning {len(files)} files...")
    
    for json_file in tqdm(files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check the specific method
            if data.get("method") == "latent_resnet":
                os.remove(json_file)
                removed_count += 1
                
        except (json.JSONDecodeError, KeyError, PermissionError) as e:
            print(f"Could not process {json_file.name}: {e}")

    print(f"--- Finished ---")
    print(f"Removed: {removed_count} files.")

if __name__ == "__main__":
    cleanup_cache(cache_dir)
import os
import json
from pathlib import Path

def update_json_files(directory_path):
    # Convert string path to a Path object
    base_dir = Path(directory_path)
    
    # Check if directory exists
    if not base_dir.exists():
        print(f"Error: The directory {directory_path} does not exist.")
        return

    # Counter for feedback
    updated_count = 0

    # Iterate over all .json files in the directory
    for json_file in base_dir.glob("*.json"):
        try:
            # 1. Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 2. Modify the specific fields
            # We use data.get() to avoid errors if a key is missing in some files
            modified = False
            
            if data.get("split") == "GastroKey":
                data["split"] = "Training set"
                modified = True
                
            if data.get("scope") == "prjs1485":
                data["scope"] = "Olympus"
                modified = True

            # 3. Save the file back if changes were made
            if modified:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                updated_count += 1
                #print(f"Updated: {json_file.name}")

        except Exception as e:
            print(f"Failed to process {json_file.name}: {e}")

    print(f"\nFinished! Total files updated: {updated_count}")

# Run the script
if __name__ == "__main__":
    # Replace this with the actual path to your JSON folder
    target_folder = "/home/middeljans/GastroKey/cache"
    update_json_files(target_folder)
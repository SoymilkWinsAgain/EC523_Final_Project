import json
import os
import random
import time
from pybooru import Danbooru

# --- PATHS ---
IMAGE_DIR = "" # Set this to your local directory containing the images"
MANIFEST_DIR = "" # Set this to your local directory for storing manifests
TRAIN_MANIFEST = os.path.join(MANIFEST_DIR, "train.jsonl")
VAL_MANIFEST = os.path.join(MANIFEST_DIR, "val.jsonl")

# --- AUTH ---
API_KEY = '' # Set this to your Danbooru API key
USERNAME = '' # Set this to your Danbooru username
client = Danbooru('danbooru', username=USERNAME, api_key=API_KEY)

def main():
    if not os.path.exists(MANIFEST_DIR):
        os.makedirs(MANIFEST_DIR, exist_ok=True)

    all_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(all_files)} files in {IMAGE_DIR}")

    manifest_entries = []
    
    # Simple ID-to-Character Cache to save API calls
    char_cache = {}

    for i, filename in enumerate(all_files):
        img_id = filename.split('.')[0]
        full_path = os.path.abspath(os.path.join(IMAGE_DIR, filename))
        
        # 1. Skip if somehow the path is wrong
        if not os.path.exists(full_path):
            continue

        # 2. Get Character Identity
        try:
            if img_id not in char_cache:
                post = client.post_show(img_id)
                tags = post.get('tag_string_character', '')
                char = tags.split()[0] if tags else "unknown"
                char_cache[img_id] = char
                print(f"[{i+1}/{len(all_files)}] {img_id} -> {char}")
                time.sleep(0.1)
            else:
                char = char_cache[img_id]

            if char != "unknown":
                manifest_entries.append({"path": full_path, "identity": char})
        except Exception as e:
            print(f"Error identifying {img_id}: {e}")

    # 3. Shuffle and Split (15% to Val)
    random.shuffle(manifest_entries)
    split_idx = int(len(manifest_entries) * 0.85)
    
    train_data = manifest_entries[:split_idx]
    val_data = manifest_entries[split_idx:]

    with open(TRAIN_MANIFEST, 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')
            
    with open(VAL_MANIFEST, 'w') as f:
        for entry in val_data:
            f.write(json.dumps(entry) + '\n')

    print(f"\nSUCCESS!")
    print(f"Training: {len(train_data)} images")
    print(f"Validation: {len(val_data)} images")
    print(f"Unique Characters: {len(set(e['identity'] for e in manifest_entries))}")

if __name__ == "__main__":
    main()
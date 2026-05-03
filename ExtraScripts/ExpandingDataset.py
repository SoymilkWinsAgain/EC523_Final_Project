import os
import time
import json
from curl_cffi import requests
from pybooru import Danbooru

# --- CONFIGURATION ---
METADATA_PATH = "" # Path to your existing metadata file (e.g., "metadata.jsonl")
SAVE_DIR = "" # Path to the directory where you want to save the images
NEW_LIMIT = 100 

# YOUR CREDENTIALS
API_KEY = 'Danbooru API Key'  # Replace with your actual API key
USERNAME = 'Danbooru Username'  # Replace with your actual username

# --- ADD YOUR NEW TAGS HERE ---
NEW_CHARACTER_TAGS = [
    # Example: "toki_(blue_archive)", "hoshino_ruby"
    # Ensure these are exactly as they appear on Danbooru, examples below:
    "kirito",
    "asuna_(sao)",
    "hyuuga_hinata"
]

client = Danbooru('danbooru', username=USERNAME, api_key=API_KEY)

# 1. Gather all characters
characters = set(NEW_CHARACTER_TAGS)
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, 'r') as f:
        for line in f:
            data = json.loads(line)
            char_tag = data.get('tag_string_character')
            if char_tag:
                characters.add(char_tag.split()[0])

print(f"Targeting {len(characters)} characters...")

# 2. Scrape
for tag in sorted(list(characters)):
    print(f"Syncing: {tag}")
    try:
        posts = client.post_list(tags=f"{tag} solo rating:g", limit=NEW_LIMIT)
        for post in posts:
            file_url = post.get('large_file_url') or post.get('file_url')
            if file_url:
                img_id = post['id']
                ext = file_url.split('.')[-1].split('?')[0]
                img_path = os.path.join(SAVE_DIR, f"{img_id}.{ext}")
                
                if not os.path.exists(img_path):
                    resp = requests.get(file_url, impersonate="chrome", headers={"Referer": "https://danbooru.donmai.us/"}, timeout=20)
                    if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                        with open(img_path, 'wb') as f:
                            f.write(resp.content)
        time.sleep(1.0)
    except Exception as e:
        print(f"  [!] Error with {tag}: {e}")

print("\nExpansion complete!")
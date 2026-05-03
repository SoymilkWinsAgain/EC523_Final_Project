import json
import os

metadata_path = "" # Path to your metadata file (e.g., "metadata.jsonl")
characters = set()

if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            char_tag = data.get('tag_string_character')
            if char_tag:
                # Danbooru often puts multiple characters in one string; we want them individually
                for char in char_tag.split():
                    characters.add(char)

print(f"Found {len(characters)} unique character tags to target.")
# Save them to a text file for the scraper
with open("target_characters.txt", "w") as f:
    f.write("\n".join(sorted(list(characters))))
import tarfile
import json
import os

# 1. Get the list of IDs you actually have in 0000
image_dir = "" # Adjust this to the path where your images are stored, e.g., "data/images/0000"
local_ids = {f.split('.')[0] for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))}
print(f"Searching for metadata for {len(local_ids)} specific images...")

input_tar = "" # Path to the tar.gz file containing the posts.json (e.g., "data/metadata/posts.tar.gz")
output_jsonl = "" # Path to the output .jsonl file

matches = 0
with tarfile.open(input_tar, "r:gz") as tar:
    # We look for the posts.json file inside the archive
    member = tar.getmember("posts.json")
    f = tar.extractfile(member)
    
    with open(output_jsonl, "w") as out:
        for line in f:
            data = json.loads(line)
            if str(data['id']) in local_ids:
                out.write(json.dumps(data) + "\n")
                matches += 1
                if matches % 50 == 0:
                    print(f"Found {matches} matches...")

print(f"Done! Created {output_jsonl} with {matches} real metadata entries.")
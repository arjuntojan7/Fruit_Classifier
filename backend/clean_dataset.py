import os
from PIL import Image

def remove_corrupted_images(folder_path):
    removed = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                img = Image.open(filepath)
                img.verify()  # Just verify, don't load fully
            except (IOError, SyntaxError):
                print(f"❌ Removing corrupted image: {filepath}")
                os.remove(filepath)
                removed += 1
    print(f"\n✅ Done. Removed {removed} corrupted images from {folder_path}")

# Run for both train and test directories
remove_corrupted_images("../dataset/train")
remove_corrupted_images("../dataset/test")

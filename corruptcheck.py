import os
from PIL import Image

test_dir = "Dataset/preprocessed_dataset/test"

for root, dirs, files in os.walk(test_dir):
    for file in files:
        try:
            img_path = os.path.join(root, file)
            img = Image.open(img_path)
            img.verify()  # Check for corruption
        except Exception as e:
            print(f"Corrupt image found: {img_path}, Error: {e}")

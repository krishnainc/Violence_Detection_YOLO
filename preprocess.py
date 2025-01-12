import cv2
import os

def preprocess_images(input_folder, output_folder, img_size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    for cls in os.listdir(input_folder):
        cls_path = os.path.join(input_folder, cls)
        output_cls_path = os.path.join(output_folder, cls)
        os.makedirs(output_cls_path, exist_ok=True)

        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Warning: Unable to read image {img_path}")
                continue

            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Resize and normalize
            resized = cv2.resize(image, img_size)
            
            # Save resized image as is (normalization can be done during loading)
            cv2.imwrite(os.path.join(output_cls_path, img_name), resized)

# Example usage
preprocess_images("Dataset/split_dataset/train", "Dataset/preprocessed_dataset/train")
preprocess_images("Dataset/split_dataset/val", "Dataset/preprocessed_dataset/validation")
preprocess_images("Dataset/split_dataset/test", "Dataset/preprocessed_dataset/test")

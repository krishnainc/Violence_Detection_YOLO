import os
import shutil
from sklearn.model_selection import train_test_split

def collect_frames_from_subfolders(folder_path):
    """Collect all frame paths from subfolders."""
    frame_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.png')):  # Add other extensions if needed
                frame_paths.append(os.path.join(root, file))
    return frame_paths

def split_data(input_folder, output_folder, train_ratio=0.7):
    os.makedirs(output_folder, exist_ok=True)

    for label in os.listdir(input_folder):
        label_folder = os.path.join(input_folder, label)
        if not os.path.isdir(label_folder):
            continue

        frames = collect_frames_from_subfolders(label_folder)
        print(f"Processing label: {label} with {len(frames)} frames.")

        if len(frames) == 0:
            print(f"Warning: No frames found for label '{label}'. Skipping.")
            continue

        train, temp = train_test_split(frames, test_size=(1 - train_ratio), random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        for split, split_frames in zip(['train', 'val', 'test'], [train, val, test]):
            split_output_folder = os.path.join(output_folder, split, label)
            os.makedirs(split_output_folder, exist_ok=True)
            for frame in split_frames:
                shutil.copy(frame, split_output_folder)

# Example usage
split_data("Dataset/sampled_frames", "Dataset/split_dataset")

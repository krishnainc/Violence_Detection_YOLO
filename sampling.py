import os
import random
import shutil
from multiprocessing import Pool

def sample_frames_for_video(video_folder, input_folder, output_folder, num_samples):
    video_path = os.path.join(input_folder, video_folder)
    if os.path.isdir(video_path):
        frames = os.listdir(video_path)
        sampled_frames = random.sample(frames, min(num_samples, len(frames)))
        sampled_output_folder = os.path.join(output_folder, video_folder)
        os.makedirs(sampled_output_folder, exist_ok=True)
        for frame in sampled_frames:
            shutil.copy(os.path.join(video_path, frame), sampled_output_folder)

def sample_frames_parallel(input_folder, output_folder, num_samples, num_workers=4):
    os.makedirs(output_folder, exist_ok=True)
    video_folders = os.listdir(input_folder)
    with Pool(num_workers) as pool:
        pool.starmap(
            sample_frames_for_video,
            [(video, input_folder, output_folder, num_samples) for video in video_folders]
        )

if __name__ == "__main__":
    # Example usage
    sample_frames_parallel(
        "Dataset/frames/violent", 
        "Dataset/sampled_frames/violent", 
        20, 
        num_workers=4
    )
    sample_frames_parallel(
        "Dataset/frames/nonviolent", 
        "Dataset/sampled_frames/nonviolent", 
        20, 
        num_workers=4
    )

import cv2
import os

violent_videos_folder = "Dataset/archive/violence_dataset/Violence"
nonviolent_videos_folder = "Dataset/archive/violence_dataset/NonViolence"
output_folder_violent = "Dataset/frames/violent"
output_folder_nonviolent = "Dataset/frames/nonviolent"

def extract_frames_from_videos(video_folder, output_folder, frame_rate=10):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        video_name = os.path.splitext(video_file)[0]
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_rate == 0:
                frame_filename = os.path.join(video_output_folder, f"{video_name}_frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        print(f"Frames extracted for {video_file}")

# Extract frames
extract_frames_from_videos(violent_videos_folder, output_folder_violent, frame_rate=10)
extract_frames_from_videos(nonviolent_videos_folder, output_folder_nonviolent, frame_rate=10)
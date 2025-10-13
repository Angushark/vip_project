from ultralytics import YOLO

import time
import os
import shutil
import glob
import torch


# Model array
MODELS = ["yolov8n.pt", "yolov10n.pt", "yolo11n.pt"]
# Video array
VIDEOS = ["video17.mp4", "video18.mp4", "video19.mp4", "video20.mp4"]
OUTPUT_FOLDER = "detect_n"

print(f"CUDA : {torch.cuda.is_available()}")
print(f"CUDA : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create output folder if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Counter for tracking progress
total_tasks = len(MODELS) * len(VIDEOS)
current_task = 0

# Loop through each model
for model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")

    model = YOLO(f"models/{model_name}")

    # Generate model prefix (e.g., yolo8n.pt -> v8n)
    model_prefix = model_name.replace("yolo", "v").replace(".pt", "")

    # Loop through each video
    for video_name in VIDEOS:
        current_task += 1
        print(f"\n[{current_task}/{total_tasks}] Processing: {model_name} + {video_name}")

        video_source = f"videos/{video_name}"

        # Generate unique temporary folder name for this processing
        temp_folder = f"temp_{model_prefix}_{video_name.replace('.mp4', '')}"

        # Generate output filename
        output_name = f"{model_prefix}_{video_name}"
        final_output_path = os.path.join(OUTPUT_FOLDER, output_name)

        # Track with persist=True
        results = model.track(
            source=video_source,
            classes=[0],
            show=False,
            save=True,
            stream=True,
            project=".",
            name=temp_folder,
            exist_ok=True,
            imgsz=640,
            conf=0.25,
            device=0,
            persist=True
        )

        start_time = time.time()

        for result in results:
            pass

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Find generated video files in temp folder
        video_files = glob.glob(f"{temp_folder}/*.avi") + glob.glob(f"{temp_folder}/*.mp4")

        if video_files:
            original_file = video_files[0]

            # Convert AVI to MP4 if needed
            if original_file.endswith('.avi'):
                import subprocess
                subprocess.run(['ffmpeg', '-y', '-i', original_file, '-c:v', 'libx264', final_output_path],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Copy to final destination
                shutil.copy2(original_file, final_output_path)

            # Clean up temporary folder
            shutil.rmtree(temp_folder, ignore_errors=True)

            print(f"  Time: {elapsed_time:.2f} sec")
            print(f"  Output: {final_output_path}")
        else:
            print(f"  Time: {elapsed_time:.2f} sec")
            print(f"  Warning: No video file generated")
            # Clean up temp folder if exists
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder, ignore_errors=True)

print(f"\n{'='*60}")
print(f"All tasks completed! Total: {total_tasks}")
print(f"{'='*60}")

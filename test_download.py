from ultralytics import YOLO
import cv2
import time
import os
import shutil
import glob
import torch


print(f"CUDA : {torch.cuda.is_available()}")
print(f"CUDA : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


model = YOLO("yolo11m.pt")

video_source = "video9.mp4"
output_name = "v11m_v9.mp4"
output_folder = "detect_m"

results = model.predict(
    source=video_source,
    classes=[0],
    show=False,
    save=True,
    stream=True,
    project=".",
    name=output_folder,
    exist_ok=True,
    device=0
)

start_time = time.time()

for result in results:
    pass

end_time = time.time()
elapsed_time = end_time - start_time


video_files = glob.glob(f"{output_folder}/*.avi") + glob.glob(f"{output_folder}/*.mp4")

if video_files:
    original_file = video_files[0]
    new_file = os.path.join(output_folder, output_name)

    
    if original_file.endswith('.avi'):
        import subprocess
        subprocess.run(['ffmpeg', '-y', '-i', original_file, '-c:v', 'libx264', new_file],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(original_file)
    else:
        shutil.move(original_file, new_file)

    print(f"final time use : {elapsed_time:.2f} sec")
    print(f": {new_file}")
else:
    print(f"final time use : {elapsed_time:.2f} sec")
  
    


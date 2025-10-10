from ultralytics import YOLO
import time
import torch


MODEL_NAME = "yolov8n.pt"
VIDEO_NAME = "video10.mp4"

print(f"CUDA : {torch.cuda.is_available()}")
print(f"CUDA : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")



try:
    model = YOLO(f"models/{MODEL_NAME}")
except Exception as e:
    
    model = YOLO(MODEL_NAME)  

video_source = f"videos/{VIDEO_NAME}"

start_time = time.time()

results = model.track(
    source=video_source,
    classes=[0],
    show=True,  
    save=False,
    persist=True,
    stream=True,
    device=0,
    imgsz=640,
    conf=0.25,
)


frame_count = 0
for result in results:
    frame_count += 1

end_time = time.time()
elapsed_time = end_time - start_time
print(f"final time use : {elapsed_time:.2f} sec ")
print(f"total frames : {frame_count}")

from ultralytics import YOLO
import time
import torch

print(f"CUDA : {torch.cuda.is_available()}")
print(f"CUDA : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")



model = YOLO("yolo11m.pt")

video_source = "video9.mp4"

start_time = time.time()

results = model.predict(
    source=video_source,
    classes=[0],
    show=True,  #
    save=False,
    stream=True,
    device=0,
    conf=0.25,
)


frame_count = 0
for result in results:
    frame_count += 1

end_time = time.time()
elapsed_time = end_time - start_time
print(f"final time use : {elapsed_time:.2f} sec ")
print(f"total frames : {frame_count}")

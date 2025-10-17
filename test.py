from ultralytics import YOLO
import cv2
import time
import torch


MODEL_NAME = "best.pt"
VIDEO_NAME = "video18.mp4"


print(f"CUDA : {torch.cuda.is_available()}")
print(f"CUDA : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")



model = YOLO(f"models/{MODEL_NAME}")

video_source = f"videos/{VIDEO_NAME}"

#

results = model.track(
    
    source=video_source,
    #classes=[0],
    show=False,
    save=False,
    stream=True,
    device=0,
    imgsz=640,
    conf=0.25,
)

cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

start_time = time.time()
frame_count = 0
total_yolo_time = 0

for result in results:
    yolo_start_time = time.time()

    frame_count += 1

    frame = result.plot()

    yolo_end_time = time.time()
    total_yolo_time += (yolo_end_time - yolo_start_time)

    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

end_time = time.time()
elapsed_time = end_time - start_time


avg_time_per_frame = (total_yolo_time / frame_count * 1000) if frame_count > 0 else 0
actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
theoretical_fps = frame_count / total_yolo_time if total_yolo_time > 0 else 0


print(f"total run time: {elapsed_time:.2f} seconds")
print(f"total frames: {frame_count}")
print(f"average processing time / frame: {avg_time_per_frame:.2f} ms")
print(f"actual FPS: {actual_fps:.2f}")
print(f"ideal FPS: {theoretical_fps:.2f}")

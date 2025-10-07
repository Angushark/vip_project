from ultralytics import YOLO
import cv2
import time
import torch

print(f"CUDA : {torch.cuda.is_available()}")
print(f"CUDA : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")



model = YOLO("yolo11m.pt")

video_source = "video9.mp4"

#
cap = cv2.VideoCapture(video_source)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = 1.0 / fps
cap.release()

results = model.predict(
    source=video_source,
    classes=[0],
    show=False,
    save=False,
    stream=True,
    device=0,
    conf=0.25,
)

cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

start_time = time.time()
frame_count = 0

for result in results:

    frame_count += 1

    
    frame = result.plot()
    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"final time use : {elapsed_time:.2f} sec ")
print(f"total frames : {frame_count}")
"""
Auto-labeling script - Automatically label videos using YOLO model
Saves individual frame images instead of video files
"""
import torch

# Monkey patch torch.load to disable weights_only for PyTorch 2.6
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load


import sys
from pathlib import Path
from datetime import datetime
import cv2
from ultralytics import YOLO
import config


def ensure_directories():
    """Ensure all necessary directories exist"""
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    config.AUTO_LABELED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_video_files():
    """Get all video files to process"""
    video_files = []
    for ext in config.VIDEO_FORMATS:
        video_files.extend(config.VIDEOS_DIR.glob(f"*{ext}"))
    return sorted(video_files)


def create_batch_folder():
    """Create new batch folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = f"{config.BATCH_NAME_PREFIX}_{timestamp}"
    batch_dir = config.AUTO_LABELED_DIR / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def save_frame_and_label(frame, label_data, frame_name):
    """Save individual frame as image and corresponding label directly to roboflow_sync"""

    # Ensure roboflow_sync directories exist
    config.ROBOFLOW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.ROBOFLOW_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save image
    image_path = config.ROBOFLOW_IMAGES_DIR / f"{frame_name}.jpg"
    cv2.imwrite(str(image_path), frame)

    # Save label
    label_path = config.ROBOFLOW_LABELS_DIR / f"{frame_name}.txt"
    with open(label_path, 'w') as f:
        f.write(label_data)

    return image_path, label_path


def process_video_with_yolo(video_path, model, batch_name):
    """Process video and save each frame with labels directly to roboflow_sync"""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    frame_count = 0
    saved_count = 0
    video_name = video_path.stem
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only frames according to stride
        if frame_count % config.VID_STRIDE == 0:
            # Check batch limit
            if config.MAX_IMAGES_PER_BATCH > 0 and saved_count >= config.MAX_IMAGES_PER_BATCH:
                print(f"   Reached batch limit ({config.MAX_IMAGES_PER_BATCH})")
                break

            # Run YOLO prediction on this frame
            results = model.predict(
                source=frame,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                max_det=config.MAX_DET,
                verbose=False
            )

            # Extract label data in YOLO format (without confidence for Roboflow)
            label_lines = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    # Get box data
                    cls = int(box.cls[0])

                    # Get normalized coordinates (YOLO format)
                    x_center, y_center, width, height = box.xywhn[0].tolist()

                    # Format: class x_center y_center width height (Roboflow standard YOLO format)
                    label_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Filter by minimum labels
            if config.MIN_LABELS_PER_IMAGE > 0 and len(label_lines) < config.MIN_LABELS_PER_IMAGE:
                frame_count += 1
                continue

            # Generate frame name with batch info
            frame_name = f"{batch_name}_{video_name}_frame{saved_count:06d}_{timestamp}"

            # Save frame and label
            label_data = "\n".join(label_lines) if label_lines else ""
            save_frame_and_label(frame, label_data, frame_name)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count


def run_auto_labeling():
    """Execute auto-labeling and save directly to roboflow_sync"""

    # Check if model exists
    if not config.MODEL_PATH.exists():
        print(f"Error: Model file not found at {config.MODEL_PATH}")
        print(f"Please place trained model in {config.MODELS_DIR} directory")
        sys.exit(1)

    # Check video directory
    ensure_directories()
    video_files = get_video_files()

    if not video_files:
        print(f"Error: No video files found in {config.VIDEOS_DIR}")
        print(f"Supported formats: {', '.join(config.VIDEO_FORMATS)}")
        sys.exit(1)

    print(f"Found {len(video_files)} video file(s)")

    # Generate batch name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = f"{config.BATCH_NAME_PREFIX}_{timestamp}"
    print(f"Batch name: {batch_name}")

    # Load YOLO model
    print(f"Loading model: {config.MODEL_PATH.name}")
    model = YOLO(str(config.MODEL_PATH))

    # Process each video
    total_frames = 0
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Processing: {video_path.name}")

        frame_count = process_video_with_yolo(video_path, model, batch_name)
        total_frames += frame_count
        print(f"   Saved {frame_count} frames")

    print(f"\nAuto-labeling complete!")
    print(f"Total saved: {total_frames} frames")
    print(f"Output location: {config.ROBOFLOW_SYNC_DIR}")
    print(f"   Images: {config.ROBOFLOW_IMAGES_DIR}")
    print(f"   Labels: {config.ROBOFLOW_LABELS_DIR}")
    print(f"\nNext step:")
    print(f"   Run sync_to_roboflow.py to upload to Roboflow")


if __name__ == "__main__":
    run_auto_labeling()

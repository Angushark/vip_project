"""
Configuration file - All adjustable parameters are centrally managed here
"""

from pathlib import Path

# ============ Path Configuration ============
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Input/Output paths
VIDEOS_DIR = DATASETS_DIR / "videos"
AUTO_LABELED_DIR = DATASETS_DIR / "auto_labeled"
ROBOFLOW_SYNC_DIR = DATASETS_DIR / "roboflow_sync"
ROBOFLOW_IMAGES_DIR = ROBOFLOW_SYNC_DIR / "images"
ROBOFLOW_LABELS_DIR = ROBOFLOW_SYNC_DIR / "labels"

# ============ YOLO Model Configuration ============
MODEL_PATH = MODELS_DIR / "v51027.pt"
DATA_YAML_PATH = BASE_DIR / "data.yaml"  # Path to data.yaml with class names
VID_STRIDE = 20                    # Video frame stride (extract 1 frame every N frames)
CONFIDENCE_THRESHOLD = 0.20        # Confidence threshold (0-1)
IOU_THRESHOLD = 0.45               # IoU threshold for NMS (0-1)
MAX_DET = 300                      # Maximum detections per image

# ============ Data Processing Configuration ============
BATCH_NAME_PREFIX = "auto_labeled"
MIN_LABELS_PER_IMAGE = 1           # Minimum labels per image (0 = keep all)
MAX_IMAGES_PER_BATCH = 1000        # Maximum images per batch (0 = no limit)

# ============ Roboflow Configuration ============
ROBOFLOW_WORKSPACE = "drowning-deteciotn"
ROBOFLOW_PROJECT = "drowning_detection_project-tft24"
ROBOFLOW_API_KEY = ""              # Leave empty to read from environment variable

# ============ Other Configuration ============
VERBOSE = False                    # Show verbose YOLO output
SAVE_VISUALIZATIONS = True         # Save visualization results (required for organize_data.py)
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
VIDEO_FORMATS = ['.mp4']

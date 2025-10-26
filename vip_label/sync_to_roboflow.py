"""
Roboflow sync script - Upload organized data to Roboflow using Python API
Maps class IDs to correct class names from data.yaml or model_artifacts.json
"""

import sys
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from roboflow import Roboflow
import config


def load_class_names():
    """Load class names from data.yaml (preferred) or model_artifacts.json (fallback)"""

    # Try data.yaml first
    data_yaml_path = Path(__file__).parent / "data.yaml"
    if data_yaml_path.exists():
        try:
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                class_names = data.get('names', [])
                print(f"Loaded class names from data.yaml: {class_names}")
                return class_names
        except Exception as e:
            print(f"Error loading data.yaml: {e}")

    # Fallback to model_artifacts.json
    artifacts_path = Path(__file__).parent / "model_artifacts.json"
    if artifacts_path.exists():
        try:
            with open(artifacts_path, 'r') as f:
                data = json.load(f)
                class_names = data.get('names', [])
                print(f"Loaded class names from model_artifacts.json: {class_names}")
                return class_names
        except Exception as e:
            print(f"Error loading class names: {e}")

    print(f"Warning: No class name file found (data.yaml or model_artifacts.json)")
    print(f"Using default class names (class_0, class_1, ...)")
    return None


def check_upload_directory():
    """Check if upload directory has data"""
    if not config.ROBOFLOW_IMAGES_DIR.exists() or not config.ROBOFLOW_LABELS_DIR.exists():
        return False, 0, 0

    images = list(config.ROBOFLOW_IMAGES_DIR.glob("*"))
    labels = list(config.ROBOFLOW_LABELS_DIR.glob("*.txt"))

    images = [img for img in images if img.suffix.lower() in config.IMAGE_FORMATS]

    return len(images) > 0, len(images), len(labels)


def get_api_key():
    """Get Roboflow API Key"""
    if config.ROBOFLOW_API_KEY:
        return config.ROBOFLOW_API_KEY

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if api_key:
        return api_key

    return None


def upload_to_roboflow():
    """Upload data to Roboflow using Python API"""

    # Load class names
    class_names = load_class_names()
    if class_names:
        print(f"Using class mapping:")
        for idx, name in enumerate(class_names):
            print(f"   {idx} -> {name}")
    else:
        print(f"Warning: No class names loaded, classes may not be mapped correctly")

    # Check upload directory
    has_data, image_count, label_count = check_upload_directory()
    if not has_data:
        print(f"\nError: No data in upload directory")
        print(f"Please run auto_label.py first")
        sys.exit(1)

    print(f"\nPreparing upload:")
    print(f"   Images: {image_count}")
    print(f"   Labels: {label_count}")

    if image_count != label_count:
        print(f"Warning: Image and label counts do not match")
        response = input(f"Continue upload? (y/n): ")
        if response.lower() != 'y':
            print(f"Upload cancelled")
            sys.exit(0)

    # Check API Key
    api_key = get_api_key()
    if not api_key:
        print(f"\nError: ROBOFLOW_API_KEY not set")
        print(f"Please set ROBOFLOW_API_KEY in config.py")
        print(f"Or set environment variable: set ROBOFLOW_API_KEY=your_api_key")
        sys.exit(1)

    # Generate batch name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = f"{config.BATCH_NAME_PREFIX}_{timestamp}"

    print(f"\nStarting upload to Roboflow...")
    print(f"   Workspace: {config.ROBOFLOW_WORKSPACE}")
    print(f"   Project: {config.ROBOFLOW_PROJECT}")
    print(f"   Batch name: {batch_name}")

    try:
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)

        # Get workspace and project
        workspace = rf.workspace(config.ROBOFLOW_WORKSPACE)
        project = workspace.project(config.ROBOFLOW_PROJECT)

        print(f"\nUploading {image_count} images...")

        # Upload images with annotations
        uploaded_count = 0
        failed_count = 0

        for image_file in config.ROBOFLOW_IMAGES_DIR.glob("*"):
            if image_file.suffix.lower() not in config.IMAGE_FORMATS:
                continue

            # Find corresponding label file
            label_file = config.ROBOFLOW_LABELS_DIR / f"{image_file.stem}.txt"

            if not label_file.exists():
                print(f"   Warning: No label for {image_file.name}")
                failed_count += 1
                continue

            try:
                # Upload with annotation as prediction (suggested labels for manual review)
                # This shows auto-labels as reference in Roboflow Annotating queue
                if class_names:
                    # Create class_map dictionary (class_id -> class_name)
                    class_map = {str(i): name for i, name in enumerate(class_names)}

                    project.upload(
                        image_path=str(image_file),
                        annotation_path=str(label_file),
                        batch_name=batch_name,
                        num_retry_uploads=3,
                        tag_names=[batch_name],
                        is_prediction=True,  # Upload as prediction (not final label)
                        annotation_labelmap=class_map  # Map class IDs to names
                    )
                else:
                    # Upload without class mapping
                    project.upload(
                        image_path=str(image_file),
                        annotation_path=str(label_file),
                        batch_name=batch_name,
                        num_retry_uploads=3,
                        tag_names=[batch_name],
                        is_prediction=True  # Upload as prediction
                    )

                uploaded_count += 1

                if uploaded_count % 10 == 0:
                    print(f"   Uploaded: {uploaded_count}/{image_count}")

            except Exception as e:
                print(f"   Failed to upload {image_file.name}: {e}")
                failed_count += 1

        print(f"\nUpload complete!")
        print(f"   Successfully uploaded: {uploaded_count}")
        if failed_count > 0:
            print(f"   Failed: {failed_count}")

        # Clean up any temporary directories
        temp_labels_dir = config.ROBOFLOW_SYNC_DIR / "temp_labels"
        if temp_labels_dir.exists():
            import shutil
            shutil.rmtree(temp_labels_dir)
            print(f"Cleaned up temporary files")

        # Ask to clear upload directory
        response = input(f"\nClear upload directory? (y/n): ")
        if response.lower() == 'y':
            for f in config.ROBOFLOW_IMAGES_DIR.glob("*"):
                f.unlink()
            for f in config.ROBOFLOW_LABELS_DIR.glob("*"):
                f.unlink()
            print(f"Upload directory cleared")

    except Exception as e:
        # Clean up temp files even if upload failed
        temp_labels_dir = config.ROBOFLOW_SYNC_DIR / "temp_labels"
        if temp_labels_dir.exists():
            import shutil
            shutil.rmtree(temp_labels_dir)
            print(f"Cleaned up temporary files")
        print(f"\nUpload failed!")
        print(f"Error: {e}")
        print(f"\nPossible solutions:")
        print(f"   1. Check if API Key is correct")
        print(f"   2. Check if workspace name is correct: {config.ROBOFLOW_WORKSPACE}")
        print(f"   3. Check if project name is correct: {config.ROBOFLOW_PROJECT}")
        print(f"   4. Verify you have upload permissions for this project")
        print(f"   5. Make sure your Roboflow project has the correct classes defined")
        print(f"   6. Check network connection")

        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    upload_to_roboflow()

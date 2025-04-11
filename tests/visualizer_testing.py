import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import sys
from pathlib import Path


current_path = Path(__file__).resolve()
project_parent = (
    current_path.parent.parent
)  # Adjust the times of .parent according to the actual structure


if project_parent.exists():
    sys.path.insert(0, str(project_parent))
else:
    raise FileNotFoundError(f"destination directory does not exist: {project_parent}")

try:
    from modules import MNISTEfficientNet, ModelVisualizer


except ImportError as e:
    print(f"Import failed: {e}")


if __name__ == "__main__":

    # initial model
    model = MNISTEfficientNet().model
    # Create a visualization instance
    visualizer = ModelVisualizer(
        model=model,
        input_size=(1, 1, 224, 224),  # Batch size x channel x height x width
    )
    # Perform visualization
    visualizer.visualize()

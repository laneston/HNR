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
    from modules import MNISTMultiDigitPredictor

except ImportError as e:
    print(f"Import failed: {e}")


if __name__ == "__main__":
    # Initialize predictor (requires pre training to save model)
    predictor = MNISTMultiDigitPredictor("model/mnist_efficientnet.pth")

    # Prediction Example Image
    image_path = "handwriting.jpg"

    try:
        # Obtain prediction results
        predictions = predictor.predict_digits(image_path)
        print(f"the forecast results: {predictions}")

        # Visualization processing procedure
        predictor.visualize_processing(image_path, predictions)

    except ValueError as e:
        print(f"process error: {str(e)}")

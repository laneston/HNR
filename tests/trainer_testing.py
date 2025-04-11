import torch
import sys
from pathlib import Path
import warnings

current_path = Path(__file__).resolve()
project_parent = (
    current_path.parent.parent
)  # Adjust the times of .parent according to the actual structure


if project_parent.exists():
    sys.path.insert(0, str(project_parent))
else:
    raise FileNotFoundError(f"destination directory does not exist: {project_parent}")

try:
    from modules import MNISTEfficientNet

except ImportError as e:
    print(f"Import failed: {e}")

if __name__ == "__main__":

    # Ignore unnecessary warnings
    warnings.filterwarnings("ignore")
    print(f"the torch version is: {torch.__version__}")
    # Initialize the trainer
    trainer = MNISTEfficientNet(use_amp=True)

    # Start training (parameters can be adjusted as needed)
    history, model = trainer.train(epochs=20, lr=1e-3, batch_size=16)

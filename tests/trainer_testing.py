import torch
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
    from modules import Model

except ImportError as e:
    print(f"Import failed: {e}")

if __name__ == "__main__":
    # input parameter is default or none.
    modelHandle = Model("default")
    # draw the neural network graph.
    # modelHandle.graph_draw()
    __model = modelHandle.base_train(16, "./data", 15)

    # save the model.
    torch.save(__model, "model/mnist_efficientnet.pth")

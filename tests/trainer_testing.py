import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import sys
from pathlib import Path

# import warnings

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
    # warnings.filterwarnings("ignore")
    print(f"the torch version is: {torch.__version__}")
    # Initialize the trainer
    trainer = MNISTEfficientNet(use_amp=True)

    # Start training (parameters can be adjusted as needed)
    history, model = trainer.train(epochs=20, lr=1e-3, batch_size=128)

    # 检查当前路径下是否存在该文件夹
    if not os.path.exists("model"):
        # 不存在则创建
        os.makedirs("model")
        print(f"✅ 目录 model 已创建")
    else:
        print(f"⚠️ 目录 model 已存在")
    
    # save the model after trained.
    torch.save(model.state_dict(), "model/mnist_efficientnet.pth")

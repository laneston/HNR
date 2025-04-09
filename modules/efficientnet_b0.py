import torch.nn as nn
import torch
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from torchvision import transforms
from torchviz import make_dot


class Model:
    def __init__(self, __weights=None):

        if isinstance(__weights, str) and __weights == "default":
            _model = efficientnet_b0(weights="EfficientNet_B0_Weights.IMAGENET1K_V1")
        else:
            _model = efficientnet_b0(weights=None)

        # Modify the first convolutional layer to adapt to a single channel input
        original_conv = _model.features[0][0]

        _model.features[0][0] = nn.Conv2d(
            in_channels=1,  # Change to single channel
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        nn.init.kaiming_normal_(
            _model.features[0][0].weight, mode="fan_out", nonlinearity="relu"
        )

        # Modify the classification layer
        _model.classifier[1] = nn.Linear(
            in_features=_model.classifier[1].in_features, out_features=10
        )

        nn.init.xavier_uniform_(_model.classifier[1].weight)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_ = _model.to(self._device)

    def base_train(self, batch_size=None, mnist_path=None, num_epochs=None) -> dict:
        if 0 < batch_size < 128 and batch_size % 8 == 0:
            _batch_size = batch_size
            print(f"the batch_size of training set is: {_batch_size}")
        else:
            raise TypeError(
                f"{batch_size} is not in compliance with regulations, should be rang 0 to 128, and must be multiple of 8."
            )

        if type(mnist_path) == str:
            _mnist_path = mnist_path
            print(f"the path of training set is: {_mnist_path}")
        else:
            raise TypeError(
                f"{mnist_path} is not in compliance with regulations, should be string format."
            )

        # Set random seeds to ensure repeatability
        torch.manual_seed(42)

        # Data preprocessing
        # self._transform = transforms.Compose(
        #     [
        #         transforms.Resize((224, 224)),
        #         transforms.Grayscale(
        #             num_output_channels=1
        #         ),  # Explicitly declare a single channel
        #         transforms.ToTensor(),
        #         # MNIST statistical values,
        #         # the original MNIST statistical values [0.1307, 0.3081] are applicable to 28x28 inputs,
        #         # but the distribution changes when enlarged to 224x224.
        #         transforms.Normalize(
        #             mean=[0.485],
        #             std=[0.229],  # Official Recommended Parameters[3](@ref)
        #         ),
        #     ]
        # )

        train_transform = transforms.Compose(
            [
                transforms.Resize(224),  # 网页1推荐的EfficientNet标准输入尺寸
                transforms.RandomRotation(15),  # 增加旋转增强
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移增强
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.456], std=[0.224]
                ),  # 网页8建议的归一化参数调整
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.456], std=[0.224]),
            ]
        )

        # Load training dataset
        _train_dataset = torchvision.datasets.MNIST(
            root=mnist_path, train=True, download=True, transform=train_transform
        )

        # _train_loader = DataLoader(_train_dataset, batch_size=batch_size, shuffle=True)
        _train_loader = DataLoader(
            _train_dataset,
            batch_size=_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        # Load test dataset
        _test_dataset = torchvision.datasets.MNIST(
            root=_mnist_path, train=False, download=True, transform=test_transform
        )
        # _test_loader = DataLoader(_test_dataset, batch_size=_batch_size, shuffle=True)

        _test_loader = DataLoader(
            _test_dataset, batch_size=_batch_size, num_workers=2, pin_memory=True
        )

        # Define loss function and optimizer
        # _criterion = nn.CrossEntropyLoss()
        # _optimizer = optim.Adam(self._model_.parameters(), lr=0.001)
        # _scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     _optimizer, "max", patience=2
        # )

        _criterion = nn.CrossEntropyLoss()
        _optimizer = optim.AdamW(
            self._model_.parameters(), lr=1e-3, weight_decay=1e-4
        )  # 网页8推荐的AdamW优化器
        _scheduler = optim.lr_scheduler.CosineAnnealingLR(
            _optimizer, T_max=30
        )  # 网页1提到的余弦退火策略

        best_acc = 0.0

        for epoch in range(num_epochs):
            self._model_.train()
            running_loss = 0.0

            for images, labels in _train_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)

                _optimizer.zero_grad()
                outputs = self._model_(images)
                loss = _criterion(outputs, labels)
                loss.backward()
                _optimizer.step()

                running_loss += loss.item()

            # verify
            self._model_.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in _test_loader:
                    images = images.to(self._device)
                    labels = labels.to(self._device)
                    outputs = self._model_(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            _scheduler.step(acc)

            print(
                f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Loss: {running_loss/len(_train_loader):.4f} | "
                f"Test Acc: {acc:.2f}%"
            )

            # Save the best model
            if acc > best_acc:
                best_acc = acc
                r_model = self._model_.state_dict()

        print(f"Best Test Accuracy: {best_acc:.2f}%")
        return r_model

    def graph_draw(self):
        # Virtual input of numerical values to generate calculation graphs
        x = torch.randn(1, 1, 224, 224, requires_grad=True)
        x = x.to(self._device)

        # Generate a computational graph
        output = self._model_(x)
        graph = make_dot(
            output,
            params=dict(list(self._model_.named_parameters()) + [("input", x)]),
            show_attrs=True,
            show_saved=True,
        )

        # Save as Image
        graph.render("model/efficientnet_b0_create_graph", format="svg", cleanup=True)

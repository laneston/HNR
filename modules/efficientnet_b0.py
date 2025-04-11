import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(
            x
        )  # Mathematical equivalence implementation[3,5](@ref)


class MNISTEfficientNet:
    def __init__(self, use_amp=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.scaler = GradScaler(enabled=use_amp)
        self.use_amp = use_amp
        self.writer = SummaryWriter()  # 初始化TensorBoard写入器

    def _build_model(self):
        """building and configure EfficientNet-B0 model"""
        model = efficientnet_b0(weights=None)  # Training from scratch

        # Modify the first layer convolution to adapt to single channel input
        original_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # nn.init.kaiming_normal_(
        #     model.features[0][0].weight,
        #     mode="fan_out",
        #     nonlinearity="silu",  # EfficientNet use Swish function.
        # )

        # according to torch version, it cannot use the silu activation function
        nn.init.kaiming_normal_(
            model.features[0][0].weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        model.features[0][2] = SiLU()

        # Modify the classification layer
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 10)

        # Classification layer initialization
        nn.init.normal_(model.classifier[1].weight, 0, 0.01)
        nn.init.zeros_(model.classifier[1].bias)

        return model.to(self.device)

    def _compute_mnist_stats(self, batch_size=128):
        # Initialize statistical variables
        channels_sum = 0.0
        channels_sq_sum = 0.0
        num_batches = 0
        total_pixels = 0

        loader, _ = self._get_dataloaders(batch_size)

        # Traverse all data
        for images, _ in loader:
            # Calculate the current batch statistic
            batch_pixels = images.size(0) * images.size(2) * images.size(3)
            channels_sum += torch.sum(images, dim=[0, 2, 3])  # 按通道求和
            channels_sq_sum += torch.sum(images**2, dim=[0, 2, 3])
            total_pixels += batch_pixels
            num_batches += 1

            # Printing progress
            if num_batches % 50 == 0:
                print(f"Processed {num_batches} batches...")

        # Final calculation
        mean = channels_sum / total_pixels
        std = torch.sqrt((channels_sq_sum / total_pixels) - (mean**2))

        return mean.numpy(), std.numpy()

    def _get_transforms(self):
        """
        Data augmentation and preprocessing
        Calculate the statistical value of MNIST at a size of 224x224
        Approximate value obtained from actual calculation
        (exact value can be calculated from the complete dataset)
        """
        # mean, std = self._compute_mnist_stats()
        # print(f"MNIST 224x224 statistical value:")
        # print(f"Mean: {mean[0]:.4f}")
        # print(f"Std:  {std[0]:.4f}")

        mean = [0.1307]
        std = [0.3081]

        train_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        return train_transform, test_transform

    def _get_dataloaders(self, batch_size=128, data_dir="./data"):
        """Get data loader"""
        train_transform, test_transform = self._get_transforms()

        train_set = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=train_transform
        )

        test_set = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=test_transform
        )

        # Automatically optimize numw_workers
        num_workers = min(8, os.cpu_count())

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        test_loader = DataLoader(
            test_set,
            # Use a larger batch size to accelerate validation
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, test_loader

    def train(self, epochs=20, lr=1e-3, batch_size=128):
        """Training process"""
        # Initialize the model and optimizer
        self.model.train()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999)
        )

        # Cosine annealing learning rate scheduling
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        # Get data loader
        train_loader, test_loader = self._get_dataloaders(batch_size)

        best_acc = 0.0
        history = {"train_loss": [], "test_acc": []}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 记录计算图（第一个epoch的第一个batch）
                if epoch == 0 and i == 0:
                    self.writer.add_graph(self.model, images)

                # Automatic Hybrid Precision Training
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                # Gradient update
                optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item() * images.size(0)

            # verify
            test_acc = self.evaluate(test_loader)
            scheduler.step()

            # Record the training process
            avg_loss = total_loss / len(train_loader.dataset)
            # Printing progress
            current_lr = optimizer.param_groups[0]["lr"]

            # 记录标量数据
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/test", test_acc, epoch)
            self.writer.add_scalar("Learning Rate", current_lr, epoch)

            # 记录参数分布
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f"Parameters/{name}", param, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

            history["train_loss"].append(avg_loss)
            history["test_acc"].append(test_acc)

            # Save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_weights = self.model.state_dict().copy()

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {test_acc:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

        # Load the best model weights
        self.model.load_state_dict(best_weights)
        print(f"\nBest Test Accuracy: {best_acc:.2f}%")

        # 关闭写入器
        self.writer.close()

        return history, best_weights

    def evaluate(self, loader):
        """Evaluation"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Using mixed precision inference
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

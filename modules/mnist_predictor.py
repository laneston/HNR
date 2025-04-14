import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from .efficientnet_b0 import MNISTEfficientNet


class MNISTMultiDigitPredictor:
    def __init__(
        self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"the path '{model_path}' not exists.")
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()

        # Same normalization parameters as during training
        self.mean = [0.1307]
        self.std = [0.3081]

    """Load pre trained model"""

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"the path '{model_path}' not exists.")

        # Ensure that the MNISTEfficientNet class is available
        model = MNISTEfficientNet().model

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device)

    """
    Using OpenCV to detect and split multiple numbers
    Return the sorted list of numerical regions (from left to right)
    """

    def _detect_and_crop_digits(self, image_path, min_contour_area=100):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"the path '{image_path}' not exists.")

        # Read images and preprocess them
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find all contours
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("No digital contour detected")

        # Filter and sort contours (from left to right)
        digit_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Filter small areas
            if area > min_contour_area and w > 5 and h > 10:
                digit_contours.append((x, y, w, h))

        # Sort by x-coordinate
        digit_contours = sorted(digit_contours, key=lambda c: c[0])

        # return self._crop_and_process_digits(gray, digit_contours)
        return digit_contours

    """Process each detected numerical region"""

    def _crop_and_process_digits(self, gray_img, contours):
        processed_digits = []

        for x, y, w, h in contours:
            # Extract a single numerical region
            digit_roi = gray_img[y : y + h, x : x + w]

            # Binarization processing
            _, thresh = cv2.threshold(
                digit_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )

            # Add boundary and center
            border_size = 20
            padded = cv2.copyMakeBorder(
                thresh,
                border_size,
                border_size,
                border_size,
                border_size,
                cv2.BORDER_CONSTANT,
                value=0,
            )

            # Zoom and center to 224x224
            scaled = self._scale_and_center(padded)

            # Convert to model input format
            tensor_img = transforms.ToTensor()(scaled).float()
            tensor_img = transforms.Normalize(self.mean, self.std)(tensor_img)

            processed_digits.append(tensor_img.to(self.device))

        return processed_digits

    """Maintain aspect ratio scaling and centering"""

    def _scale_and_center(self, img):

        h, w = img.shape
        scale = 200 / max(h, w)
        resized = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

        # Create a canvas and center it
        canvas = np.zeros((224, 224), dtype=np.uint8)
        y_start = (224 - resized.shape[0]) // 2
        x_start = (224 - resized.shape[1]) // 2
        canvas[
            y_start : y_start + resized.shape[0], x_start : x_start + resized.shape[1]
        ] = resized
        return canvas

    """Perform multi digit prediction"""

    def predict_digits(self, image_path) -> list:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"the path '{image_path}' not exists.")

        # Read images and preprocess them
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        # Detect and process digital regions
        digit_contours = self._detect_and_crop_digits(image_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_digits = self._crop_and_process_digits(gray, digit_contours)

        if not processed_digits:
            return []

        # Batch forecasting
        batch_tensor = torch.stack(processed_digits)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy().tolist()

    """Visualization processing procedure"""

    def visualize_processing(self, image_path, predictions=[]):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"the path '{image_path}' not exists.")

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Obtain contour information
        contours = self._detect_and_crop_digits(image_path)

        if len(contours) != len(predictions):
            raise ValueError(
                f"the len of the contours is: {len(contours)};the len of the predictions is: {len(predictions)}. they are not equal."
            )

        # Draw a detection box on the original image
        for i, (x, y, w, h) in enumerate(contours):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(
                img,
                str(predictions[i]),
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 255, 0),
                2,
            )

        cv2.namedWindow("Digit Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Digit Detection", 1200, 800)
        cv2.moveWindow("Digit Detection", 100, 100)  # 设置窗口位置
        cv2.imshow("Digit Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

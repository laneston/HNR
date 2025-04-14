
## _scale_and_center


````
    def _scale_and_center(self, img):

        h, w = img.shape
        scale = 200 / max(h, w)
````

- ​功能​：
  - 根据图像原始尺寸 (h, w) 计算缩放比例 scale，使图像的长边缩放到 200 像素（短边按比例缩放）。
- ​技术依据​：
  - 保持宽高比，避免图像拉伸形变。
  - 限制最大边为 200 像素，为后续居中填充至 224x224 提供边界空间。

````
        resized = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
````
- ​功能​：
  - 使用 OpenCV 的 resize 函数按比例缩小图像，并采用 ​区域插值（INTER_AREA）​​ 算法。
- ​参数解析​：
  - INTER_AREA​：通过像素区域关系重采样，适合图像缩小操作，可避免波纹（Moiré）效应。
  - 若图像需放大（如短边不足 200），则 INTER_AREA 等效于最近邻插值，可能产生锯齿，但此处设计目标为缩小，故合理。

````
        # Create a canvas and center it
        canvas = np.zeros((224, 224), dtype=np.uint8)
        y_start = (224 - resized.shape[0]) // 2
        x_start = (224 - resized.shape[1]) // 2
        canvas[
            y_start : y_start + resized.shape[0], x_start : x_start + resized.shape[1]
        ] = resized
        return canvas
````


- ​功能​：
  - 创建 224x224 黑色背景画布（像素值 0）。
  - 计算缩放后图像的左上角坐标 (x_start, y_start)，使其居中。
- 数学逻辑​：
  - 居中位置 = (画布尺寸 - 缩放后图像尺寸) // 2（整数除法确保像素对齐）。
  - 示例：若缩放后图像为 150x200，则 y_start=(224-150)//2=37，x_start=(224-200)//2=12。






## _crop_and_process_digits


此函数用于 ​从灰度图像中裁剪数字区域，并进行标准化预处理，最终生成适配深度学习模型输入的数据格式。

### 数字区域裁剪

````
    def _crop_and_process_digits(self, gray_img, contours):
        processed_digits = []

        for x, y, w, h in contours:
            # Extract a single numerical region
            digit_roi = gray_img[y : y + h, x : x + w]
````

- ​功能​：根据轮廓坐标 (x, y, w, h) 从灰度图像中提取单个数字区域（ROI）。
- ​技术依据​：
  - 基于 OpenCV 的 ROI 切片操作。
  - 输入 contours 需为经过过滤和排序的轮廓列表（如外接矩形坐标）。

### 二值化处理

````
            # Binarization processing
            _, thresh = cv2.threshold(
                digit_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
````

- ​功能​：
  - ​大津算法（OTSU）​​：自动计算最优阈值，适应光照不均场景。
  - ​反向二值化（BINARY_INV）​​：将数字区域转为白底黑字（原图暗区域变为白色，亮区域变为黑色），便于后续处理。
- ​意义​：
  - 消除灰度噪声，强化数字边缘特征。
  - 适配模型输入格式（如 MNIST 数据集通常为黑底白字）。

### 边界填充与居中

````
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
````

- ​功能​：在二值化图像四周添加 20 像素的黑色边框。
- ​技术依据​：
  - 使用 BORDER_CONSTANT 填充常数值（黑色）。
- 意义​：
  - 防止缩放时边缘像素丢失。
  - 为后续居中操作提供空间。


````
            # Zoom and center to 224x224
            scaled = self._scale_and_center(padded)
````

缩放与居中，[请参考 _scale_and_center](#_scale_and_center)



````
            # Convert to model input format
            tensor_img = transforms.ToTensor()(scaled).float()
            tensor_img = transforms.Normalize(self.mean, self.std)(tensor_img)
````

- 功能​：
  - ​ToTensor​：将图像转换为 [0,1] 范围的浮点张量。
  - ​归一化​：根据预训练模型的均值和标准差标准化数据（如 ImageNet 的 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）。
- ​意义​：
  - 适配 PyTorch 等框架的输入格式。
  - 提升模型收敛速度和泛化性。


````
            processed_digits.append(tensor_img.to(self.device))

        return processed_digits
````

将预处理后的图像张量移动到指定设备（如 GPU），然后将处理后的张量追加到列表 processed_digits 中，形成批量输入数据，并返回列表字典。


## detect_and_crop_digits

对图像进行滤波与二值化处理，并按图像中的数字轮廓进行切分、排序、存储。

````
    def detect_and_crop_digits(self, image_path, min_contour_area=100):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"the path '{image_path}' not exists.")

        # Read images and preprocess them
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
````

核验传入参数是否非法。

````
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
````

- ​功能​：将BGR彩色图像转换为灰度图。
- ​意义​：减少数据维度，简化计算，许多图像处理算法（如阈值化）仅需亮度信息。

````
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
````

- ​功能​：使用5×5高斯核平滑图像，消除高频噪声。
- ​意义​：避免噪声在后续阈值处理中造成干扰，提升鲁棒性。标准差为0时自动根据核大小计算（σ≈1.25）。

````
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
````

- ​功能​：对每个像素基于局部邻域计算阈值，生成二值图像。
- ​参数解析​：
  - ADAPTIVE_THRESH_GAUSSIAN_C：使用高斯加权邻域均值作为阈值。
  - THRESH_BINARY_INV：反向二值化（原值>阈值→0，否则→255）。
  - 块大小11×11：较大的邻域适应光照渐变。
  - 常数C=2：从均值中减去，用于微调阈值敏感度。
- 意义​：处理光照不均，突出暗目标（如文字、物体）为白色，背景为黑色，便于轮廓检测或OCR。

````
        # Find all contours
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("No digital contour detected")
````

- ​输入图像​：thresh.copy()
  - 使用经过自适应阈值处理后的二值图像（thresh）。
  - 调用 .copy() 是为了避免原图被 findContours 函数修改（OpenCV 的某些版本会原地修改输入图像）。
- ​参数解析​：
  - **cv2.RETR_EXTERNAL**​：
    - ​轮廓检索模式​：仅检测最外层轮廓（忽略嵌套在目标内部的轮廓）。
    - ​适用场景​：适用于提取独立物体（如文档中的文字块、图像中的独立对象）。
  - **cv2.CHAIN_APPROX_SIMPLE**​：
    - 轮廓压缩模式​：压缩水平、垂直和对角线方向的冗余点，仅保留轮廓的端点（如矩形只需存储4个顶点）。
    - 优势​：减少内存占用，同时保留关键形状信息。
- 输出结果​：
  - ​**contours**​：返回检测到的轮廓列表，每个轮廓由点的坐标数组（np.ndarray）表示。
  - **_**​：忽略层级信息（hierarchy），表示不处理轮廓之间的嵌套关系。

````
        # Filter and sort contours (from left to right)
        digit_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Filter small areas
            if area > min_contour_area and w > 5 and h > 10:
                digit_contours.append((x, y, w, h))
````

- 功能​：
  - 遍历所有轮廓，通过外接矩形和面积过滤掉噪声或无效轮廓。
  - **cv2.boundingRect**​：获取轮廓的最小外接矩形（左上角坐标 (x, y) 和宽高 (w, h)）。
  - ​**cv2.contourArea**​：基于格林公式计算轮廓面积（非负值）。
- 过滤逻辑​：
  - ​面积过滤​：剔除过小区域（如噪声点），min_contour_area 需根据实际图像调整。
  - ​尺寸过滤​：限制宽度>5、高度>10，确保轮廓是有效目标（如数字、文字）。

````
        # Sort by x-coordinate
        digit_contours = sorted(digit_contours, key=lambda c: c[0])
````

- ​功能​：按外接矩形的 ​x坐标​ 升序排列，实现 ​从左到右​ 的轮廓排序。
- 核心参数​：
  - key=lambda c: c[0]：以元组 (x, y, w, h) 中的第一个元素（x坐标）作为排序依据。
- 扩展性​：
  - 若需其他排序方式（如从上到下），可修改 key 为 c[1]（y坐标）。
  - 若需反向排序（从右到左），添加参数 reverse=True。

````
        return self._crop_and_process_digits(gray, digit_contours)
````

[_crop_and_process_digits 函数](#_crop_and_process_digits)​从灰度图像中裁剪数字区域，并进行标准化预处理，最终生成适配深度学习模型输入的数据格式。



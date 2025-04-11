##  Running

creat the env in ANACONDA, and use pip install the following plugins：

```
torch==2.3.1
torchvision==0.18.1
onnx==1.14.0
onnxruntime==1.15.1
opencv-contrib-python=4.11.0.86
pillow==11.0.0
```

### train the model

running the py file `tests/trainer_testing.py` can build the model of efficientnet B0 and saving in the path `./model`


## Q&A

```
oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
```

**- the first solution**

```
import warnings

warnings.filterwarnings("ignore")
```

**- the second solution**

```
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Should be placed before importing tensorflow
```


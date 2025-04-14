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

it maybe dosen't work.

**- the second solution**

```
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Should be placed before importing tensorflow
```

**SiLU function mathematical equivalence implementation**

````
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
````

Due to some torch versions not supporting this activation function, sigmoid function is used as an equivalent alternative.

## case

<div align="center"><img src="https://github.com/laneston/HNR/tree/main/doc/result20250414.png" width="50%"></div>




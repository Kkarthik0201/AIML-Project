'''-----------------------YOLO Model---------------------------'''

!pip install ultralytics --upgrade

data_yaml = """
train: /content/drive/MyDrive/train/images
val: /content/drive/MyDrive/valid/images
nc: 1
names:
  0: Pothole
"""

with open("/content/drive/MyDrive/data.yaml", "w") as f:
    f.write(data_yaml)


from ultralytics import YOLO
import glob
yaml_path = "/content/drive/MyDrive/data.yaml"
model = YOLO("yolo11n.pt")
images = 2025
batch_size = 4
steps_per_epoch = (images + batch_size - 1)//batch_size  # 507
epochs = 120
model.train(
    data=yaml_path,
    epochs=epochs,
    imgsz=640,
    batch=batch_size,
    name="yolo11_custom_small",
    project="/content/drive/MyDrive/yolo11_training",
    exist_ok=True,
    augment=True,
    patience=5
)



from ultralytics import YOLO
import glob
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

model_path = "/content/drive/MyDrive/yolo11_training/yolo11_custom_small/weights/best.pt"
model = YOLO(model_path)

test_images = "/content/drive/MyDrive/test/images/*"
test_images_list = glob.glob(test_images)

for img_path in test_images_list[1:25]:
    results = model.predict(source=img_path, conf=0.25, save=True)
    save_dir = Path(results[0].save_dir)
    result_img_path = save_dir / Path(img_path).name
    img = cv2.imread(str(result_img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()





from ultralytics import YOLO
import torch
import torch.nn as nn
import math
import os
from pathlib import Path


activation = "gelu"   # choose: 'gelu', 'mish', 'relu', 'lrelu', 'hardswish', 'elu'
pretrained = "yolo11n.pt"   # base pretrained model (keep file in runtime or path)
yaml_path = "/content/drive/MyDrive/data.yaml"  # your dataset yaml
project_folder = Path("/content/drive/MyDrive/yolo11_training")
train_name = "yolo11_custom_act"
batch_size = 4
imgsz = 640
epochs = 60
patience = 5
exist_ok = True
augment = True
# ----------------------------------------

# helper: create activation module by name
class MishModule(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def make_activation(act_name):
    act_name = act_name.lower()
    if act_name == 'gelu':
        return nn.GELU()
    if act_name == 'mish':
        # PyTorch has nn.Mish in recent versions; fallback to functional implementation
        if hasattr(nn, 'Mish'):
            return nn.Mish()
        else:
            # custom Mish using function
            class Mish(nn.Module):
                def forward(self, x):
                    return x * torch.tanh(F.softplus(x))
            return Mish()
    if act_name == 'relu':
        return nn.ReLU(inplace=True)
    if act_name == 'lrelu' or act_name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if act_name == 'hardswish' or act_name == 'hardswish':
        # Hardswish available in nn
        return nn.Hardswish()
    if act_name == 'elu':
        return nn.ELU()
    # default: SiLU
    return nn.SiLU()


import torch.nn.functional as F

def replace_activations(module, new_act_module_factory):
    """
    Recursively replace activation modules inside `module` whose class name is among
    known activation names. Returns number of replacements done.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        # class name
        cls_name = child.__class__.__name__.lower()
        # match common activation names
        if cls_name in ('silu', 'swish', 'mish', 'gelu', 'relu', 'leakyrelu', 'leakyrelu', 'hardswish', 'elu'):
            try:
                setattr(module, name, new_act_module_factory())
                replaced += 1
            except Exception:
                # fallback: set in _modules dict
                module._modules[name] = new_act_module_factory()
                replaced += 1
        else:
            # recurse
            replaced += replace_activations(child, new_act_module_factory)
    return replaced

# check device
device = 0 if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Load pretrained YOLO model
if not os.path.exists(pretrained):
    print(f"Warning: pretrained file {pretrained} not found in runtime. If you have it on Drive, give the full path.")

model = YOLO(pretrained)
print("Loaded YOLO model from:", pretrained)

# Build activation factory
new_act = make_activation(activation)

def new_act_factory():
    # return a *new instance* each call
    return make_activation(activation)


replacements = replace_activations(model.model, new_act_factory)
print(f"Replaced {replacements} activation modules with '{activation}'")

# Optional: print a brief module summary for activations (counts)
from collections import Counter
act_counts = Counter([m.__class__.__name__ for m in model.model.modules()])
print("Top module classes in model (sample):")
for k, v in list(act_counts.most_common(12)):
    print(f" - {k}: {v}")

# Now train using ultralytics API
print("Starting training with modified activations...")
model.train(
    data=yaml_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    name=train_name,
    project=str(project_folder),
    exist_ok=exist_ok,
    augment=augment,
    patience=patience,
    device=device
)

print("Training finished. Check outputs under:", project_folder / train_name)

from ultralytics import YOLO
import glob
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


model_path = "/content/drive/MyDrive/yolo11_training/yolo11_custom_act/weights/last.pt"
model = YOLO(model_path)


test_images = "/content/drive/MyDrive/test/images/*"
test_images_list = glob.glob(test_images)

for img_path in test_images_list[1:25]:
    results = model.predict(source=img_path, conf=0.25, save=True)
    save_dir = Path(results[0].save_dir)
    result_img_path = save_dir / Path(img_path).name
    img = cv2.imread(str(result_img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


''' ---------------------------THE END --------------------------'''

''' ---------------------------TRF-DETR --------------------------'''

!pip install -q rfdetr==1.2.1 supervision==0.26.1 roboflow
dataset_path = "/content/drive/MyDrive/potholes-dataset"
output_path = "/content/output"
from rfdetr import RFDETRSmall

model = RFDETRSmall()

model.train(
    dataset_dir=dataset_path,
    epochs=120,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=output_path,
    early_stopping=True
  )

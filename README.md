# Expression Detection — ResNet18 + In-Browser Demo

**Live:** https://expression-rieltzx.vercel.app/

I built this to learn how transfer learning works on a small dataset. The idea was simple: fine-tune ResNet18 on FER2013, then ship it as an in-browser app so anyone can try it with their webcam — no backend, nothing uploaded.

Two models run back-to-back in the browser. YOLOv11n-face handles face localization, then ResNet18 takes each crop and outputs probabilities for 7 emotions.

---

## What it detects

7 emotion classes from FER2013:

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## How it works

The pipeline runs entirely in your browser via ONNX Runtime (WASM):

1. Each webcam frame → resized to 640×640 → YOLOv11n-face detects all faces
2. Each bounding box → cropped, converted to grayscale, resized to 48×48
3. All crops batched → ResNet18 → softmax probabilities for each face
4. UI shows the top emotion + a donut chart of all 7 scores + an emotion timeline

Nothing leaves your device.

---

## Repo layout

```
expression-detection-resnet18/
├── app/                         # React + TypeScript frontend (Vercel)
│   ├── src/                     # inference, donut chart, timeline, App
│   ├── public/
│   │   └── models/
│   │       ├── face.onnx              # YOLOv11n-face detector (10 MB)
│   │       └── expression.onnx        # ResNet18 classifier (44 MB)
│   ├── vercel.json
│   └── package.json
└── training/                    # Python training pipeline
    ├── train.py                 # fine-tunes ResNet18 on FER2013
    ├── detect_webcam.py         # local webcam demo via OpenCV
    ├── expression_resnet18_pytorch_finetuned.pth
    └── archive/                 # FER2013 dataset (train/ + test/)
        ├── train/
        │   └── angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
        └── test/
            └── angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
```

---

## Running the web app locally

```bash
cd app
npm install
npm run dev
```

Opens at `http://localhost:5173`. Works in any browser with WebAssembly support — Chrome, Firefox, Edge.

---

## Training the model yourself

```bash
cd training
pip install torch torchvision opencv-python matplotlib
python train.py
```

Config: ResNet18 pretrained on ImageNet, input channel changed to 1 (grayscale), 60 epochs, Adam lr=0.0001, LR scheduler on plateau. Saves to `training/expression_resnet18_pytorch_finetuned.pth`.

To export the trained weights to ONNX for the browser:

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("expression_resnet18_pytorch_finetuned.pth"))
model.eval()

dummy = torch.randn(1, 1, 48, 48)
torch.onnx.export(
    model, dummy, "expression.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=11
)
```

To test locally without the browser:

```bash
python detect_webcam.py
```

Uses OpenCV Haar cascade for face detection and the `.pth` model directly — useful for checking accuracy before exporting.

---

## Stack

**Frontend:** React · TypeScript · Vite · ONNX Runtime Web · WebAssembly

**Training:** Python · PyTorch · ResNet18 · FER2013 · OpenCV

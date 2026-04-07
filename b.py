import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

# ==== Load Model & Class Names ====
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("expression_resnet18_pytorch_finetuned.pth", map_location=device))
model = model.to(device)
model.eval()

# ==== Image Transform for Webcam ====
transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==== OpenCV Face Detector ====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==== Webcam Loop ====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        # Preprocessing
        face_pil = transforms.ToPILImage()(face_img)
        face_tensor = transform_test(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            _, pred = torch.max(output, 1)
            label = class_names[pred.item()]

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("Facial Expression Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

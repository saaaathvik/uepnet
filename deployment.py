import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

device = torch.device("cpu")
EXIT_THRESHOLD = 0.8

class DepthRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.mobilenet_v3_small(weights=None)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        in_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

regressor = DepthRegressor()
regressor.load_state_dict(torch.load("uep_regressor_100k.pth", map_location="cpu", weights_only=False))
regressor.eval()

model_early = YOLO("yolov8m_early_100k.pt")
model_final = YOLO("yolov8m_final_100k.pt")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            difficulty = regressor(tensor).item()

        if difficulty < EXIT_THRESHOLD:
            results = model_early(frame, verbose=False)
            mode = "EARLY"
        else:
            results = model_final(frame, verbose=False)
            mode = "FINAL"

        annotated = results[0].plot()
        cv2.putText(annotated, f"EXIT: {mode}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        out.write(annotated)

    cap.release()
    out.release()

    return out_path

demo = gr.Interface(
    fn=predict,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Video(label="Annotated Output"),
    title="UEPNet: An Upfront Exit Prediction Framework for Early-Exit Neural Networks in Intelligent Vehicles",
)

demo.launch(share=True, css=".prose h1 { text-align: center !important; }")
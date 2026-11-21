"""Emotion & stress detector - Gradio web app (Render-ready)."""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

torch.set_num_threads(2)
device = torch.device("cpu")
EMO = ["Neutral", "Happy", "Sad", "Angry", "Fear"]
STR = ["Low Stress", "Moderate", "High Stress"]
WEIGHTS = os.environ.get("WEIGHTS_PATH", "multimodal_emotion_stress_pro.pth")
HF_NAME = "dima806/facial_emotions_image_detection"


class MultimodalEmotionStressNet(nn.Module):
    def __init__(self, num_emotions=5, num_stress=3):
        super().__init__()
        self.visual_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(), nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.audio_net = nn.Sequential(nn.Linear(52, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 64), nn.ReLU())
        self.physio_net = nn.Sequential(nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(192, 192), nn.Sigmoid())
        self.shared_fc = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.3))
        self.emotion_head = nn.Linear(128, num_emotions)
        self.stress_head = nn.Linear(128, num_stress)

    def forward(self, vis, aud, phy):
        fused = torch.cat([self.visual_net(vis), self.audio_net(aud), self.physio_net(phy)], dim=1)
        attn = self.attention(fused)
        shared = self.shared_fc(fused * attn)
        return self.emotion_head(shared), self.stress_head(shared), attn


# ---- load models once at startup ----
net = None
if os.path.exists(WEIGHTS):
    try:
        net = MultimodalEmotionStressNet().to(device)
        net.load_state_dict(torch.load(WEIGHTS, map_location=device))
        net.eval()
        print("Loaded custom weights:", WEIGHTS)
    except Exception as e:
        print("Custom weights failed to load, using HF model only:", e)
        net = None
else:
    print("No custom weights found, using HF model only.")

hf_proc = AutoImageProcessor.from_pretrained(HF_NAME)
hf_net = AutoModelForImageClassification.from_pretrained(HF_NAME).to(device).eval()
hf_labels = {int(k): str(v).lower() for k, v in hf_net.config.id2label.items()}

face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))


def hf_emotion(rgb_roi):
    inp = hf_proc(images=Image.fromarray(rgb_roi), return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(hf_net(**inp).logits, dim=-1)[0].numpy()
    out = np.zeros(5, dtype=np.float32)
    for i, p in enumerate(probs):
        l = hf_labels.get(i, "")
        if "neutral" in l: out[0] += p
        elif "happy" in l: out[1] += p
        elif "sad" in l: out[2] += p
        elif "angry" in l or "anger" in l: out[3] += p
        elif "fear" in l: out[4] += p
        elif "surprise" in l: out[4] += 0.5 * p; out[1] += 0.3 * p
        elif "disgust" in l: out[3] += 0.6 * p; out[2] += 0.4 * p
    s = out.sum()
    return out / s if s > 1e-6 else np.full(5, 0.2, dtype=np.float32)


def predict(rgb):
    if rgb is None:
        return None, {}, {}
    h, w, _ = rgb.shape
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.12, 4, minSize=(45, 45))
    if len(faces):
        x, y, bw, bh = [int(v) for v in max(faces, key=lambda b: b[2] * b[3])]
        found = True
    else:
        s = min(h, w) // 2
        x, y, bw, bh = (w - s) // 2, (h - s) // 2, s, s
        found = False
    roi = rgb[y:y + bh, x:x + bw]

    ep = hf_emotion(roi)
    sp = None
    if net is not None:
        v = cv2.resize(roi, (64, 64)).transpose(2, 0, 1).astype(np.float32) / 255.0
        v = (v - v.mean(axis=(1, 2), keepdims=True)) / (v.std(axis=(1, 2), keepdims=True) + 1e-6)
        with torch.no_grad():
            oe, os_, _ = net(torch.tensor(v)[None], torch.zeros(1, 52), torch.zeros(1, 16))
        ep = 0.85 * ep + 0.15 * torch.softmax(oe / 1.8, 1)[0].numpy()
        ep /= ep.sum()
        sp = torch.softmax(os_ / 1.8, 1)[0].numpy()

    # stress estimate derived from emotion (same heuristic as the notebook)
    hi = float(np.clip(0.15 + ep[3] * 0.75 + ep[4] * 0.70 + ep[2] * 0.35 - ep[1] * 0.25, 0.05, 0.90))
    lo = float(np.clip(0.15 + ep[0] * 0.65 + ep[1] * 0.60 - hi * 0.4, 0.05, 0.90))
    est = np.array([lo, max(0.05, 1 - hi - lo), hi], dtype=np.float32)
    sp = est if sp is None else 0.3 * sp + 0.7 * est
    sp = sp / sp.sum()

    out = rgb.copy()
    col = (0, 255, 0) if found else (255, 215, 0)
    cv2.rectangle(out, (x, y), (x + bw, y + bh), col, 2)
    cv2.putText(out, f"{EMO[int(ep.argmax())]} ({ep.max() * 100:.0f}%)", (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return out, {k: float(v) for k, v in zip(EMO, ep)}, {k: float(v) for k, v in zip(STR, sp)}


with gr.Blocks(title="Emotion & Stress Detector") as demo:
    gr.Markdown("# Emotion & Stress Detector\nAllow camera access. Stress is an **estimate** "
                "inferred from facial emotion, not a clinical measurement. Frames are processed in memory and not saved.")
    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera")
        ann = gr.Image(label="Detection", type="numpy")
    with gr.Row():
        emo = gr.Label(label="Emotion", num_top_classes=5)
        stress = gr.Label(label="Stress (estimate)", num_top_classes=3)
    cam.stream(predict, inputs=cam, outputs=[ann, emo, stress],
               stream_every=1.0, concurrency_limit=2)

demo.queue(max_size=10)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))

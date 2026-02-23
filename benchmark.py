import time
import torch
import numpy as np
import json

# -------- CONFIG --------
ALPHABET_MODEL_FILE = "alphabet_mlp.pt"
ALPHABET_LABELS_FILE = "alphabet_letters.json"

DYNAMIC_MODEL_FILE = "gesture_lstm_pytorch.pth"
DYNAMIC_LABELS_FILE = "gestures_dynamic.txt"

SEQ_LEN = 30
ALPHABET_INPUT_DIM = 63
DYNAMIC_INPUT_DIM = 126

device = torch.device("cpu")

# -------- Load Alphabet Model --------
class AlphaMLP(torch.nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

with open(ALPHABET_LABELS_FILE) as f:
    alpha_labels = json.load(f)

alpha_model = AlphaMLP(ALPHABET_INPUT_DIM, len(alpha_labels))
alpha_model.load_state_dict(torch.load(ALPHABET_MODEL_FILE, map_location=device))
alpha_model.eval()

# -------- Load Dynamic Model --------
class GestureLSTM(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, 128, 2, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

with open(DYNAMIC_LABELS_FILE) as f:
    dyn_labels = [l.strip() for l in f if l.strip()]

dyn_model = GestureLSTM(DYNAMIC_INPUT_DIM, len(dyn_labels))
dyn_model.load_state_dict(torch.load(DYNAMIC_MODEL_FILE, map_location=device))
dyn_model.eval()

# -------- Benchmark Alphabet --------
alpha_input = torch.randn(1, ALPHABET_INPUT_DIM)

runs = 100
start = time.perf_counter()
for _ in range(runs):
    with torch.no_grad():
        alpha_model(alpha_input)
end = time.perf_counter()

alpha_latency = ((end - start) / runs) * 1000
print(f"Alphabet Inference Time: {alpha_latency:.2f} ms per frame")

# -------- Benchmark Dynamic --------
dyn_input = torch.randn(1, SEQ_LEN, DYNAMIC_INPUT_DIM)

start = time.perf_counter()
for _ in range(runs):
    with torch.no_grad():
        dyn_model(dyn_input)
end = time.perf_counter()

dyn_latency = ((end - start) / runs) * 1000
print(f"Dynamic Inference Time: {dyn_latency:.2f} ms per 30-frame sequence")

# -------- Estimate FPS --------
total_latency = max(alpha_latency, dyn_latency)
fps = 1000 / total_latency
print(f"Approximate Max FPS: {fps:.2f}")

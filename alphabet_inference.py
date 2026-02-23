import json
import numpy as np
import torch
import torch.nn as nn
from collections import deque

MODEL_PATH = "alphabet_mlp.pt"
LETTERS_PATH = "alphabet_letters.json"
SMOOTH = 7
THRESHOLD = 0.6

class MLP(nn.Module):
    def __init__(self, input_dim, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, classes)
        )
    def forward(self, x):
        return self.net(x)

class AlphabetPredictor:
    def __init__(self):
        with open(LETTERS_PATH, "r") as f:
            self.letters = json.load(f)

        self.model = MLP(63, len(self.letters))
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

        self.window = deque(maxlen=SMOOTH)
        self.conf_window = deque(maxlen=SMOOTH)

    def predict(self, vec):
        x = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().numpy()

        idx = np.argmax(probs)
        conf = probs[idx]
        letter = self.letters[idx]

        self.window.append(letter)
        self.conf_window.append(conf)

        if len(self.window) == SMOOTH and np.mean(self.conf_window) > THRESHOLD:
            # majority vote
            values, counts = np.unique(self.window, return_counts=True)
            return values[counts.argmax()], np.mean(self.conf_window)

        return None, None

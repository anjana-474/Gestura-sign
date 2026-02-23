# live_alphabet_test.py
# Live tester for alphabet MLP (no Flask). Matches training architecture (includes Dropout layer).
# Run: python live_alphabet_test.py

import time
import json
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

from handpoints_utils import create_hands, extract_normalized_landmarks

# === CONFIG ===
MODEL_PATH = "alphabet_mlp.pt"
LETTERS_PATH = "alphabet_letters.json"
SMOOTH_WINDOW = 7        # number of frames for majority vote
CONF_THRESHOLD = 0.60
CAM_IDX = 0
# ==============

# ---- model (must match training arch exactly) ----
class MLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # net.0
            nn.ReLU(),                  # net.1
            nn.Dropout(0.2),            # net.2  <-- important (matches train)
            nn.Linear(128, 64),         # net.3
            nn.ReLU(),                  # net.4
            nn.Linear(64, n_classes)    # net.5
        )
    def forward(self, x):
        return self.net(x)

def load_model():
    # load letters mapping
    try:
        with open(LETTERS_PATH, "r", encoding="utf-8") as fh:
            letters = json.load(fh)
    except FileNotFoundError:
        raise RuntimeError(f"Letters mapping not found: {LETTERS_PATH}")

    # init model and load weights
    model = MLP(63, len(letters))
    # If your PyTorch version supports weights_only=True and you'd like more safety:
    # state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, letters

def main():
    model, letters = load_model()

    cap = cv2.VideoCapture(CAM_IDX)
    if not cap.isOpened():
        # try DirectShow on Windows if default fails
        try:
            cap = cv2.VideoCapture(CAM_IDX, cv2.CAP_DSHOW)
        except Exception:
            pass

    if not cap.isOpened():
        print(f"Cannot open camera index {CAM_IDX}. Try camera diagnostic script or change CAM_IDX.")
        return

    hands = create_hands(max_num_hands=1)
    win = deque(maxlen=SMOOTH_WINDOW)
    confs = deque(maxlen=SMOOTH_WINDOW)

    print("Press 'q' to quit. Showing predictions on window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pred_text = "No hand"
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            vec = extract_normalized_landmarks(lm).astype(np.float32)
            with torch.no_grad():
                logits = model(torch.from_numpy(vec).unsqueeze(0))
                probs = torch.softmax(logits, dim=1)[0].numpy()
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            letter = letters[idx]
            win.append(letter)
            confs.append(conf)

            if len(win) == SMOOTH_WINDOW and np.mean(confs) >= CONF_THRESHOLD:
                vals, counts = np.unique(list(win), return_counts=True)
                pred = vals[counts.argmax()]
                pred_conf = float(np.mean(confs))
                pred_text = f"{pred} ({pred_conf:.2f})"
            else:
                pred_text = f"{letter} (unstable {conf:.2f})"

        # draw
        cv2.putText(frame, "Live alphabet test", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40,200,40), 2)
        cv2.putText(frame, f"Pred: {pred_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10,10,200), 2)
        cv2.imshow("Live Alphabet Tester", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# realtime_test.py

import cv2
import numpy as np
import torch
import torch.nn as nn
import time

from handpoints_utils import create_hands, extract_two_hand_vector, draw_hand_landmarks


SEQ_LEN = 30
FEATURE_DIM = 126
MODEL_FILE = "gesture_lstm_pytorch.pth"
GESTURE_LABELS_FILE = "gestures_dynamic.txt"


class GestureLSTM(nn.Module):
    def __init__(self, input_size, num_classes,
                 hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(
            hidden_size * (2 if bidirectional else 1),
            num_classes
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]    # (B, H*dirs)
        logits = self.fc(last)
        return logits


def load_gesture_labels(path):
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels


def record_sequence(cap, hands, seq_len=30):
    """
    Records one dynamic 2-hand sequence of length seq_len.
    Returns numpy array of shape (seq_len, 126) or None if user quits.
    """
    # 3s countdown
    start_time = time.time()
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        remaining = 3 - int(time.time() - start_time)
        cv2.putText(frame,
                    f"Recording starts in {remaining}s",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.imshow("Realtime Gesture Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    sequence = []
    print("Recording sequence... perform gesture now!")

    while len(sequence) < seq_len:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        info_text = f"Recording frame {len(sequence)+1}/{seq_len}"

        if results.multi_hand_landmarks:
            vec126 = extract_two_hand_vector(results)
            if vec126 is not None:
                sequence.append(vec126)

            # draw all hands
            for hand_lm in results.multi_hand_landmarks:
                draw_hand_landmarks(frame, hand_lm)
        else:
            info_text += " | No hand!"

        cv2.putText(frame,
                    info_text,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        cv2.imshow("Realtime Gesture Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None

    sequence = np.array(sequence, dtype=np.float32)  # (T, 126)

    # pad if needed
    if sequence.shape[0] < seq_len:
        last = sequence[-1]
        pad_len = seq_len - sequence.shape[0]
        pad = np.tile(last, (pad_len, 1))
        sequence = np.concatenate([sequence, pad], axis=0)

    return sequence


def main():
    # Load labels
    gesture_labels = load_gesture_labels(GESTURE_LABELS_FILE)
    num_classes = len(gesture_labels)
    print("Loaded gesture labels:", gesture_labels)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = GestureLSTM(
        input_size=FEATURE_DIM,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    hands = create_hands(max_num_hands=2)

    current_pred = ""
    current_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Show last prediction
        cv2.putText(frame,
                    f"Last prediction: {current_pred} ({current_conf:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.putText(frame,
                    "Press 'r' to record gesture | 'q' to quit",
                    (10, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 0), 2)

        cv2.imshow("Realtime Gesture Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            seq = record_sequence(cap, hands, seq_len=SEQ_LEN)
            if seq is None:
                break

            # Predict
            x = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, T, F)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                idx = torch.argmax(probs).item()
                conf = probs[idx].item()

            current_pred = gesture_labels[idx]
            current_conf = conf
            print(f"Predicted: {current_pred} ({conf:.3f})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

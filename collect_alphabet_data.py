# collect_alphabet_data.py
# Collect single-frame landmark vectors for alphabet letters (A-Y, excluding J and Z).
# Output: data_alphabet/<letter>/*.npy  (each file is a 63-dim numpy array)
#
# NOTE: J and Z are intentionally excluded because they require dynamic motion
# (tracing) rather than a single static pose. If you later want to support them,
# collect short-motion sequences instead of single frames.

import cv2
import numpy as np
import os
import time
from handpoints_utils import create_hands, extract_normalized_landmarks

# ---------- CONFIG ----------
DATA_DIR = "data_alphabet"
# All lowercase letters except 'j' and 'z'
LETTERS = "y"
SAMPLES_PER_LETTER = 300    # adjust: 150-400 depending on time / diversity
CAP_DEVICE = 0
SHOW_WINDOW = True
# ----------------------------

os.makedirs(DATA_DIR, exist_ok=True)
for L in LETTERS:
    os.makedirs(os.path.join(DATA_DIR, L), exist_ok=True)

def main():
    cap = cv2.VideoCapture(CAP_DEVICE)
    hands = create_hands(max_num_hands=1)  # single-hand alphabet
    print("Starting alphabet collection. Letters:", LETTERS)
    try:
        for letter in LETTERS:
            print(f"\n=== Collecting letter: '{letter}' ===")
            saved = len([n for n in os.listdir(os.path.join(DATA_DIR, letter)) if n.endswith(".npy")])
            print(f"Already saved: {saved}. Target per letter: {SAMPLES_PER_LETTER}.")
            time.sleep(1.0)
            while saved < SAMPLES_PER_LETTER:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                status = f"Letter: {letter} | Saved: {saved}/{SAMPLES_PER_LETTER}"

                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    vec63 = extract_normalized_landmarks(lm)
                    if vec63 is not None:
                        # Save the vector
                        path = os.path.join(DATA_DIR, letter, f"{saved:04d}.npy")
                        np.save(path, vec63)
                        saved += 1
                        status += " | SAVED"
                else:
                    status += " | No hand detected"

                if SHOW_WINDOW:
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    cv2.imshow("Collect Alphabet Data (press 'q' to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User requested exit. Stopping early.")
                        return
    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        print("Collection finished (or stopped). Check the", DATA_DIR, "folder.")

if __name__ == "__main__":
    main() 



# collect_dynamic_data.py

import cv2
import numpy as np
import os
import time

from handpoints_utils import create_hands, extract_two_hand_vector, draw_hand_landmarks

# ------------ CONFIG ------------
GESTURES = [
    "family"
]

SEQUENCES_PER_GESTURE = 30      # increase to 40 later if you want
SEQ_LEN = 30                    # frames per sequence (~2–3 seconds)
DATA_DIR = "data_dynamic"
# --------------------------------

os.makedirs(DATA_DIR, exist_ok=True)
for g in GESTURES:
    os.makedirs(os.path.join(DATA_DIR, g), exist_ok=True)


def main():
    cap = cv2.VideoCapture(0)
    hands = create_hands(max_num_hands=2)

    for gesture in GESTURES:
        print(f"\n=== Collecting for gesture: {gesture} ===")

        for seq_idx in range(SEQUENCES_PER_GESTURE):
            print(f"\nPrepare for sequence {seq_idx+1}/{SEQUENCES_PER_GESTURE} of '{gesture}'")

            # ---- 3-second countdown ----
            start_time = time.time()
            while time.time() - start_time < 3:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)

                remaining = 3 - int(time.time() - start_time)
                cv2.putText(frame,
                            f"Get ready: {gesture} | Starts in {remaining}s",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                cv2.imshow("Dynamic Data Collection (2 Hands)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Exited by user.")
                    return

            # ---- Record one sequence ----
            sequence = []
            print(f"Recording sequence {seq_idx+1}... Perform the gesture NOW with your 2 hands!")

            while len(sequence) < SEQ_LEN:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                info_text = f"{gesture} | Seq {seq_idx+1}/{SEQUENCES_PER_GESTURE} | Frame {len(sequence)+1}/{SEQ_LEN}"

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

                cv2.imshow("Dynamic Data Collection (2 Hands)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Exited by user.")
                    return

            # ---- Save sequence ----
            sequence = np.array(sequence)  # shape: (len_seq, 126)

            # If fewer frames captured, pad with last frame
            if sequence.shape[0] < SEQ_LEN:
                last = sequence[-1]
                pad_len = SEQ_LEN - sequence.shape[0]
                pad = np.tile(last, (pad_len, 1))
                sequence = np.concatenate([sequence, pad], axis=0)

            save_path = os.path.join(DATA_DIR, gesture, f"seq_{seq_idx}.npy")
            np.save(save_path, sequence)
            print(f"Saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nAll sequences collected!")


if __name__ == "__main__":
    main()

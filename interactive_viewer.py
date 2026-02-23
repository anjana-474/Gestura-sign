import os
import numpy as np
import cv2
import mediapipe as mp

# ===== CONFIG =====
DATA_DIR = r"data_dynamic/call"   # folder with seq_0.npy, seq_1.npy, ...
WINDOW = 600                          # canvas size
FEATURES = 126                        # 2 hands * 63
# ==================


mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS


def normalize(coords_21x3, w, h):
    """
    coords_21x3: (21, 3) normalized around wrist (same style as your collector)
    Returns (21, 2) in pixel coordinates on a w×h canvas.
    """
    xy = coords_21x3[:, :2]  # (21, 2)

    # scale and center
    scale = min(w, h) * 0.4
    xy_scaled = xy * scale
    center = np.array([w / 2.0, h / 2.0])
    xy_shifted = xy_scaled + center

    return xy_shifted


def draw_hand(canvas, pts_21x2, color):
    """
    Draw one hand (21 keypoints + connections) on canvas.
    """
    # joints
    for x, y in pts_21x2:
        cv2.circle(canvas, (int(x), int(y)), 4, color, -1)

    # connections
    for conn in HAND_CONNECTIONS:
        i, j = conn
        x1, y1 = pts_21x2[i]
        x2, y2 = pts_21x2[j]
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


def load_all_sequences(data_dir):
    """
    Loads all .npy sequences from a directory.
    Each file is expected to be shape: (T, 126)
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in {data_dir}")

    seqs = []
    for f in files:
        path = os.path.join(data_dir, f)
        arr = np.load(path)  # (T, 126)
        seqs.append((f, arr))

    print(f"Loaded {len(seqs)} sequences from {data_dir}")
    return seqs


def main():
    seqs = load_all_sequences(DATA_DIR)
    seq_i = 0
    frame_i = 0

    print("\nControls:")
    print("  Right arrow or 'd'  → next frame")
    print("  Left arrow  or 'a'  → previous frame")
    print("  n                   → next sequence")
    print("  b                   → previous sequence")
    print("  q                   → quit\n")

    while True:
        name, seq = seqs[seq_i]
        seq_len = seq.shape[0]
        frame_i = max(0, min(frame_i, seq_len - 1))

        frame_vec = seq[frame_i]  # (126,)

        # split into two hands, (63,) each → (21,3)
        hand1_vec = frame_vec[:63]
        hand2_vec = frame_vec[63:]

        hand1_coords = hand1_vec.reshape(21, 3)
        hand2_coords = hand2_vec.reshape(21, 3)

        # blank canvas
        canvas = np.zeros((WINDOW, WINDOW, 3), dtype=np.uint8)

        # normalize to canvas coords
        h1_xy = normalize(hand1_coords, WINDOW, WINDOW)
        h2_xy = normalize(hand2_coords, WINDOW, WINDOW)

        # draw both hands
        draw_hand(canvas, h1_xy, (0, 255, 0))     # left-ish hand
        draw_hand(canvas, h2_xy, (255, 0, 0))     # right-ish hand

        # info text
        text = f"{name} | Seq {seq_i+1}/{len(seqs)} | Frame {frame_i+1}/{seq_len}"
        cv2.putText(canvas, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Interactive Sequence Viewer", canvas)

        # IMPORTANT: don't mask with & 0xFF if you want arrow keys
        key = cv2.waitKey(0)

        # ----- quitting -----
        if key == ord('q'):
            break

        # ----- sequence navigation -----
        elif key == ord('n'):          # next sequence
            seq_i = (seq_i + 1) % len(seqs)
            frame_i = 0

        elif key == ord('b'):          # previous sequence
            seq_i = (seq_i - 1) % len(seqs)
            frame_i = 0

        # ----- frame navigation (arrow keys) -----
        elif key == 2555904:           # right arrow
            frame_i = min(seq_len - 1, frame_i + 1)

        elif key == 2424832:           # left arrow
            frame_i = max(0, frame_i - 1)

        # ----- frame navigation (WASD backup) -----
        elif key == ord('d'):          # next frame
            frame_i = min(seq_len - 1, frame_i + 1)

        elif key == ord('a'):          # previous frame
            frame_i = max(0, frame_i - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

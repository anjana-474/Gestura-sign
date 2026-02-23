# prepare_dynamic_dataset.py

import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "data_dynamic"

# Desired split ratios
TEST_RATIO = 0.15
VAL_RATIO  = 0.15   # of the full data (not of train)


def load_sequences(data_dir):
    data_dir = Path(data_dir)

    gestures = [
        d.name for d in sorted(data_dir.iterdir())
        if d.is_dir() and any(f.suffix == ".npy" for f in d.iterdir())
    ]

    print("Detected gesture classes:", gestures)

    X_list = []
    y_list = []

    for label_idx, gesture in enumerate(gestures):
        gdir = data_dir / gesture
        npy_files = sorted([f for f in gdir.glob("*.npy")])
        print(f"{gesture}: {len(npy_files)} sequences")

        for f in npy_files:
            arr = np.load(f).astype(np.float32)   # (T, F)
            X_list.append(arr)
            y_list.append(label_idx)

    X = np.stack(X_list, axis=0)  # (N, T, F)
    y = np.array(y_list, dtype=np.int64)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y, gestures


def main():
    if not os.path.exists(DATA_DIR):
        print(f"{DATA_DIR} not found")
        return

    X, y, gestures = load_sequences(DATA_DIR)

    N, SEQ_LEN, FEATURE_DIM = X.shape
    print(f"\nDetected SEQ_LEN={SEQ_LEN}, FEATURE_DIM={FEATURE_DIM}")

    # ---- Train / Test / Val split ----
    # 1) split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        stratify=y,
        random_state=42
    )

    # 2) split temp into train + val
    # remaining ratio = 1 - TEST_RATIO
    val_ratio_adjusted = VAL_RATIO / (1.0 - TEST_RATIO)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        stratify=y_temp,
        random_state=42
    )

    print("\nFinal split sizes:")
    print("  Train:", X_train.shape[0])
    print("  Val  :", X_val.shape[0])
    print("  Test :", X_test.shape[0])

    # ---- Save to .npz ----
    np.savez(
        "dynamic_dataset.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        gestures=np.array(gestures)
    )

    print("\nSaved split dataset to dynamic_dataset.npz")


if __name__ == "__main__":
    main()

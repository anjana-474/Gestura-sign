# evaluate_all_auto.py
"""
Auto evaluation for Alphabet (MLP) + Dynamic (LSTM) models.

Place this file in your project root (same folder as app.py).
It will:
 - auto-detect model filenames in the current folder
 - use data_alphabet_test/ (fallback data_alphabet/) and data_dynamic_test/ (fallback data_dynamic/)
 - output confusion matrices, classification reports and evaluation/report.json in evaluation/
"""

import os, glob, json, sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ----------------------------
OUT_DIR = "evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def find_alphabet_model():
    candidates = ["alphabet_mlp.pt", "alphabet_mlp.pth"]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def find_alphabet_labels():
    if os.path.exists("alphabet_letters.json"):
        return "alphabet_letters.json"
    # look for json with 'alphabet' in name
    for f in glob.glob("*.json"):
        if "alphabet" in f.lower() or "letters" in f.lower():
            return f
    return None

def find_dynamic_model():
    # heuristic search
    patterns = ["*lstm*.pt","*lstm*.pth","gesture*.pt","gesture*.pth","*bilstm*.pt","*bilstm*.pth"]
    for pat in patterns:
        found = glob.glob(pat)
        if found:
            return found[0]
    return None

def find_dynamic_labels():
    if os.path.exists("gestures_dynamic.txt"):
        return "gestures_dynamic.txt"
    for f in glob.glob("*.txt"):
        if "gesture" in f.lower() or "gestur" in f.lower() or "dynamic" in f.lower():
            return f
    return None

def save_confusion(cm, labels, name):
    plt.figure(figsize=(max(6,len(labels)*0.35), max(5, len(labels)*0.35)))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"{name} confusion")
    png = os.path.join(OUT_DIR, f"{name}_confusion.png")
    plt.savefig(png, bbox_inches="tight", dpi=150)
    plt.close()
    csv = os.path.join(OUT_DIR, f"{name}_confusion.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(csv)
    return png, csv

def save_text(text, fname):
    with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
        f.write(text)

# ----------------------------
# Model classes (must match training architecture)
# ----------------------------
class AlphaMLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x): return self.net(x)

class GestureLSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

# ----------------------------
# Alphabet evaluation
# ----------------------------
def eval_alphabet():
    model_file = find_alphabet_model()
    labels_file = find_alphabet_labels()
    data_dir = "data_alphabet_test" if os.path.exists("data_alphabet_test") else ("data_alphabet" if os.path.exists("data_alphabet") else None)

    if model_file is None or labels_file is None or data_dir is None:
        print("[Alphabet] missing model/labels/data. Skipping alphabet eval.")
        return None

    print("[Alphabet] model:", model_file, "labels:", labels_file, "data:", data_dir)

    # load labels mapping (saved mapping may contain train order)
    with open(labels_file, "r", encoding="utf-8") as fh:
        saved_labels = json.load(fh)

    # load test files
    X, y = [], []
    folder_labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not folder_labels:
        print("[Alphabet] no subfolders in", data_dir)
        return None

    for idx, lab in enumerate(folder_labels):
        folder = os.path.join(data_dir, lab)
        for f in sorted(glob.glob(os.path.join(folder, "*.npy"))):
            arr = np.load(f)
            if arr.shape[0] != 63:
                print(f"[Alphabet] skip {f} (shape {arr.shape})")
                continue
            X.append(arr.astype(np.float32))
            y.append(idx)

    if len(X) == 0:
        print("[Alphabet] no valid .npy files found")
        return None

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)

    # load model
    model = AlphaMLP(63, len(saved_labels))
    st = torch.load(model_file, map_location="cpu")
    model.load_state_dict(st)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        preds = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, target_names=folder_labels, zero_division=0)
    cm = confusion_matrix(y, preds)

    png, csv = save_confusion(cm, folder_labels, "alphabet")
    save_text(report, "alphabet_classification_report.txt")

    summary = {"accuracy": float(acc), "n_samples": int(len(y)), "labels": folder_labels,
               "confusion_png": png, "confusion_csv": csv, "report_file": os.path.join(OUT_DIR,"alphabet_classification_report.txt")}
    print("[Alphabet] done. acc=%.2f%%" % (acc*100))
    return summary

# ----------------------------
# Dynamic evaluation
# ----------------------------
def eval_dynamic():
    model_file = find_dynamic_model()
    labels_file = find_dynamic_labels()
    data_dir = "data_dynamic_test" if os.path.exists("data_dynamic_test") else ("data_dynamic" if os.path.exists("data_dynamic") else None)

    if model_file is None or labels_file is None or data_dir is None:
        print("[Dynamic] missing model/labels/data. Skipping dynamic eval.")
        return None

    print("[Dynamic] model:", model_file, "labels:", labels_file, "data:", data_dir)

    # read label lines
    with open(labels_file, "r", encoding="utf-8") as fh:
        lines = [l.strip() for l in fh.readlines() if l.strip()]

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not classes:
        print("[Dynamic] no class folders in", data_dir)
        return None

    X, y = [], []
    # expect each .npy to be (T, feat_dim). We'll infer feat_dim from first file.
    feat_dim = None
    seq_len = None
    # gather
    for idx, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for f in sorted(glob.glob(os.path.join(folder, "*.npy"))):
            seq = np.load(f)
            if seq.ndim != 2:
                print(f"[Dynamic] skip {f} ndim={seq.ndim}")
                continue
            if feat_dim is None:
                feat_dim = seq.shape[1]
            if seq.shape[1] != feat_dim:
                print(f"[Dynamic] skip {f} featdim mismatch {seq.shape[1]} != {feat_dim}")
                continue
            X.append(seq.astype(np.float32))
            y.append(idx)
    if len(X) == 0:
        print("[Dynamic] no valid sequence files found")
        return None

    # choose seq_len = median length or max
    lengths = [s.shape[0] for s in X]
    seq_len = int(np.median(lengths))
    seq_len = max(seq_len, min(30, max(lengths)))  # reasonable default
    # pad/truncate
    X_pad = np.zeros((len(X), seq_len, feat_dim), dtype=np.float32)
    for i, s in enumerate(X):
        L = s.shape[0]
        if L >= seq_len:
            X_pad[i] = s[:seq_len]
        else:
            X_pad[i, :L] = s

    # load model (assume class list in labels_file matches classes order)
    model = GestureLSTM(feat_dim, len(lines))
    st = torch.load(model_file, map_location="cpu")
    model.load_state_dict(st)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_pad))
        preds = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, target_names=classes, zero_division=0)
    cm = confusion_matrix(y, preds)

    png, csv = save_confusion(cm, classes, "dynamic")
    save_text(report, "dynamic_classification_report.txt")

    summary = {"accuracy": float(acc), "n_samples": int(len(y)), "classes": classes,
               "confusion_png": png, "confusion_csv": csv, "report_file": os.path.join(OUT_DIR,"dynamic_classification_report.txt")}
    print("[Dynamic] done. acc=%.2f%%" % (acc*100))
    return summary

# ----------------------------
# Run both
# ----------------------------
def main():
    out = {"alphabet": None, "dynamic": None}
    try:
        out["alphabet"] = eval_alphabet()
    except Exception as e:
        print("[ERROR] alphabet evaluation failed:", e)
    try:
        out["dynamic"] = eval_dynamic()
    except Exception as e:
        print("[ERROR] dynamic evaluation failed:", e)

    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("Saved evaluation summary to", os.path.join(OUT_DIR,"report.json"))

if __name__ == "__main__":
    main()

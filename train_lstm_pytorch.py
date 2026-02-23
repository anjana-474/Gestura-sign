# train_lstm_pytorch.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


DATA_FILE = "dynamic_dataset.npz"
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3


class NpzSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()   # (N, T, F)
        self.y = torch.from_numpy(y).long()    # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        last = out[:, -1, :]    # (B, hidden*dirs)
        logits = self.fc(last)  # (B, num_classes)
        return logits


def main():
    # ---- Load dataset ----
    data = np.load(DATA_FILE, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    gestures = list(data["gestures"])

    print("Gestures:", gestures)
    print("Train shape:", X_train.shape)
    print("Val shape  :", X_val.shape)

    _, SEQ_LEN, FEATURE_DIM = X_train.shape
    num_classes = len(gestures)

    train_ds = NpzSequenceDataset(X_train, y_train)
    val_ds   = NpzSequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = GestureLSTM(
        input_size=FEATURE_DIM,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---- Training loop ----
    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * Xb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)

                logits = model(Xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * Xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

    # ---- Save model and labels ----
    torch.save(model.state_dict(), "gesture_lstm_pytorch.pth")
    with open("gestures_dynamic.txt", "w") as f:
        for g in gestures:
            f.write(g + "\n")

    print("\nTraining done. Saved:")
    print("  gesture_lstm_pytorch.pth")
    print("  gestures_dynamic.txt")


if __name__ == "__main__":
    main()

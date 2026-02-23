import os, glob, json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

DATA_DIR = "data_alphabet"
MODEL_OUT = "alphabet_mlp.pt"
LETTERS_OUT = "alphabet_letters.json"
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
letters = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("Letters:", letters)

X_list, y_list = [], []
for idx, letter in enumerate(letters):
    files = glob.glob(os.path.join(DATA_DIR, letter, "*.npy"))
    for f in files:
        arr = np.load(f).astype(np.float32)
        X_list.append(arr)
        y_list.append(idx)

X = np.stack(X_list)
y = np.array(y_list, dtype=np.int64)

print("Dataset:", X.shape, "Labels:", y.shape)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Define MLP
class MLP(nn.Module):
    def __init__(self, input_dim, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, classes)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(63, len(letters)).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# Training
for epoch in range(1, EPOCHS+1):
    model.train()
    perm = np.random.permutation(len(X_train))
    losses = []
    for i in range(0, len(perm), BATCH_SIZE):
        idxs = perm[i:i+BATCH_SIZE]
        xb = torch.from_numpy(X_train[idxs]).to(DEVICE)
        yb = torch.from_numpy(y_train[idxs]).long().to(DEVICE)

        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_val).to(DEVICE)
        yb = torch.from_numpy(y_val).long().to(DEVICE)
        val_logits = model(xb)
        val_loss = loss_fn(val_logits, yb)
        val_acc = (val_logits.argmax(1) == yb).float().mean().item()

    print(f"Epoch {epoch} | train_loss={np.mean(losses):.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    # Save best model
    torch.save(model.state_dict(), MODEL_OUT)
    with open(LETTERS_OUT, "w") as f:
        json.dump(letters, f)

print("Training complete!")
print("Saved model ->", MODEL_OUT)
print("Saved mapping ->", LETTERS_OUT)

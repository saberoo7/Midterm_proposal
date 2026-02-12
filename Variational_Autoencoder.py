import os
import time
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# 0) DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# =========================================================
# 1) PATHS & HYPERPARAMS
# =========================================================
train_csv = "/home/ifar3/Documents/project/metadata/train_processed_fixed.csv"
val_csv   = "/home/ifar3/Documents/project/metadata/val_processed_fixed.csv"

BATCH_SIZE   = 32
NUM_WORKERS  = 4
LATENT_DIM   = 256
EPOCHS       = 200
PATIENCE     = 15
MIN_IMPROVEMENT = 0.0005  # 0.05%
BETA_KL      = 1e-3       # β-VAE: lower KL weight → sharper images

checkpoint_dir = "/home/ifar3/Documents/project/checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

# =========================================================
# 2) DATASETS & DATALOADERS
# =========================================================

# Train: add light augmentation (horizontal flip)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # images in [-1, 1]
])

# Val: no augmentation
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

class ChestXDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return {"image": img}

train_dataset = ChestXDataset(train_csv, transform=train_transform)
val_dataset   = ChestXDataset(val_csv,   transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print("Train size:", len(train_dataset))
print("Val size:",   len(val_dataset))

# quick sanity check
batch = next(iter(train_loader))
print("Batch OK. Shape:", batch["image"].shape)  # [B,1,256,256]

# =========================================================
# 3) VAE ARCHITECTURE  (SAME SHAPES AS BEFORE)
#    256 → 128 → 64 → 32 → 16  (4 conv downs)
#    Flatten: 256 * 16 * 16 = 65536
# =========================================================

class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,   32, 4, 2, 1), nn.ReLU(),  # 256 → 128
            nn.Conv2d(32,  64, 4, 2, 1), nn.ReLU(),  # 128 → 64
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),  # 64  → 32
            nn.Conv2d(128,256, 4, 2, 1), nn.ReLU(),  # 32  → 16
            nn.Flatten(),                             # 256×16×16 = 65536
        )
        self.fc_mu     = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),  # 16 → 32
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ReLU(),  # 32 → 64
            nn.ConvTranspose2d(64,   32, 4, 2, 1), nn.ReLU(),  # 64 → 128
            nn.ConvTranspose2d(32,    1, 4, 2, 1),
            nn.Tanh(),  # output in [-1, 1] to match normalized targets
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 16, 16)
        return self.net(h)


class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z   = mu + eps * std
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=BETA_KL):
    # Reconstruction loss
    mse = F.mse_loss(recon, x, reduction="mean")
    # KL divergence
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = mse + beta * kl
    return loss, mse, kl


vae = VAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-4)

print("\nTraining starting…")

# =========================================================
# 4) TRAINING LOOP WITH EARLY STOPPING
# =========================================================
best_loss = float("inf")
epochs_no_improve = 0

train_history = []
val_history   = []

start_total = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    vae.train()
    running_train = 0.0

    for batch in train_loader:
        x = batch["image"].to(device)

        optimizer.zero_grad()
        recon, mu, logvar = vae(x)
        loss, mse, kl = vae_loss(recon, x, mu, logvar, beta=BETA_KL)
        loss.backward()
        optimizer.step()

        running_train += loss.item()

    train_loss = running_train / len(train_loader)
    train_history.append(train_loss)

    # ---------- Validation ----------
    vae.eval()
    running_val = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            recon, mu, logvar = vae(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, beta=BETA_KL)
            running_val += loss.item()

    val_loss = running_val / len(val_loader)
    val_history.append(val_loss)

    # Improvement tracking
    improvement = (best_loss - val_loss) / best_loss if best_loss != float("inf") else 1.0

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.5f}")
    print(f"Val Loss:   {val_loss:.5f}")
    print(f"Improvement: {improvement*100:.3f}%")

    # Save per-epoch checkpoint
    torch.save(vae.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"))

    # Save best model
    if val_loss < best_loss - MIN_IMPROVEMENT:
        best_loss = val_loss
        epochs_no_improve = 0
        torch.save(vae.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
        print("🔥 New best model saved!")
    else:
        epochs_no_improve += 1
        print(f"No significant improvement ({epochs_no_improve}/{PATIENCE})")

    epoch_time = time.time() - epoch_start
    print(f"Epoch time: {epoch_time:.2f} sec ({epoch_time/60:.2f} min)")

    if epochs_no_improve >= PATIENCE:
        print("⚠️ Early stopping triggered.")
        break

total_time = time.time() - start_total
print("\n========== Training Complete ==========")
print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
print(f"Best validation loss: {best_loss:.5f}")

# =========================================================
# 5) QUICK RECONSTRUCTION VISUALISATION (from best model)
# =========================================================
# Reload the best model to be safe
best_path = os.path.join(checkpoint_dir, "best_model.pt")
vae.load_state_dict(torch.load(best_path, map_location=device))
vae.eval()
print(f"\nLoaded best model from: {best_path}")

batch = next(iter(val_loader))
x = batch["image"].to(device)

with torch.no_grad():
    recon, mu, logvar = vae(x)

x_np     = x[:8].cpu().numpy()
recon_np = recon[:8].cpu().numpy()

def show_reconstruction(idx):
    orig = x_np[idx, 0]
    rec  = recon_np[idx, 0]
    diff = np.abs(orig - rec)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(rec, cmap="gray")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    axes[2].imshow(diff, cmap="inferno")
    axes[2].set_title("Difference")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

for i in range(6):
    show_reconstruction(i)

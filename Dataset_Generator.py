import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from PIL import Image

# =========================
# 0) DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================
# 1) PATHS
# =========================
vae_path  = "/home/ifar3/Documents/project/checkpoints/best_model.pt"
ddpm_path = "/home/ifar3/Documents/project/checkpoints/ddpm_cond.pt"
lat_path  = "/home/ifar3/Documents/project/latents/latents_train.pt"

# =========================
# 2) CLASSES
# =========================
classes = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
    'Emphysema','Fibrosis','Hernia','Infiltration','Mass','No Finding',
    'Nodule','Pleural_Thickening','Pneumonia','Pneumothorax'
]
num_classes = len(classes)

# =========================
# 3) LOAD LATENT STATS (for matching VAE mu distribution)
# =========================
lat_train = torch.load(lat_path, map_location="cpu")  # (N,256)
lat_mean  = lat_train.mean(dim=0).to(device)
lat_std   = lat_train.std(dim=0).clamp_min(1e-6).to(device)
print("Latents:", lat_train.shape, "| mean/std ready")

# =========================
# 4) VAE DECODER (MUST MATCH YOUR TRAINED ARCH EXACTLY)
# =========================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.conv(x))

class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        self.up = nn.Sequential(
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            ResBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            ResBlock(32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh(),  # output in [-1,1]
        )
    def forward(self, z):
        h = self.fc(z).view(-1, 256, 16, 16)
        return self.up(h)

# Load decoder weights from VAE checkpoint (decoder.* keys)
decoder = Decoder().to(device)
vae_sd = torch.load(vae_path, map_location="cpu")
decoder_sd = {k.replace("decoder.", ""): v for k, v in vae_sd.items() if k.startswith("decoder.")}
decoder.load_state_dict(decoder_sd, strict=True)
decoder.eval()
print("✔ Decoder loaded (strict=True).")

# =========================
# 5) DDPM MODEL (MUST MATCH YOUR TRAINED CHECKPOINT)
#    Your checkpoint expects:
#    - time_embed = [SinusoidalTimeEmbedding, Linear, SiLU]  (Linear is index 1)
#    - NO label_embed parameters (labels concatenated directly)
#    - net.0.weight shape = [512, 335]  where 335 = 256 + 64 + 15
# =========================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int64
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * -(np.log(10000.0) / (half - 1))
        )
        t = t.float().unsqueeze(1)  # (B,1)
        args = t * freqs.unsqueeze(0)  # (B,half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B,2*half)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb

class CondDenoiser(nn.Module):
    def __init__(self, latent_dim=256, num_classes=15, time_dim=64, hidden_dim=512):
        super().__init__()
        # IMPORTANT: Linear must be index 1 to match checkpoint keys time_embed.1.weight/bias
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),     # index 0 (no params)
            nn.Linear(time_dim, time_dim),         # index 1 (params exist in checkpoint)
            nn.SiLU(),                              # index 2
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim + num_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, x, t, y):
        # x: (B,256), t:(B,), y:(B,15)
        t_emb = self.time_embed(t)                 # (B,64)
        h = torch.cat([x, t_emb, y], dim=1)        # (B,335)
        return self.net(h)                         # (B,256)

ddpm = CondDenoiser(latent_dim=256, num_classes=num_classes, time_dim=64, hidden_dim=512).to(device)
ddpm_sd = torch.load(ddpm_path, map_location="cpu")
# STRICT LOAD — if this fails, architecture doesn't match and we should NOT sample.
ddpm.load_state_dict(ddpm_sd, strict=True)
ddpm.eval()
print("✔ DDPM loaded (strict=True).")

# =========================
# 6) DIFFUSION SCHEDULE (MATCH TRAINING)
# =========================
T = 500
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
alpha_hat = torch.cumprod(alphas, dim=0)

# =========================
# 7) SAMPLING (STANDARD DDPM REVERSE STEP)
# =========================
@torch.no_grad()
def sample_latents(y_onehot: torch.Tensor, steps=T) -> torch.Tensor:
    b = y_onehot.size(0)
    x = torch.randn(b, 256, device=device)

    for t in reversed(range(steps)):
        t_tensor = torch.full((b,), t, device=device, dtype=torch.long)

        eps = ddpm(x, t_tensor, y_onehot)

        a_t = alphas[t]
        ah_t = alpha_hat[t]

        # DDPM reverse mean:
        x = (1.0 / torch.sqrt(a_t)) * (x - ((1.0 - a_t) / torch.sqrt(1.0 - ah_t)) * eps)

        if t > 0:
            x = x + torch.sqrt(betas[t]) * torch.randn_like(x)

    # match encoder (mu) distribution
    x = x * lat_std + lat_mean
    return x

@torch.no_grad()
def decode_latents(z: torch.Tensor) -> torch.Tensor:
    imgs = decoder(z)            # [-1,1]
    imgs = (imgs + 1) / 2        # [0,1]
    imgs = imgs.clamp(0, 1)
    return imgs.cpu()            # (B,1,256,256)

def show_grid(imgs: torch.Tensor, title: str):
    # imgs: (B,1,H,W)
    B = imgs.shape[0]
    cols = min(5, B)
    rows = int(np.ceil(B / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.3, rows*2.3))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(title, fontsize=14)

    for i in range(rows*cols):
        ax = axes[i]
        ax.axis("off")
        if i < B:
            ax.imshow(imgs[i,0].numpy(), cmap="gray", vmin=0, vmax=1)
    plt.tight_layout()
    plt.show()

# =========================
# 8) GENERATE + VISUALIZE
# =========================
synthetic_images = {}
N_PER_CLASS = 10

for idx, cls in enumerate(classes):
    print(f"\n🔥 Generating: {cls}")
    y = torch.zeros(N_PER_CLASS, num_classes, device=device)
    y[:, idx] = 1.0

    z = sample_latents(y)
    imgs = decode_latents(z)

    # quick sanity: ensure not all-white/all-black
    print("  z mean/std:", float(z.mean().cpu()), float(z.std().cpu()))
    print("  img min/max:", float(imgs.min()), float(imgs.max()))

    synthetic_images[cls] = imgs
    show_grid(imgs, cls)

print("\n✅ Done.")
# ===============================
# SAVE SYNTHETIC DATASET (FIXED)
# ===============================

save_root = "/home/ifar3/Documents/project/synthetic_dataset"
img_root  = os.path.join(save_root, "images")
os.makedirs(img_root, exist_ok=True)

records = []

for cls_name, imgs in synthetic_images.items():
    cls_dir = os.path.join(img_root, cls_name)
    os.makedirs(cls_dir, exist_ok=True)

    for i in range(len(imgs)):
        # imgs[i] is a torch Tensor: (1, H, W)
        img = imgs[i][0]                 # (H, W)
        img = img.cpu().numpy()          # → NumPy
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

        fname = f"{cls_name}_{i:04d}.png"
        fpath = os.path.join(cls_dir, fname)

        Image.fromarray(img).save(fpath)

        records.append({
            "image_path": fpath,
            "label": cls_name
        })

# Save metadata
df = pd.DataFrame(records)
csv_path = os.path.join(save_root, "synthetic_metadata.csv")
df.to_csv(csv_path, index=False)

print("✅ Synthetic dataset saved successfully")
print("📁 Root folder:", save_root)
print("🖼️  Total images:", len(df))
print("📄 Metadata CSV:", csv_path)

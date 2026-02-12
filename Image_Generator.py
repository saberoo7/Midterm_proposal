import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import entropy

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================
# CLASSES
# =========================
classes = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
    'Emphysema','Fibrosis','Hernia','Infiltration','Mass','No Finding',
    'Nodule','Pleural_Thickening','Pneumonia','Pneumothorax'
]
num_classes = len(classes)

# =========================
# LOAD LATENT STATS
# =========================
lat_train = torch.load(
    "/home/ifar3/Documents/project/latents/latents_train.pt",
    map_location="cpu"
)
lat_mean = lat_train.mean(dim=0).to(device)
lat_std  = lat_train.std(dim=0).clamp_min(1e-6).to(device)

CLIP_K = 2.5
z_min = lat_mean - CLIP_K * lat_std
z_max = lat_mean + CLIP_K * lat_std

# =========================
# MODELS (MATCH TRAINING)
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
            nn.Tanh(),
        )
    def forward(self, z):
        return self.up(self.fc(z).view(-1, 256, 16, 16))

class CondDenoiser(nn.Module):
    def __init__(self, latent_dim=256, num_classes=15, time_dim=64, hidden_dim=512):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU()
        )
        self.label_embed = nn.Linear(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim + num_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, x, t, y):
        t = t.float().unsqueeze(1) / 500.0
        return self.net(torch.cat([
            x,
            self.time_embed(t),
            self.label_embed(y)
        ], dim=1))

# =========================
# LOAD CHECKPOINTS
# =========================
vae_path  = "/home/ifar3/Documents/project/checkpoints/best_model.pt"
ddpm_path = "/home/ifar3/Documents/project/checkpoints/ddpm_cond.pt"

decoder = Decoder().to(device)
decoder.load_state_dict({
    k.replace("decoder.", ""): v
    for k, v in torch.load(vae_path, map_location=device).items()
    if k.startswith("decoder.")
}, strict=True)
decoder.eval()

ddpm = CondDenoiser(num_classes=num_classes).to(device)
ddpm.load_state_dict(torch.load(ddpm_path, map_location=device), strict=False)
ddpm.eval()

print("✔ Models loaded")

# =========================
# DIFFUSION SCHEDULE
# =========================
T = 500
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1 - betas
alpha_hat = torch.cumprod(alphas, dim=0)

# =========================
# SAMPLER
# =========================
@torch.no_grad()
def sample_latent(y):
    x = torch.randn(y.size(0), 256, device=device)
    for t in reversed(range(T)):
        tvec = torch.full((y.size(0),), t, device=device, dtype=torch.long)
        eps = ddpm(x, tvec, y)
        a, ah = alphas[t], alpha_hat[t]
        x = (x - ((1-a)/torch.sqrt(1-ah)) * eps) / torch.sqrt(a)
    x = x * lat_std + lat_mean
    return torch.max(torch.min(x, z_max), z_min)

@torch.no_grad()
def decode(z):
    img = (decoder(z) + 1) / 2
    return img.clamp(0,1)

# =========================
# WINDOWING
# =========================
def window(img, p1=1, p99=99):
    lo, hi = np.percentile(img, p1), np.percentile(img, p99)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

# =========================
# PROMPT
# =========================
disease = input(f"\nEnter disease name {classes}: ").strip()
idx = classes.index(disease)

y = torch.zeros(1, num_classes, device=device)
y[0, idx] = 1

img = decode(sample_latent(y))[0,0].cpu().numpy()
img_disp = window(img)

# =========================
# QUALITATIVE METRICS
# =========================
edges = ndimage.sobel(img_disp)
lap_var = ndimage.laplace(img_disp).var()
hist, _ = np.histogram(img_disp, bins=64, range=(0,1), density=True)

metrics = {
    "Min / Max": (float(img.min()), float(img.max())),
    "Mean / Std": (float(img.mean()), float(img.std())),
    "Entropy": float(entropy(hist + 1e-6)),
    "Edge density": float(np.mean(np.abs(edges))),
    "Laplacian variance (sharpness)": float(lap_var),
    "Intensity sparsity": float(np.mean(img_disp < 0.05) + np.mean(img_disp > 0.95))
}

print("\n📊 Qualitative Metrics")
for k, v in metrics.items():
    print(f"{k}: {v}")

# =========================
# DISPLAY
# =========================
plt.figure(figsize=(4,4))
plt.imshow(img_disp, cmap="gray")
plt.title(f"Synthetic {disease}")
plt.axis("off")
plt.show()

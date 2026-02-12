import time

T = 500
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1 - betas
alpha_hat = torch.cumprod(alphas, dim=0)

model = CondDenoiser().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 30
print_every = 100

best_loss = float("inf")
save_path = "/home/ifar3/Documents/project/checkpoints/ddpm_cond.pt"

total_start = time.time()   # <-- START total timer

print("\n🔥 Starting conditional DDPM training...\n")

for epoch in range(1, EPOCHS+1):

    epoch_start = time.time()   # <-- epoch timer start
    model.train()
    train_loss = 0

    for step, (x0, y) in enumerate(train_loader):

        x0 = x0.to(device)
        y  = y.to(device)

        b = x0.size(0)
        t = torch.randint(0, T, (b,), device=device)

        noise = torch.randn_like(x0)
        a_hat = alpha_hat[t].view(-1,1)

        x_t = torch.sqrt(a_hat) * x0 + torch.sqrt(1 - a_hat) * noise

        opt.zero_grad()
        noise_pred = model(x_t, t, y)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        opt.step()

        train_loss += loss.item()

        if (step+1) % print_every == 0:
            print(f"Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss {loss.item():.6f}")

    avg_loss = train_loss / len(train_loader)
    epoch_time = time.time() - epoch_start

    print(f"\nEpoch {epoch} finished.")
    print(f"Avg loss: {avg_loss:.6f}")
    print(f"⏳ Epoch time: {epoch_time:.2f} sec ({epoch_time/60:.2f} min)\n")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print(f"🔥 Best model updated! Saved to {save_path}\n")

# ----------------------------
# END: Total training time
# ----------------------------
total_time = time.time() - total_start
print("\n========================================")
print(f"⏳ TOTAL TRAINING TIME: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
print("========================================\n")
print("DDPM training complete!")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# torch.set_default_dtype(torch.float64)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logvar_clip = torch.tensor(85)


# -----------------------------
# Dataset
# -----------------------------

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    "./data", train=True, download=True, transform=transform
)

val_dataset = datasets.MNIST(
    "./data", train=False, transform=transform
)


# -----------------------------
# Dataset corruption
# -----------------------------

def corrupt_dataset(dataset):

    subset_size = np.random.randint(100, 1000)

    idx = np.random.choice(len(dataset), subset_size, replace=False)

    return Subset(dataset, idx)


# -----------------------------
# VAE model
# -----------------------------

class VAE(nn.Module):

    def __init__(self, in_features, hidden_dim_list, latent_dim, logvar_clip, torch_gen):

        super().__init__()

        self.logvar_clip = logvar_clip

        encoder_dim_list = [in_features] + hidden_dim_list

        encoder_layers = self.get_layers_from_dim_list(encoder_dim_list)

        self.encoder = nn.Sequential(*encoder_layers)

        self.mu = nn.Linear(hidden_dim_list[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dim_list[-1], latent_dim)

        decoder_dim_list = [latent_dim] + hidden_dim_list[::-1]
        decoder_layers = self.get_layers_from_dim_list(decoder_dim_list)

        self.decoder_logits = nn.Sequential(*(decoder_layers + [nn.Linear(hidden_dim_list[0], in_features)]))
        self.sigmoid = nn.Sigmoid()

        self.torch_gen = torch_gen

    def get_layers_from_dim_list(self, dim_list):
        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            layers.append(nn.ReLU())

        return layers

    def encode(self, x):

        h = self.encoder(x)

        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp_max(logvar, logvar_clip)

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std, device=device, generator=self.torch_gen)

        return mu + eps * std

    def decode(self, z):
        recon_logits = self.decoder_logits(z)
        recon = self.sigmoid(recon_logits)
        return recon, recon_logits

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        recon, recon_logits = self.decode(z)

        return recon, recon_logits, mu, logvar


# -----------------------------
# Loss function
# -----------------------------

def vae_loss(recon_logits, x, mu, logvar, beta, logvar_clip):
    # Average or sum?

    # Clip or not?
    # eps = 1e-12
    # recon = torch.clamp(recon, eps, 1 - eps)

    recon_loss = nn.functional.binary_cross_entropy_with_logits(
        recon_logits, x, reduction="sum"
    )

    # Clip logvar
    logvar = torch.clamp_max(logvar, logvar_clip)

    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + beta * kl


# -----------------------------
# Training
# -----------------------------

def train_vae(
    hidden_dim_list,
    latent_dim,
    lr,
    beta,
    epochs,
    batch_size,
    p_outlier,
    torch_gen
):

    dataset = train_dataset

    if np.random.rand() < p_outlier:
        dataset = corrupt_dataset(dataset)
        print("Corrupted dataset used:", len(dataset))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch_gen)

    model = VAE(784, hidden_dim_list, latent_dim, logvar_clip, torch_gen=torch_gen).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for x,_ in loader:

            x = x.view(-1,784).to(device)

            _, recon_logits, mu, logvar = model(x)

            loss = vae_loss(recon_logits, x, mu, logvar, beta, logvar_clip)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}, loss:",
            total_loss / len(loader.dataset)
        )

    return model


# -----------------------------
# Evaluation
# -----------------------------

def evaluate(model):

    loader = DataLoader(val_dataset, batch_size=256)

    model.eval()

    total = 0

    with torch.no_grad():

        for x,_ in loader:

            x = x.view(-1,784).to(device)

            _, recon_logits, mu, logvar = model(x)

            loss = vae_loss(recon_logits, x, mu, logvar, beta=1.0, logvar_clip=logvar_clip)

            total += loss.item()

    return total / len(loader.dataset)

def plot_reconstructions(model, dataset, device, n=8):

    loader = DataLoader(dataset, batch_size=n, shuffle=True)
    x, _ = next(iter(loader))

    x = x.view(-1, 784).to(device)

    model.eval()
    with torch.no_grad():
        recon, _, _, _ = model(x)

    x = x.view(-1, 28, 28).cpu()
    recon = recon.view(-1, 28, 28).cpu()

    fig, axes = plt.subplots(2, n, figsize=(n*2, 4))

    for i in range(n):
        axes[0, i].imshow(x[i], cmap="gray")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon[i], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruction")

    plt.tight_layout()
    plt.show()

def plot_generated(model, device, n=8):

    latent_dim = model.mu.out_features

    z = torch.randn(n, latent_dim).to(device)

    model.eval()
    with torch.no_grad():
        samples, _ = model.decode(z)

    samples = samples.view(-1, 28, 28).cpu()

    fig, axes = plt.subplots(1, n, figsize=(n*2, 2))

    for i in range(n):
        axes[i].imshow(samples[i], cmap="gray")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def _run_vae(
        hidden_dim_list,
        latent_dim,
        lr,
        beta,
        epochs,
        batch_size,
        p_outlier,
        torch_gen
):
    model = train_vae(
        hidden_dim_list=hidden_dim_list,
        latent_dim=latent_dim,
        lr=lr,
        beta=beta,
        epochs=epochs,
        batch_size=batch_size,
        p_outlier=p_outlier,
        torch_gen=torch_gen
    )

    validation_loss = evaluate(model)

    return validation_loss

def run_vae(latent_dim, lr, beta, first_hidden_dim, L, epochs, batch_size, p_outlier, torch_gen):
    assert 0.5 <= L <= 3.5, L
    if L <= 1.5:
        _L = 1
    elif L <= 2.5:
        _L = 2
    else:
        _L = 3

    assert latent_dim <= first_hidden_dim <= 784, (latent_dim, first_hidden_dim)

    _batch_size = int(batch_size)
    _latent_dim = int(latent_dim)
    _first_hidden_dim = int(first_hidden_dim)
    _hidden_dim_list = np.logspace(np.log10(latent_dim), np.log10(first_hidden_dim), _L + 1)[1:]
    hidden_dim_list = [int(_tmp) for _tmp in _hidden_dim_list][::-1]

    return _run_vae(
        hidden_dim_list,
        _latent_dim,
        lr,
        beta,
        epochs,
        _batch_size,
        p_outlier,
        torch_gen
    )


if __name__ == "__main__":

    # -----------------------------
    # Run experiment
    # -----------------------------

    torch_gen = torch.Generator(device=device)

    model = train_vae(
        hidden_dim_list=[512],
        latent_dim=20,
        lr=0.05,
        beta=5.0,
        epochs=5,
        batch_size=128,
        p_outlier=0.0,
        torch_gen=torch_gen
    )

    val_loss = evaluate(model)

    print("Validation ELBO:", val_loss)

    plot_reconstructions(model, val_dataset, device, n=8)
    plot_generated(model, device, n=8)
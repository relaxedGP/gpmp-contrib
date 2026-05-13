import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# -----------------------------
# Simple GAN proxy
# -----------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, channels=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

# -----------------------------
# GAN objective compatible GP
# -----------------------------
def _gan_objective(x, rng, epochs=3):
    """
    x: (n_points, 7) -> latent_dim, lr_gen, lr_disc, beta1, dropout_gen, dropout_disc, batch_size
    """
    assert x.ndim == 2, x.shape
    res = np.zeros([x.shape[0]])

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    for i in range(x.shape[0]):
        torch_seed = int(rng.integers(0, 2**63 - 1))
        torch_gen = torch.Generator()
        torch_gen.manual_seed(torch_seed)

        latent_dim = int(np.exp(x[i, 0]))
        lr_gen = np.exp(x[i, 1])
        lr_disc = np.exp(x[i, 2])
        beta1 = np.exp(x[i, 3])
        dropout_disc = x[i, 4]
        batch_size = int(np.exp(x[i, 5]))

        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch_gen)

        G = Generator(latent_dim).to(device)
        D = Discriminator(dropout=dropout_disc).to(device)

        criterion = nn.BCELoss()
        optimizerG = optim.Adam(G.parameters(), lr=lr_gen, betas=(beta1, 0.999))
        optimizerD = optim.Adam(D.parameters(), lr=lr_disc, betas=(beta1, 0.999))

        for epoch in range(epochs):
            for imgs, _ in loader:
                imgs = imgs.to(device)
                current_batch_size = imgs.size(0)
                real_labels = torch.ones(current_batch_size, 1, device=device)
                fake_labels = torch.zeros(current_batch_size, 1, device=device)

                # Discriminator
                D.zero_grad()
                outputs_real = D(imgs)
                z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device, generator=torch_gen)
                fake_imgs = G(z)
                outputs_fake = D(fake_imgs.detach())
                lossD = criterion(outputs_real, real_labels) + criterion(outputs_fake, fake_labels)
                lossD.backward()
                optimizerD.step()

                # Generator
                G.zero_grad()
                outputs_fake = D(fake_imgs)
                lossG = criterion(outputs_fake, real_labels)
                lossG.backward()
                optimizerG.step()

        # Proxy metric : instabilité D(real)-D(fake)
        with torch.no_grad():
            imgs, _ = next(iter(loader))
            imgs = imgs.to(device)
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device, generator=torch_gen)
            fake_imgs = G(z)
            real_mean = D(imgs).mean().item()
            fake_mean = D(fake_imgs).mean().item()
        res[i] = abs(real_mean - fake_mean)

    return res

# -----------------------------
# Domaine compatible GP
# -----------------------------
_gan_dict = {
    "input_dim": 7,
    "input_box": [
        [np.log(16), np.log(1e-5), np.log(1e-5), np.log(0.1), 0.0, 0.0, np.log(16)],
        [np.log(128), np.log(5e-3), np.log(5e-3), np.log(0.9), 0.5, 0.5, np.log(128)]
    ],
}

# -----------------------------
# ComputerExperiment wrapper
# -----------------------------
from gpmpcontrib.computerexperiment import ComputerExperiment

def gan_experiment(rng, epochs=3):
    gan = ComputerExperiment(
        _gan_dict["input_dim"],
        _gan_dict["input_box"],
        single_objective=lambda x: _gan_objective(x, rng, epochs=epochs)
    )
    gan.noiseless_problem = None
    return gan

# -----------------------------
# Quick main for testing
# -----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Test point dans le domaine
    x_test = np.array([[np.log(64), np.log(1e-3), np.log(1e-3), np.log(0.5), 0.2, 0.2, np.log(64)]])

    res = _gan_objective(x_test, rng, epochs=3)
    print("Proxy metric:", res)

    # Quick generation visualization
    latent_dim = int(np.exp(x_test[0,0]))
    torch_gen = torch.Generator()
    torch_gen.manual_seed(42)
    G = Generator(latent_dim).to(device)
    z = torch.randn(8, latent_dim, 1, 1, device=device, generator=torch_gen)
    with torch.no_grad():
        fake_imgs = G(z).cpu()

    # Denormalize images
    imgs = (fake_imgs + 1) / 2

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 8, figsize=(16,2))
    for i in range(8):
        axes[i].imshow(np.transpose(imgs[i].numpy(), (1,2,0)))
        axes[i].axis("off")
    plt.show()
import torch
from model import Generator
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def evaluate(pretrained_path: str):
    device = torch.device(f"cuda:0")
    model_params = {
        "latent_dim": 100,
        "image_num_channels": 3,
        "G_feature_map_dim": 64,
        "D_feature_map_dim": 64,
        "device": device,
    }
    ##### END CONFIG ######

    netG = Generator(**model_params).to(device)
    checkpoint = torch.load(pretrained_path, device)
    netG.load_state_dict(checkpoint["G_state_dict"])

    noise = torch.randn(64, model_params["latent_dim"], 1, 1, device=device)
    fake = netG(noise).detach().cpu()
    plt.imsave("test.pdf", vutils.make_grid(fake, padding=2, normalize=True).permute(1, 2, 0))

if __name__ == "__main__":
    evaluate("ckpts/checkpoint-epoch199.pth")

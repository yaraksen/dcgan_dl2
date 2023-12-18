import torch
from model import Generator
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from dataset import KittyDataset
from torch.utils.data import DataLoader
from numpy.random import choice
from torchvision.datasets import ImageFolder
from torchvision import transforms
from piq import FID, ssim


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
    fake = netG(noise).detach()
    
    print(ssim(fake, ))
    
    
    
    
    
    
    
    
    # plt.imsave(
    #     "generated_samples.pdf",
    #     vutils.make_grid(fake, padding=2, normalize=True).permute(1, 2, 0),
    # )


def show_examples():
    data_path = "cats/"
    train_dataset = KittyDataset(data_path)
    samples = torch.cat(
        [train_dataset[i] for i in choice(len(train_dataset), 64)], dim=0
    )
    plt.imsave(
        "train_samples.pdf",
        vutils.make_grid(samples, padding=2, normalize=True).permute(1, 2, 0),
    )


def data_fp():
    data_path = "data/cats/"
    train_batch_size = 128
    
    # train_dataset = ImageFolder(
    #     root=data_path,
    #     transform=transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]
    #     ),
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=4
    # )
    
    train_dataset = KittyDataset(data_path)
    train_loader = DataLoader(
        train_dataset, train_batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    
    for batch in train_loader:
        print(batch.shape)

    plt.imsave(
        "train1_samples.pdf",
        vutils.make_grid(batch[:64], padding=2, normalize=True).permute(1, 2, 0),
    )
        

if __name__ == "__main__":
    # evaluate("ckpts/checkpoint-epoch100.pth")
    # show_examples()
    data_fp()

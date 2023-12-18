import torch
from model import Generator
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from dataset import TestKittyDataset, KittyDataset
from torch.utils.data import DataLoader
from numpy.random import choice
from torchvision.datasets import ImageFolder
from torchvision import transforms
from piq import FID, ssim
import gc
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


def to_pos_range(batch):
        return (batch + 1.) / 2.

def evaluate():
    torch.manual_seed(42)
    device = "cpu"
    print(device)
    data_path = "data/cats/"
    model_params = {
        "latent_dim": 100,
        "image_num_channels": 3,
        "G_feature_map_dim": 64,
        "D_feature_map_dim": 64,
        "device": device,
    }
    ##### END CONFIG ######

    netG = Generator(**model_params).to(device)
    
    test_dataset = KittyDataset(data_path)
    test_loader = DataLoader(
        test_dataset, 1024, shuffle=False, num_workers=4
    )
    
    ssim_history = []
    fid_history = []
    
    os.makedirs("generated", exist_ok=True)

    with torch.no_grad():
        noise = torch.randn(1024, model_params["latent_dim"], 1, 1, device=device)
        ckpts = list(glob("ckpts/*.pth"))
        for ckpt_num, ckpt in tqdm(enumerate(ckpts), total=len(ckpts), desc="Evaluation"):
            checkpoint = torch.load(ckpt, device)
            netG.load_state_dict(checkpoint["G_state_dict"])
            
            fake = to_pos_range(netG(noise).detach())
            real = to_pos_range(next(iter(test_loader)).to(device))
            ssim_score = ssim(fake, real).item()
            
            ssim_history.append(ssim_score)
            plt.imsave(
                f"generated/ckpt{ckpt_num}.png",
                vutils.make_grid(fake[:64].cpu(), padding=2, normalize=True).permute(1, 2, 0).numpy(),
            )
            
            torch.cuda.empty_cache()
            gc.collect()
            
            fid_metric = FID()
            fake_dl = DataLoader(TestKittyDataset(fake), 8, shuffle=False, num_workers=4)
            real_dl = DataLoader(TestKittyDataset(real), 8, shuffle=False, num_workers=4)
            fake_feats = fid_metric.compute_feats(fake_dl)
            real_feats = fid_metric.compute_feats(real_dl)
            fid_score = fid_metric.compute_metric(fake_feats, real_feats)
            fid_history.append(fid_score)
    
    fig, axes = plt.subplots(ncols=2)
    axes[0].plot(ssim_history)
    axes[0].set_title("SSIM")
    axes[0].set_xlabel("Step")
    
    axes[1].plot(fid_history)
    axes[1].set_title("FID")
    axes[1].set_xlabel("Step")
        
    fig.savefig("scores_history.png")


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
    
    
def evaluate_one(ckpt_path: str):
    torch.manual_seed(42)
    device = "cuda"
    print(device)
    model_params = {
        "latent_dim": 100,
        "image_num_channels": 3,
        "G_feature_map_dim": 64,
        "D_feature_map_dim": 64,
        "device": device,
    }
    ##### END CONFIG ######

    netG = Generator(**model_params).to(device)
    checkpoint = torch.load(ckpt_path, device)
    netG.load_state_dict(checkpoint["G_state_dict"])

    with torch.no_grad():
        noise = torch.randn(64, model_params["latent_dim"], 1, 1, device=device)
        fake = to_pos_range(netG(noise).detach().cpu())
        
        plt.imsave(
            f"generated.png",
            vutils.make_grid(fake, padding=2, normalize=True).permute(1, 2, 0).numpy(),
        )
        

if __name__ == "__main__":
    evaluate_one("checkpoint-epoch1900.pth")

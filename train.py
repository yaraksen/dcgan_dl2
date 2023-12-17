import argparse
import torch
from dataset import KittyDataset
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from torch.nn import BCELoss
from trainer import Trainer
import wandb
from math import ceil
import os


SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def main(args):
    device = torch.device("cpu") if args.use_cpu else torch.device(f"cuda:0")
    # use_bf16 = False if args.use_cpu else True
    if os.path.exists("/kaggle"):
        data_path = "/kaggle/input/cats-faces-64x64-for-generative-models/cats/"
    else:
        data_path = "cats/"
    train_batch_size = 128
    num_epochs = 8000
    lr = 2e-4
    beta1 = 0.5
    weight_decay = 0.0
    model_params = {
        "latent_dim": 100,
        "image_num_channels": 3,
        "G_feature_map_dim": 64,
        "D_feature_map_dim": 64,
        "device": device,
    }
    wandb_project = "murky_gan"
    ##### END CONFIG ######

    wandb.login(relogin=True, key=args.wandb_key)
    wandb.init(entity="yaraksen", project=wandb_project, config=model_params)

    train_dataset = KittyDataset(data_path)
    train_loader = DataLoader(
        train_dataset, train_batch_size, shuffle=True, drop_last=True, num_workers=4
    )

    print("Train dataset size:", len(train_dataset))
    print("Train loader length:", len(train_loader))

    netG = Generator(**model_params).to(device)
    netD = Discriminator(**model_params).to(device)
    print(netG)
    print(netD)

    criterion = BCELoss()

    optimizerD = torch.optim.AdamW(
        netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay
    )
    optimizerG = torch.optim.AdamW(
        netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay
    )

    G_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizerG,
        anneal_strategy="cos",
        pct_start=0.05,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
    )

    D_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizerD,
        anneal_strategy="cos",
        pct_start=0.05,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
    )

    trainer = Trainer(
        netG,
        netD,
        optimizerG,
        optimizerD,
        G_scheduler,
        D_scheduler,
        criterion,
        train_loader,
        num_epochs,
        model_params["latent_dim"],
        device,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MurkyLM")
    parser.add_argument(
        "-wk",
        "--wandb_key",
        default=None,
        type=str,
        help="Wandb API key",
    )
    parser.add_argument("-cpu", "--use_cpu", action="store_true")
    main(parser.parse_args())

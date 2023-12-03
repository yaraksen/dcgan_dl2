import argparse
import torch
from dataset import TinyStories
from torch.utils.data import DataLoader
from model import MurkyLM
from torch.nn import CrossEntropyLoss
from trainer import Trainer
import wandb
# from utils import find_device
from math import ceil


SEED = 42
torch.manual_seed(SEED)

def main(args):
    device = torch.device("cpu") if args.use_cpu else torch.device(f"cuda:0")
    data_path = "tiny_stories_tokenized.npy"
    sp_model_prefix = "MurkyLM"
    use_bf16 = True
    train_batch_size = 256
    num_epochs = 20
    grad_accum_steps = 2
    vocab_size = 5000
    max_len = 256
    model_params = {
        "d_model": 512,
        "nhead": 8,
        "d_hid": 2048,
        "nlayers": 8,
        "dropout": 0.1,
        "max_len": max_len,
        "device": device
    }
    wandb_project = "murkylm"
    ##### END CONFIG ######
    
    wandb.login(relogin=True, key=args.wandb_key)
    wandb.init(entity="yaraksen",
               project=wandb_project,
               config=model_params)

    train_dataset = TinyStories(data_path, train=True, limit=4000)
    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    test_dataset = TinyStories(data_path, train=False, limit=4000)
    test_loader = DataLoader(test_dataset, train_batch_size, shuffle=False, num_workers=4)
    
    print('Train dataset size:', len(train_dataset))
    print('Train loader length:', len(train_loader))
    print('Test dataset size:', len(test_dataset))
    print('Test loader length:', len(test_loader))

    model = MurkyLM(vocab_size, **model_params)
    print(model)
    model = model.to(device)

    criterion = CrossEntropyLoss()
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-7)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, anneal_strategy="cos", pct_start=0.2, max_lr=1e-3,
                                                       steps_per_epoch=ceil(len(train_loader) / grad_accum_steps), epochs=num_epochs)

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        train_loader,
        test_loader,
        num_epochs,
        grad_accum_steps,
        use_bf16,
        device
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
    parser.add_argument(
        "-cpu",
        "--use_cpu",
        action='store_true'
    )
    main(parser.parse_args())

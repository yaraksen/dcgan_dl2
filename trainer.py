from genericpath import exists
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch import nn
from torch import Tensor
from typing import Tuple
from model import MurkyLM
from numpy import exp as np_exp
import wandb
from torch.cuda.amp import GradScaler
from torch import autocast
import os

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# some functions are from my DLA homework template


class Trainer():
    def __init__(self, model: MurkyLM, criterion, optimizer, lr_scheduler, train_loader, test_loader, num_epochs: int, grad_accum_steps, use_bf16: bool, device):
        super().__init__()
        self.train_dataloader = train_loader
        self.test_dataloader = test_loader
        self.len_epoch = len(self.train_dataloader)
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model = model
        self.vocab_size = model.vocab_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.grad_accum_steps = grad_accum_steps
        self.step = 0
        self.use_bf16 = use_bf16
        
        self.scaler = GradScaler()
    
    def train(self) -> None:
        save_interval = 1
        for epoch in range(self.num_epochs):
            self._train_epoch(epoch)
            self._test_epoch()
            if (epoch % save_interval == 0) or (epoch + 1 == self.num_epochs):
                self.save_checkpoint(epoch)

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        total_loss = 0.0
        log_interval = 500
        self.optimizer.zero_grad()

        num_batches = len(self.train_dataloader)
        for batch_num, input_ids in enumerate(tqdm(self.train_dataloader, desc="Train")):
            input_ids = input_ids.to(self.device)
            
            with autocast(enabled=self.use_bf16, device_type='cuda', dtype=torch.float16):
                output = self.model(input_ids[..., :-1])
                loss = self.criterion(output.reshape(-1, output.shape[-1]), input_ids[..., 1:].reshape(-1))
                loss = loss / self.grad_accum_steps
            
            total_loss += loss.item()
            self.scaler.scale(loss).backward()
            
            if ((batch_num + 1) % self.grad_accum_steps == 0) or ((batch_num + 1) == num_batches):
                self.step += 1
                self.scaler.step(self.optimizer)
                self.scaler.update()
                wandb.log({"grad norm": self.get_grad_norm()}, step=self.step)
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
            
            if self.step % log_interval == 0 and self.step > 0:
                lr = self.optimizer.param_groups[0]['lr']
                cur_loss = total_loss / (log_interval * self.grad_accum_steps)
                ppl = np_exp(cur_loss)
                wandb.log({
                    "epoch": epoch,
                    "lr": lr,
                    "loss": cur_loss,
                    "ppl": ppl
                    }, step=self.step)
                # print(f'| epoch {epoch} | loss {cur_loss:.4f} | ppl {ppl:.4f}')
                total_loss = 0

    def _test_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for input_ids in tqdm(self.test_dataloader, desc="Test"):
                input_ids = input_ids.to(self.device)
                output = self.model(input_ids[..., :-1])
                loss = self.criterion(output.reshape(-1, output.shape[-1]), input_ids[..., 1:].reshape(-1))
                total_loss += loss.item()
        test_loss = total_loss / len(self.test_dataloader)
        test_ppl = np_exp(test_loss)
        wandb.log({
            "test_loss": test_loss,
            "test_ppl": test_ppl
            }, step=self.step)

    # def _clip_grad_norm(self):
    #     if self.config["trainer"].get("grad_norm_clip", None) is not None:
    #         clip_grad_norm_(
    #             self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
    #         )

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def save_checkpoint(self, epoch):
        os.makedirs("ckpts", exist_ok=True)
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict()
        }
        filename = f"ckpts/checkpoint-epoch{epoch}.pth"
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        self.log_ckpt(filename, "checkpoint")

    def log_ckpt(self, ckpt_path: str, cktp_name: str):
        artifact = wandb.Artifact(name=cktp_name, type="model")
        artifact.add_file(local_path=ckpt_path)
        wandb.log_artifact(artifact)

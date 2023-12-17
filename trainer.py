import torch
from tqdm import tqdm
import wandb
import os
import torchvision.utils as vutils

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# some functions are from my DLA homework template


class Trainer:
    def __init__(
        self,
        netG,
        netD,
        optimizerG,
        optimizerD,
        G_scheduler,
        D_scheduler,
        criterion,
        train_loader,
        num_epochs,
        latent_dim,
        device,
    ):
        super().__init__()
        self.train_dataloader = train_loader
        self.len_epoch = len(self.train_dataloader)
        self.G_scheduler = G_scheduler
        self.D_scheduler = D_scheduler
        self.device = device
        self.netG = netG
        self.netD = netD
        self.criterion = criterion
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.num_epochs = num_epochs
        self.step = 0
        self.fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
        self.latent_dim = latent_dim

    def train(self) -> None:
        save_interval = 10
        for epoch in range(self.num_epochs):
            self._train_epoch(epoch)
            if (epoch % save_interval == 0) or (epoch + 1 == self.num_epochs):
                self.save_checkpoint(epoch)

    def _train_epoch(self, epoch: int) -> None:
        self.netD.train()
        self.netG.train()
        real_label = 1.0
        fake_label = 0.0
        accum_loss_G, accum_loss_D = 0., 0.

        num_batches = len(self.train_dataloader)
        for batch_num, images in enumerate(tqdm(self.train_dataloader, desc="Train")):
            # Optimize discriminator
            self.netD.zero_grad()

            # Real batch
            real_cpu = images[0].to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full(
                (b_size,), real_label, dtype=torch.float, device=self.device
            )

            output = self.netD(real_cpu).view(-1)
            errD_real = self.criterion(output, label)

            errD_real.backward()
            D_x = output.mean().item()

            # Fake batch
            noise = torch.randn(b_size, self.latent_dim, 1, 1, device=self.device)
            fake = self.netG(noise)
            label.fill_(fake_label)
            output = self.netD(fake.detach()).view(-1)
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.optimizerD.step()

            # Optimize generator
            self.netG.zero_grad()
            label.fill_(real_label)
            output = self.netD(fake).view(-1)
            errG = self.criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.optimizerG.step()
            self.step += 1
            
            self.G_scheduler.step()
            self.D_scheduler.step()

            if (self.step == 1 or self.step % 500 == 0) or (
                (epoch == self.num_epochs - 1) and (batch_num == num_batches - 1)
            ):
                with torch.no_grad():
                    fake = self.netG(self.fixed_noise).detach().cpu()
                wandb.log(
                    {
                        "generated": wandb.Image(
                            vutils.make_grid(fake, padding=2, normalize=True)
                        )
                    },
                    step=self.step
                )
                
            accum_loss_G += errG.item()
            accum_loss_D += errD.item()

            if batch_num == num_batches - 1:
                wandb.log(
                    {
                        "epoch": epoch,
                        "lr": self.optimizerG.param_groups[0]["lr"],
                        "G_loss": accum_loss_G / num_batches,
                        "G grad norm": self.get_grad_norm(self.netG.parameters()),
                        "D_loss": accum_loss_D / num_batches,
                        "D grad norm": self.get_grad_norm(self.netD.parameters()),
                    },
                    step=self.step
                )

    @torch.no_grad()
    def get_grad_norm(self, parameters, norm_type=2):
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
        state = {
            "epoch": epoch,
            "G_state_dict": self.netG.state_dict(),
            "D_state_dict": self.netD.state_dict(),
            "G_optimizer": self.optimizerG.state_dict(),
            "D_optimizer": self.optimizerD.state_dict(),
            "G_scheduler": self.G_scheduler.state_dict(),
            "D_scheduler": self.D_scheduler.state_dict(),
        }
        filename = f"ckpts/checkpoint-epoch{epoch}.pth"
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        self.log_ckpt(filename, "checkpoint")

    def log_ckpt(self, ckpt_path: str, cktp_name: str):
        artifact = wandb.Artifact(name=cktp_name, type="model")
        artifact.add_file(local_path=ckpt_path)
        wandb.log_artifact(artifact)

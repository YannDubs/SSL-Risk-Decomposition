
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from typing import Any, Optional
from torchmetrics.functional import accuracy


from utils.architectures import get_Architecture
from utils.optim import LARS, LinearWarmupCosineAnnealingLR


class Predictor(pl.LightningModule):
    """Main network for downstream prediction."""

    def __init__(
        self, hparams: Any, z_dim: int, n_labels: int
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.z_dim = z_dim
        self.n_labels = n_labels
        Architecture = get_Architecture(**self.hparams.predictor.arch_kwargs)
        self.predictor = Architecture(self.z_dim, self.n_labels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Perform prediction for `z`.

        Parameters
        ----------
        z : torch.Tensor of shape=[batch_size, *data.shape]
            Data to represent.
        """
        # shape: [batch_size,  y_dim]
        Y_pred = self.predictor(z)
        return Y_pred

    def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, dict, dict]:
        x, y = batch

        # list of Y_hat. Each Y_hat shape: [batch_size,  y_dim]
        Y_hat = self(x)

        # Shape: []
        loss, logs, other = self.loss(Y_hat, y)

        return loss, logs, other

    def loss(self, Y_hat: torch.Tensor, y: torch.Tensor,) -> tuple[torch.Tensor, dict, dict]:
        """Compute the MSE or cross entropy loss."""
        loss = F.cross_entropy(Y_hat, y.squeeze().long())

        logs = dict()
        logs["acc"] = accuracy(Y_hat.argmax(dim=-1), y)
        logs["err"] = 1 - logs["acc"]
        logs["loss"] = loss

        return loss, logs, dict()

    def shared_step(
        self, batch: torch.Tensor, mode: str, *args, **kwargs
    ) -> Optional[torch.Tensor]:
        loss, logs, other = self.step(batch)

        name = f"{mode}/{self.hparams.data.name}/{self.hparams.component}"
        self.log_dict(
            {
                f"{name}/{k}": v
                for k, v in logs.items()
            },
        )
        return loss


    def test_step(self, batch, *args, **kwargs):
        return self.shared_step(batch, "test", *args, **kwargs)

    def training_step(self, batch, *args, **kwargs):
        return self.shared_step(batch, "train", *args, **kwargs)

    def configure_optimizers(self):

        cfgo = self.hparams.predictor.opt_kwargs
        # linear warmup depending on batch size
        linear_lr = cfgo.lr * self.hparams.data.kwargs.batch_size / 256

        if cfgo.optim == "sgd":
            optimizer = torch.optim.SGD(
                self.predictor.parameters(),
                linear_lr,
                momentum=0.9,
                weight_decay=cfgo.weight_decay
            )

        elif cfgo.optim == "adam":
            # makes adam and sgd lr more similar: SGD(0.1) ~ Adam(1e-3).
            # less dependence between params simplifies hyperparameter tuning
            linear_lr = linear_lr / 100
            optimizer = torch.optim.Adam(
                self.predictor.parameters(),
                lr=linear_lr,
                weight_decay=cfgo.weight_decay,
                eps=1e-4,  # can give issues due to fp16
            )

        elif cfgo.optim == "lars":
            optimizer = LARS(
                self.predictor.parameters(),
                linear_lr,
                momentum=0.9,
                weight_decay=cfgo.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer={cfgo.optim}.")


        if cfgo.scheduler == "warmcosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.hparams.trainer.max_epochs,
                eta_min=0
            )

        elif cfgo.scheduler == "cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=round(self.hparams.trainer.max_epochs / 10),  # 10%
                max_epochs=self.hparams.trainer.max_epochs,
                eta_min=0
            )
        elif cfgo.scheduler == "multistep":
            e = self.hparams.trainer.max_epochs
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(0.6*e), int(0.8*e), int(0.95*e)],
                gamma=0.1,
            )
        else:
            raise ValueError(f"Unknown scheduler={cfgo.optim}.")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


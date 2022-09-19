import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VITWrapper(nn.Module):
    """
    Follows https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf
    /eval_linear.py#L196

    Parameters
    ----------
    encoder : model
        VIT model.

    representation : {"cls", "cls+avg", "4xcls"}
        Which feature to use as the representation.

    is_interp_pos_encoding : bool
        Whether to interpolate the position encoding to the input size. Should not be needed if using the same input size as trained on.
    """
    def __init__(self, encoder : nn.Module, representation: str,  is_interp_pos_encoding : bool = True):
        super().__init__()
        self.encoder = encoder
        self.is_interp_pos_encoding = is_interp_pos_encoding

        if representation == "cls+avg":
            self.n_last_blocks = 1
            self.avgpool_patchtokens = True
        elif representation == "4xcls":
            self.n_last_blocks = 4
            self.avgpool_patchtokens = False
        elif representation == "cls":
            self.n_last_blocks = 1
            self.avgpool_patchtokens = False
        else:
            raise ValueError(f"Unknown extract_mode={representation}")

    def forward(self, x: torch.Tensor):
        intermediate_output = self.encoder.get_intermediate_layers(x, self.n_last_blocks)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if self.avgpool_patchtokens:
            output = torch.cat((output.unsqueeze(-1),
                                torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            output = output.reshape(output.shape[0], -1)

        return output

    def interpolate_pos_encoding(self, x, pos_embed):
        """Interpolated the position encoding to the input size. Should not be needed if using the same input size as trained on."""
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = F.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)
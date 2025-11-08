import torch
from torch import nn


class AspectRatioEmbedding(nn.Module):
    """
    Project aspect ratio values into the UNet timestep embedding space.

    The input can be a scalar, a tensor of shape [B], or [B, 1]. It is always
    reshaped to [B, 1] (or [1, 1] for a scalar) and converted to float before
    projection. All linear layers are zero-initialized so the embedding starts
    as a no-op until trained.
    """

    def __init__(self, timestep_emb_dim: int, hidden_dim: int = 350):
        super().__init__()
        self.timestep_emb_dim = timestep_emb_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(1, hidden_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, timestep_emb_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in (self.linear1, self.linear2):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, aspect_ratio):
        if not torch.is_tensor(aspect_ratio):
            aspect_ratio = torch.tensor(
                aspect_ratio, dtype=torch.float32, device=self.linear1.weight.device
            )
        else:
            aspect_ratio = aspect_ratio.to(
                device=self.linear1.weight.device, dtype=torch.float32
            )

        if aspect_ratio.dim() == 0:
            aspect_ratio = aspect_ratio.unsqueeze(0)

        if aspect_ratio.dim() == 1:
            aspect_ratio = aspect_ratio.unsqueeze(-1)
        elif aspect_ratio.dim() == 2 and aspect_ratio.size(-1) == 1:
            pass
        else:
            raise ValueError(
                f"Aspect ratio must have shape [], [B], or [B, 1]; got {tuple(aspect_ratio.shape)}."
            )

        return self.linear2(self.act(self.linear1(aspect_ratio)))


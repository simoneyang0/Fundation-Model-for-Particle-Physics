# PATCHED MINI OMNIJET (STABLE VQ-VAE + COMPATIBLE BACKBONE)

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

# ----------------------------------------------------------------------
# ---------------------- STABLE VECTOR QUANTIZER -----------------------
# ----------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=256, code_dim=16, commitment_cost=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.1)

    def forward(self, z, mask=None):
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)

        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.codebook.weight ** 2).sum(dim=1)
        distances = z_sq + e_sq - 2 * z_flat @ self.codebook.weight.t()

        indices = torch.argmin(distances, dim=-1).view(B, T)
        z_q = self.codebook(indices)

        z_q_st = z + (z_q - z).detach()

        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            z = z * mask_exp
            z_q = z_q * mask_exp

        vq_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = self.commitment_cost * F.mse_loss(z_q, z.detach())

        return {
            "z_q": z_q,
            "z_q_st": z_q_st,
            "indices": indices,
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
        }

# ----------------------------------------------------------------------
# ---------------------- NORMFORMER ------------------------------------
# ----------------------------------------------------------------------

class NormformerBlockMini(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.norm1.weight)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=(mask == 0) if mask is not None else None
        )
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x

class NormformerStackMini(nn.Module):
    def __init__(self, dim, num_heads=2, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            NormformerBlockMini(dim, num_heads)
            for _ in range(num_blocks)
        ])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x

# ----------------------------------------------------------------------
# ---------------------- VQ-VAE ----------------------------------------
# ----------------------------------------------------------------------

class MiniVQVAE(nn.Module):
    def __init__(self, input_dim=4, latent_dim=16, hidden_dim=64, num_codes=256):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.encoder = NormformerStackMini(hidden_dim)
        self.latent_projection_in = nn.Linear(hidden_dim, latent_dim)

        self.vq_layer = VectorQuantizer(num_codes, latent_dim)

        self.latent_projection_out = nn.Linear(latent_dim, hidden_dim)
        self.decoder = NormformerStackMini(hidden_dim)

        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, mask=None):
        h = self.input_projection(x)
        h = self.encoder(h, mask)

        z = self.latent_projection_in(h)
        if mask is not None:
            z = z * mask.unsqueeze(-1)

        vq_out = self.vq_layer(z, mask)

        h_reco = self.latent_projection_out(vq_out["z_q_st"])
        if mask is not None:
            h_reco = h_reco * mask.unsqueeze(-1)

        h_reco = self.decoder(h_reco, mask)
        x_reco = self.output_projection(h_reco)

        if mask is not None:
            x_reco = x_reco * mask.unsqueeze(-1)

        return {
            "reconstructed": x_reco,
            "indices": vq_out["indices"],
            "vq_loss": vq_out["vq_loss"],
            "commitment_loss": vq_out["commitment_loss"],
        }

# ----------------------------------------------------------------------
# ---------------------- LOSS ------------------------------------------
# ----------------------------------------------------------------------

def vqvae_loss(output, target, mask, beta=0.25):
    mask = mask.unsqueeze(-1)
    diff = (output["reconstructed"] - target) * mask
    recon_loss = (diff ** 2).sum() / mask.sum()

    total = recon_loss + beta * (output["vq_loss"] + output["commitment_loss"])
    return total, recon_loss

# ----------------------------------------------------------------------
# ---------------------- MINI OMNIJET ----------------------------------
# ----------------------------------------------------------------------

class MiniOmniJet(nn.Module):
    def __init__(self, input_dim=4, num_codes=256):
        super().__init__()
        self.vqvae = MiniVQVAE(input_dim=input_dim, num_codes=num_codes)

    def forward_vqvae(self, x, mask):
        return self.vqvae(x, mask)

# ----------------------------------------------------------------------
# ---------------------- LIGHTNING -------------------------------------
# ----------------------------------------------------------------------

class MiniOmniJetLightning(L.LightningModule):
    def __init__(self, optimizer):
        super().__init__()
        self.model = MiniOmniJet()
        self.optimizer_fn = optimizer

    def training_step(self, batch, batch_idx):
        out = self.model.forward_vqvae(batch['part_features'], batch['part_mask'])

        loss, recon = vqvae_loss(
            out,
            batch['part_features'],
            batch['part_mask']
        )

        self.log("train_loss", loss)
        self.log("recon_loss", recon)

        return loss

    def configure_optimizers(self):
        return self.optimizer_fn(self.parameters())



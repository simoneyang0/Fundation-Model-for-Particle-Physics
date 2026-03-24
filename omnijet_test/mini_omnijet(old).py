# mini_omnijet_complete.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L 
import numpy as np
from typing import Dict, Any, Tuple, Optional

# ----------------------------------------------------------------------
# -------------------------- VQ-VAE RIDOTTO ----------------------------
# ----------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Vector Quantization layer - versione ridotta"""
    def __init__(self, num_codes=256, code_dim=16, commitment_cost=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        
        # Codebook apprendibile
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)
        
    def forward(self, z):
        # z: [batch, seq_len, code_dim]
        batch_size, seq_len, _ = z.shape
        
        # Calcola distanze
        z_flat = z.reshape(-1, self.code_dim)
        distances = torch.cdist(z_flat, self.codebook.weight)
        
        # Trova codici più vicini
        code_indices = torch.argmin(distances, dim=-1)
        code_indices = code_indices.reshape(batch_size, seq_len)
        
        # Quantizzazione
        z_q = self.codebook(code_indices)
        
        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()
        
        # Losses
        vq_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = F.mse_loss(z_q, z.detach()) * self.commitment_cost
        
        return {
            'z_q': z_q,
            'z_q_st': z_q_st,
            'indices': code_indices,
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss
        }


class NormformerBlockMini(nn.Module):
    """Normformer block ridotto"""
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Pre-norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head attention (ridotta)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # MLP ridotto (2x invece di 4x)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
        # Inizializzazione a zero per residui
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.norm1.weight)
        
    def forward(self, x, mask=None):
        # x: [batch, seq_len, dim]
        # mask: [batch, seq_len] (1 per token reale)
        
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=(mask == 0) if mask is not None else None
        )
        x = x + attn_out
        
        # MLP with pre-norm
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


class NormformerStackMini(nn.Module):
    """Stack di Normformer block ridotti"""
    def __init__(self, dim, num_heads=2, num_blocks=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            NormformerBlockMini(dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


class MiniVQVAE(nn.Module):
    def __init__(
        self,
        input_dim=4,
        latent_dim=16,
        hidden_dim=64,
        num_codes=256,
        num_heads=2,
        num_blocks=3,  # Aumentato
        commitment_cost=0.25,
        dropout=0.1  # <-- AGGIUNTO
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_codes = num_codes
        self.dropout = dropout  # <-- AGGIUNTO
        
        # Encoder con dropout
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.encoder = NormformerStackMini(
            dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout  # <-- PASSATO
        )
        self.latent_projection_in = nn.Linear(hidden_dim, latent_dim)
        
        # Vector Quantization
        self.vq_layer = VectorQuantizer(
            num_codes=num_codes,
            code_dim=latent_dim,
            commitment_cost=commitment_cost
        )
        
        # Decoder con dropout
        self.latent_projection_out = nn.Linear(latent_dim, hidden_dim)
        self.decoder = NormformerStackMini(
            dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout  # <-- PASSATO
        )
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # Dropout layers aggiuntivi
        self.dropout_layer = nn.Dropout(dropout)

        self._init_weights()  # <-- Aggiungi questa chiamata
    
    def _init_weights(self):  # <-- Aggiungi questo metodo
        """Inizializzazione pesi"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        
    def forward(self, x, mask=None):
        h = self.input_projection(x)
        h = self.dropout_layer(h)  # <-- AGGIUNTO
        h = self.encoder(h, mask=mask)
        z = self.latent_projection_in(h)
        
        if mask is not None:
            z = z * mask.unsqueeze(-1)
        
        vq_out = self.vq_layer(z)
        
        h_reco = self.latent_projection_out(vq_out['z_q_st'])
        if mask is not None:
            h_reco = h_reco * mask.unsqueeze(-1)
        h_reco = self.dropout_layer(h_reco)  # <-- AGGIUNTO
        h_reco = self.decoder(h_reco, mask=mask)
        x_reco = self.output_projection(h_reco)
        if mask is not None:
            x_reco = x_reco * mask.unsqueeze(-1)
        
        return {
            'reconstructed': x_reco,
            'indices': vq_out['indices'],
            'vq_loss': vq_out['vq_loss'],
            'commitment_loss': vq_out['commitment_loss'],
            'z_q': vq_out['z_q']
        }
    
    def tokenize(self, x, mask=None):
        """Converte input in token indices"""
        h = self.input_projection(x)
        h = self.encoder(h, mask=mask)
        z = self.latent_projection_in(h)
        if mask is not None:
            z = z * mask.unsqueeze(-1)
        vq_out = self.vq_layer(z)
        return vq_out['indices']
    
    def decode_tokens(self, tokens):
        """Decodifica token in particelle"""
        z_q = self.vq_layer.codebook(tokens)
        h = self.latent_projection_out(z_q)
        h = self.decoder(h)
        x = self.output_projection(h)
        return x


# ----------------------------------------------------------------------
# -------------------------- BACKBONE GPT RIDOTTO ----------------------
# ----------------------------------------------------------------------

class MultiHeadAttentionMini(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0.1, max_seq_len=128):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout  # <-- SALVATO
        
        self.key = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        
        self.dropout_layer = nn.Dropout(dropout)  # <-- AGGIUNTO
        self.proj = nn.Linear(dim, dim)
        
        self.register_buffer("tril", torch.tril(torch.ones(max_seq_len, max_seq_len)))
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        if mask is not None:
            padding_mask = (mask == 0).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(padding_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)  # <-- AGGIUNTO dropout dopo softmax
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        
        return out


class FeedForwardMini(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # <-- AGGIUNTO
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)   # <-- AGGIUNTO
        )
        
    def forward(self, x):
        return self.net(x)


class GPTDecoderBlockMini(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0.1, max_seq_len=128):
        super().__init__()
        
        self.attention = MultiHeadAttentionMini(
            dim, num_heads, dropout, max_seq_len
        )
        self.ffn = FeedForwardMini(dim, dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class MiniBackbone(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=64,
        num_heads=2,
        num_layers=4,
        max_seq_len=128,
        dropout=0.1,
        return_embeddings=True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.return_embeddings = return_embeddings
        self.dropout = dropout
        
        self.embedding_table = nn.Embedding(vocab_size, dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            GPTDecoderBlockMini(dim, num_heads, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        
        self._init_weights()  # <-- Chiamata corretta
        
    def _init_weights(self):  # <-- Metodo aggiunto
        """Inizializzazione pesi"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        
    def forward(self, x, mask=None):
        x = self.embedding_table(x)
        x = self.dropout_layer(x)
        
        for block in self.blocks:
            x = block(x, mask)
            
        return x


# ----------------------------------------------------------------------
# -------------------------- HEADS -------------------------------------
# ----------------------------------------------------------------------

class NextTokenPredictionHead(nn.Module):
    """Head per next-token prediction - come in backbone.py"""
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(dim, vocab_size)
        
    def forward(self, x):
        # x: [batch, seq_len, dim]
        return self.linear(x)


class ClassificationHeadSum(nn.Module):
    def __init__(self, dim, num_classes, dropout=0.1):  # <-- AGGIUNTO
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(dropout)  # <-- AGGIUNTO
        self.activation = nn.ReLU()
        
    def forward(self, x, mask):
        embeddings = self.linear1(x)
        embeddings = self.activation(embeddings)
        embeddings = self.dropout(embeddings)  # <-- AGGIUNTO
        
        embeddings_sum = torch.sum(embeddings * mask.unsqueeze(-1), dim=1)
        
        logits = self.linear2(embeddings_sum)
        return logits


class ClassificationHeadAttention(nn.Module):
    """Classification head con attention - come ClassifierNormformer in backbone.py"""
    def __init__(self, dim, num_classes, num_heads=2, num_blocks=2, dropout=0.1):
        super().__init__()
        
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.cross_attn_blocks = nn.ModuleList([
            NormformerBlockMini(dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x, mask):
        B = x.shape[0]
        
        # Espandi class token
        class_token = self.class_token.expand(B, -1, -1)
        
        # Concatena class token con x
        x_with_token = torch.cat([class_token, x], dim=1)
        
        # Estendi maschera per class token
        mask_with_token = torch.cat([torch.ones(B, 1, device=x.device), mask], dim=1)
        
        # Applica blocchi
        h = x_with_token
        for block in self.cross_attn_blocks:
            h = block(h, mask_with_token)
        
        # Prendi class token
        class_out = h[:, 0, :]
        
        # MLP finale
        logits = self.mlp(class_out)
        return logits


# ----------------------------------------------------------------------
# -------------------------- MODELLO COMPLETO -------------------------
# ----------------------------------------------------------------------

class MiniOmniJet(nn.Module):
    def __init__(
        self,
        # VQ-VAE params
        input_dim=4,
        latent_dim=16,
        vq_hidden_dim=64,
        num_codes=256,
        vq_num_heads=2,
        vq_num_blocks=2,
        commitment_cost=0.25,
        
        # Backbone params
        backbone_dim=64,
        backbone_num_heads=2,
        backbone_num_layers=4,
        max_seq_len=128,
        dropout=0.1,  # <-- DROPOUT GLOBALE
        
        # Classification params
        num_classes=10,
        
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_codes = num_codes
        self.vocab_size = num_codes + 2
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # 1. VQ-VAE con dropout
        self.vqvae = MiniVQVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=vq_hidden_dim,
            num_codes=num_codes,
            num_heads=vq_num_heads,
            num_blocks=vq_num_blocks,
            commitment_cost=commitment_cost,
            dropout=dropout
        )
        
        # 2. Backbone GPT con dropout
        self.backbone = MiniBackbone(
            vocab_size=self.vocab_size,
            dim=backbone_dim,
            num_heads=backbone_num_heads,
            num_layers=backbone_num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            return_embeddings=True
        )
        
        # 3. Heads con dropout
        self.generative_head = NextTokenPredictionHead(backbone_dim, self.vocab_size)
        self.classification_head = ClassificationHeadSum(backbone_dim, num_classes, dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward_vqvae(self, x, mask=None):
        """Forward VQ-VAE per training"""
        return self.vqvae(x, mask)
    
    def tokenize(self, x, mask=None):
        """Converte particelle in token"""
        return self.vqvae.tokenize(x, mask)
    
    def forward_generative(self, token_indices, mask=None, return_loss=True):
        """
        Forward per training generativo (next-token prediction)
        token_indices: [batch, seq_len]
        """
        # Embedding e backbone
        embeddings = self.backbone(token_indices, mask)
        
        # Head generativa
        logits = self.generative_head(embeddings)  # [batch, seq_len, vocab_size]
        
        if return_loss:
            # Shift per next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = token_indices[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-1
            )
            return logits, loss
        
        return logits
    
    def forward_classification(self, x, mask=None):
        """
        Forward per classificazione
        x: particelle [batch, seq_len, input_dim] o token [batch, seq_len]
        """
        # Se input sono particelle, tokenizza
        if len(x.shape) == 3 and x.shape[-1] == self.input_dim:
            with torch.no_grad():
                token_indices = self.tokenize(x, mask)
        else:
            token_indices = x
            
        # Backbone
        embeddings = self.backbone(token_indices, mask)
        
        # Classification head
        logits = self.classification_head(embeddings, mask)
        
        return logits
    
    def generate(self, start_token=None, max_new_tokens=128, temperature=1.0, device='cuda'):
        """
        Genera jet autoregressivamente
        start_token: None per START, o token iniziale
        """
        self.eval()
        
        if start_token is None:
            # Usa START token (indice num_codes)
            start_token = torch.tensor([[self.num_codes]], device=device)
        elif isinstance(start_token, int):
            start_token = torch.tensor([[start_token]], device=device)
        
        generated = start_token
        
        with torch.no_grad():
            for _ in range(max_new_tokens - start_token.shape[1]):
                # Forward
                logits = self.forward_generative(generated, return_loss=False)
                
                # Prendi ultimo token
                next_logits = logits[:, -1, :] / temperature
                
                # Sampling (escludi START token)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Concatena
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop se tutti hanno STOP (indice num_codes+1)
                if (next_token == self.num_codes + 1).all():
                    break
        
        return generated


# ----------------------------------------------------------------------
# -------------------------- LIGHTNING MODULE -------------------------
# ----------------------------------------------------------------------

class MiniOmniJetLightning(L.LightningModule):
    """
    PyTorch Lightning module per MiniOmniJet
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs: dict = {},
        task: str = 'generative',  # 'generative', 'classification', 'vqvae'
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.model = MiniOmniJet(**model_kwargs)
        self.task = task
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_loss_history = []
        self.val_loss_history = []
        
    def forward(self, batch):
        if self.task == 'vqvae':
            x = batch['part_features']
            mask = batch['part_mask']
            return self.model.forward_vqvae(x, mask)
            
        elif self.task == 'generative':
            # Per generativo, ci aspettiamo già token
            x = batch['part_features'].squeeze().long()
            mask = batch['part_mask']
            return self.model.forward_generative(x, mask)
            
        elif self.task == 'classification':
            x = batch['part_features']
            mask = batch['part_mask']
            return self.model.forward_classification(x, mask)
    
    def training_step(self, batch, batch_idx):
        if self.task == 'vqvae':
            out = self.forward(batch)
            recon_loss = F.mse_loss(out['reconstructed'], batch['part_features'])
            total_loss = recon_loss + out['vq_loss'] + out['commitment_loss']
            
            self.log('train_recon_loss', recon_loss, prog_bar=True)
            self.log('train_vq_loss', out['vq_loss'], prog_bar=True)
            
        elif self.task == 'generative':
            _, loss = self.forward(batch)
            total_loss = loss
            
        elif self.task == 'classification':
            logits = self.forward(batch)
            labels = F.one_hot(batch['jet_type_labels'].squeeze(), num_classes=10).float()
            total_loss = self.criterion(logits, labels)
            
            # Accuratezza
            preds = torch.argmax(logits, dim=1)
            true = torch.argmax(labels, dim=1)
            acc = (preds == true).float().mean()
            self.log('train_acc', acc, prog_bar=True)
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.train_loss_history.append(total_loss.item())
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        if self.task == 'vqvae':
            out = self.forward(batch)
            recon_loss = F.mse_loss(out['reconstructed'], batch['part_features'])
            total_loss = recon_loss + out['vq_loss'] + out['commitment_loss']
            
        elif self.task == 'generative':
            _, loss = self.forward(batch)
            total_loss = loss
            
        elif self.task == 'classification':
            logits = self.forward(batch)
            labels = F.one_hot(batch['jet_type_labels'].squeeze(), num_classes=10).float()
            total_loss = self.criterion(logits, labels)
            
            preds = torch.argmax(logits, dim=1)
            true = torch.argmax(labels, dim=1)
            acc = (preds == true).float().mean()
            self.log('val_acc', acc, prog_bar=True)
        
        self.log('val_loss', total_loss, prog_bar=True)
        self.val_loss_history.append(total_loss.item())
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


# ----------------------------------------------------------------------
# -------------------------- FUNZIONI DI UTILITÀ ----------------------
# ----------------------------------------------------------------------

def count_parameters(model):
    """Conta i parametri del modello"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """Stampa riepilogo del modello"""
    print("=" * 60)
    print("MODELLO MINIOMNIJET")
    print("=" * 60)
    
    total_params = count_parameters(model)
    print(f"Parametri totali: {total_params:,}")
    
    # Dettaglio per componente
    vqvae_params = count_parameters(model.vqvae)
    backbone_params = count_parameters(model.backbone)
    heads_params = (
        count_parameters(model.generative_head) +
        count_parameters(model.classification_head)
    )
    
    print(f"  VQ-VAE:       {vqvae_params:>8,} parametri")
    print(f"  Backbone GPT: {backbone_params:>8,} parametri")
    print(f"  Heads:        {heads_params:>8,} parametri")
    print("=" * 60)
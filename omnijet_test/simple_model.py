# simple_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleJetModel(nn.Module):
    """
    Modello semplificato per classificazione jet
    - Usa solo le particelle (non le feature globali del jet)
    - Transformer encoder per processare le particelle
    """
    
    def __init__(self,
                 num_particle_features=4,   # part_pt, eta, phi, energy
                 num_jet_features=4,         # jet_pt, eta, phi, energy
                 num_classes=10,              # 10 processi JetClass
                 d_model=128,
                 nhead=4,
                 num_layers=4,
                 max_particles=128,
                 dropout=0.1):
        
        super().__init__()
        
        self.d_model = d_model
        self.max_particles = max_particles
        
        # 1. Embedding delle particelle
        self.particle_embedding = nn.Sequential(
            nn.Linear(num_particle_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Positional encoding (apprendibile)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_particles, d_model) * 0.02
        )
        
        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 4. Pooling e classificazione
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, particles, mask=None):
        """
        particles: [batch, max_particles, num_particle_features]
        mask: [batch, max_particles] (True per particelle reali)
        """
        batch_size = particles.shape[0]
        
        # Embedding
        x = self.particle_embedding(particles)  # [batch, max_particles, d_model]
        
        # Aggiungi positional encoding
        x = x + self.pos_embedding
        
        # Transformer (con maschera per padding)
        # src_key_padding_mask = True per posizioni da ignorare
        src_key_padding_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Pooling: media pesata sulle particelle reali
        if mask is not None:
            # Maschera le particelle di padding
            mask_expanded = mask.unsqueeze(-1).float()  # [batch, max_particles, 1]
            x_masked = x * mask_expanded
            # Somma e dividi per numero di particelle reali
            x_pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
        else:
            x_pooled = x.mean(dim=1)
        
        # Classificazione
        logits = self.classifier(x_pooled)
        
        return logits


class SimpleJetModelWithGlobal(nn.Module):
    """
    Versione che usa anche le feature globali del jet
    """
    
    def __init__(self,
                 num_particle_features=4,
                 num_jet_features=4,
                 num_classes=10,
                 d_model=128,
                 nhead=4,
                 num_layers=4,
                 max_particles=128,
                 dropout=0.1):
        
        super().__init__()
        
        self.d_model = d_model
        self.max_particles = max_particles
        
        # 1. Embedding particelle
        self.particle_embedding = nn.Sequential(
            nn.Linear(num_particle_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Embedding jet globali
        self.jet_embedding = nn.Sequential(
            nn.Linear(num_jet_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_particles + 1, d_model) * 0.02  # +1 per il token globale
        )
        
        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 5. Classificatore
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, particles, jets, mask=None):
        """
        particles: [batch, max_particles, num_particle_features]
        jets: [batch, num_jet_features]
        mask: [batch, max_particles] (True per particelle reali)
        """
        batch_size = particles.shape[0]
        
        # Embedding particelle
        x_particles = self.particle_embedding(particles)  # [batch, max_particles, d_model]
        
        # Embedding jet globale (diventa un token speciale)
        x_jet = self.jet_embedding(jets).unsqueeze(1)  # [batch, 1, d_model]
        
        # Concatena: [CLS] token + particelle
        x = torch.cat([x_jet, x_particles], dim=1)  # [batch, 1+max_particles, d_model]
        
        # Aggiungi positional encoding
        x = x + self.pos_embedding[:, :x.shape[1], :]
        
        # Crea maschera combinata (il token [CLS] è sempre reale)
        if mask is not None:
            # Aggiungi True per il token [CLS]
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            combined_mask = torch.cat([cls_mask, mask], dim=1)
            src_key_padding_mask = ~combined_mask
        else:
            src_key_padding_mask = None
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Usa il token [CLS] per la classificazione
        cls_token = x[:, 0, :]  # [batch, d_model]
        
        # Classificazione
        logits = self.classifier(cls_token)
        
        return logits
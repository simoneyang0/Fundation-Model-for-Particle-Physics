# FULL CORRECTED PIPELINE: DATASET + VQ-VAE READY

import torch
import numpy as np
from torch.utils.data import Dataset

# ----------------------------------------------------------------------
# ---------------------- NORMALIZED DATASET (FIXED) --------------------
# ----------------------------------------------------------------------

class NormalizedJetClassDataset(Dataset):
    """Jet dataset con preprocessing corretto stile OmniJet"""

    def __init__(self, base_dataset, norm_stats=None, max_samples=10000):
        self.base_dataset = base_dataset

        if norm_stats is None:
            print("Calcolo statistiche (corrette)...")
            all_features = []

            for i in range(min(max_samples, len(base_dataset))):
                sample = base_dataset[i]
                particles = sample['particles']
                mask = sample['mask']

                particles = particles[mask]  # SOLO reali

                if len(particles) == 0:
                    continue

                # ---- trasformazioni fisicamente corrette ----
                pt = torch.log1p(particles[:, 0])
                eta = particles[:, 1]
                phi = particles[:, 2]
                energy = torch.log1p(particles[:, 3])

                sin_phi = torch.sin(phi)
                cos_phi = torch.cos(phi)

                feats = torch.stack([pt, eta, sin_phi, cos_phi, energy], dim=1)
                all_features.append(feats.numpy())

            all_features = np.concatenate(all_features, axis=0)

            self.mean = all_features.mean(axis=0)
            self.std = all_features.std(axis=0) + 1e-6

            print("Stats:")
            for i, name in enumerate(["pt", "eta", "sin_phi", "cos_phi", "energy"]):
                print(f"{name}: mean={self.mean[i]:.3f}, std={self.std[i]:.3f}")

        else:
            self.mean = norm_stats['mean']
            self.std = norm_stats['std']

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        particles = item['particles'].clone()
        mask = item['mask']

        out = torch.zeros(particles.shape[0], 5)

        if mask.sum() > 0:
            real = particles[mask]

            pt = torch.log1p(real[:, 0])
            eta = real[:, 1]
            phi = real[:, 2]
            energy = torch.log1p(real[:, 3])

            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)

            feats = torch.stack([pt, eta, sin_phi, cos_phi, energy], dim=1)

            feats = (feats - torch.tensor(self.mean)) / torch.tensor(self.std)

            out[mask] = feats

        return {
            'particles': out,
            'mask': mask,
            'label': item['label']
        }

    def get_norm_stats(self):
        return {'mean': self.mean, 'std': self.std}

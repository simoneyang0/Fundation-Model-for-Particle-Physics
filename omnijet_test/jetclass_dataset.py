import torch
from torch.utils.data import Dataset, DataLoader
from dataloader import read_file
import glob
import numpy as np

class JetClassDataset(Dataset):
    """Dataset PyTorch per JetClass usando il dataloader originale"""
    
    def __init__(self, file_paths, max_num_particles=128, max_jets_per_file=None):
        """
        file_paths: lista di percorsi ai file ROOT
        max_num_particles: numero massimo di costituenti
        max_jets_per_file: limita i jet per file (per test)
        """
        self.max_num_particles = max_num_particles
        self.file_paths = file_paths
        
        # Pre-carica tutti i dati (per dataset piccoli)
        # Per dataset grandi, dovresti caricare on-demand
        self.all_particles = []
        self.all_jets = []
        self.all_labels = []
        
        for file_path in file_paths:
            print(f"Caricamento {file_path}...")
            x_particles, x_jet, y = read_file(
                file_path, 
                max_num_particles=max_num_particles
            )
            
            # Limita il numero di jet se richiesto
            if max_jets_per_file:
                x_particles = x_particles[:max_jets_per_file]
                x_jet = x_jet[:max_jets_per_file]
                y = y[:max_jets_per_file]
            
            self.all_particles.append(x_particles)
            self.all_jets.append(x_jet)
            self.all_labels.append(y)
        
        # Concatena tutti i file
        self.particles = np.concatenate(self.all_particles, axis=0)
        self.jets = np.concatenate(self.all_jets, axis=0)
        self.labels = np.concatenate(self.all_labels, axis=0)
        
        print(f"Totale jet caricati: {len(self.particles)}")
        print(f"Shape particles: {self.particles.shape}")
        print(f"Shape jets: {self.jets.shape}")
        print(f"Shape labels: {self.labels.shape}")
        
        # Converti le etichette one-hot in indici di classe
        # Le etichette sono 10 (una per ogni processo)
        self.class_indices = np.argmax(self.labels, axis=1)
        
    def __len__(self):
        return len(self.particles)
    
    def __getitem__(self, idx):
        # particles: [num_features, max_particles] -> [max_particles, num_features]
        particles = torch.tensor(
            self.particles[idx].T,  # Trasponi per avere [particelle, features]
            dtype=torch.float32
        )
        
        # jets: [num_jet_features]
        jets = torch.tensor(
            self.jets[idx],
            dtype=torch.float32
        )
        
        # Maschera per padding (dove ci sono particelle reali)
        # La prima feature è 'part_pt' - se è 0, è padding
        mask = (particles[:, 0] != 0)
        
        # Etichetta
        label = torch.tensor(
            self.class_indices[idx],
            dtype=torch.long
        )
        
        return {
            'particles': particles,        # [max_particles, num_particle_features]
            'jets': jets,                   # [num_jet_features]
            'mask': mask,                   # [max_particles]
            'label': label,                  # scalare
            'label_onehot': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
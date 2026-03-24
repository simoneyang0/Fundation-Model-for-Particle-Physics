import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class JetDataset(Dataset):
    """Dataset semplificato per jet da file ROOT"""
    
    def __init__(self, file_paths, max_constituents=128, max_jets=None):
        """
        file_paths: lista di percorsi ai file ROOT
        max_constituents: numero massimo di costituenti per jet (pad o tronca)
        max_jets: numero massimo di jet da caricare (per test)
        """
        self.max_constituents = max_constituents
        self.file_paths = file_paths
        
        # Carica tutti i dati
        self.jets = []
        self.labels = []
        
        for i, file_path in enumerate(file_paths):
            print(f"Caricamento {file_path}...")
            jets = self._load_root_file(file_path)
            
            # Assegna etichetta in base al tipo (puoi personalizzare)
            if "HToBB" in file_path:
                label = 0  # segnale
            elif "TTBar" in file_path:
                label = 1  # fondo tipo 1
            elif "WToQQ" in file_path:
                label = 2  # fondo tipo 2
            else:
                label = 3  # altri
                
            # Limita il numero di jet se richiesto
            if max_jets:
                jets = jets[:max_jets]
            
            self.jets.extend(jets)
            self.labels.extend([label] * len(jets))
            
        print(f"Caricati {len(self.jets)} jet")
    
    def _load_root_file(self, file_path):
        """Carica un file ROOT e estrae i jet"""
        with uproot.open(file_path) as file:
            # Esplora la struttura del file per capire i nomi
            print(f"Chiavi nel file {file_path}: {file.keys()}")
            
            # Prova a leggere l'albero (di solito si chiama 'tree' o 'Events')
            tree = file['tree']  # o 'Events' - verifica con print sopra
            
            # Legge le feature dei costituenti
            # Devi adattare questi nomi in base a cosa trovi nel file
            try:
                constituents = tree.arrays([
                    'part_pt', 'part_eta', 'part_phi',  # nomi da verificare!
                ], library="ak")
            except:
                # Prova con nomi alternativi
                constituents = tree.arrays([
                    'pf_pt', 'pf_eta', 'pf_phi',
                ], library="ak")
            
            # Converte in lista di jet
            jets = []
            for i in range(len(constituents)):
                # Estrae i costituenti per questo jet
                pt = constituents['part_pt'][i] if 'part_pt' in constituents.fields else constituents['pf_pt'][i]
                eta = constituents['part_eta'][i] if 'part_eta' in constituents.fields else constituents['pf_eta'][i]
                phi = constituents['part_phi'][i] if 'part_phi' in constituents.fields else constituents['pf_phi'][i]
                
                # Crea array [n_constituents, 3] con pT, eta, phi
                jet_data = np.stack([pt, eta, phi], axis=1)
                
                # Tronca/pad al numero massimo di costituenti
                if len(jet_data) > self.max_constituents:
                    jet_data = jet_data[:self.max_constituents]
                elif len(jet_data) < self.max_constituents:
                    # Padding con zeri
                    pad = np.zeros((self.max_constituents - len(jet_data), 3))
                    jet_data = np.vstack([jet_data, pad])
                
                jets.append(jet_data)
            
            return jets
    
    def __len__(self):
        return len(self.jets)
    
    def __getitem__(self, idx):
        # Jet: [max_constituents, 3] (pt, eta, phi)
        jet = torch.tensor(self.jets[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Crea maschera per padding (True dove ci sono dati veri)
        mask = (jet[:, 0] != 0)  # pt != 0 indica costituente reale
        
        return {
            'constituents': jet,
            'mask': mask,
            'label': label,
            'n_constituents': mask.sum().item()
        }
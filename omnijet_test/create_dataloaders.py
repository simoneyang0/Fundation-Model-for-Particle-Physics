# create_dataloaders.py
import torch
from torch.utils.data import DataLoader, random_split
from my_dataloader import JetDataset
import glob

def create_dataloaders(data_dir, batch_size=32, train_ratio=0.8):
    """
    Crea dataloader di training e validation
    """
    # Trova tutti i file ROOT
    all_files = glob.glob(f"{data_dir}/*.root")
    
    # Separa per tipo (opzionale - per bilanciamento)
    h_files = [f for f in all_files if 'HToBB' in f]
    ttbar_files = [f for f in all_files if 'TTBar' in f]
    w_files = [f for f in all_files if 'WToQQ' in f]
    
    # Prendi un sottoinsieme per test (es. 2 file per classe)
    selected_files = h_files[:1] + ttbar_files[:1] + w_files[:1]
    print(f"File selezionati: {selected_files}")
    
    # Crea dataset completo
    full_dataset = JetDataset(
        file_paths=selected_files,
        max_constituents=128,
        max_jets=50000  # Limita per test
    )
    
    # Split train/val
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Crea dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader
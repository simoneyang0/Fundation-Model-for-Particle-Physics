# train_simple.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from jetclass_dataset import JetClassDataset
from simple_model import SimpleJetModel, SimpleJetModelWithGlobal
import glob
import numpy as np

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        particles = batch['particles'].to(device)
        jets = batch['jets'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Scegli il modello che stai usando
        if isinstance(model, SimpleJetModelWithGlobal):
            logits = model(particles, jets, mask)
        else:
            logits = model(particles, mask)
            
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss/len(loader), correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            particles = batch['particles'].to(device)
            jets = batch['jets'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            if isinstance(model, SimpleJetModelWithGlobal):
                logits = model(particles, jets, mask)
            else:
                logits = model(particles, mask)
                
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss/len(loader), correct/total

def main():
    # Configurazione
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Trova i file (inizia con pochi file per test)
    data_dir = "data/JetClass/Pythia/train_100M"
    all_files = glob.glob(f"{data_dir}/*.root")
    
    # Seleziona 2-3 file per classe per bilanciare
    selected_files = []
    for pattern in ['HToBB', 'TTBar', 'WToQQ']:
        files = [f for f in all_files if pattern in f][:2]  # 2 file per classe
        selected_files.extend(files)
    
    print(f"File selezionati: {[f.split('/')[-1] for f in selected_files]}")
    
    # Crea dataset (con limitazione per test)
    full_dataset = JetClassDataset(
        file_paths=selected_files,
        max_num_particles=128,
        max_jets_per_file=10000  # Limita a 10k jet per file
    )
    
    # Split train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Crea modello
    # model = SimpleJetModel(
    #     num_particle_features=4,  # part_pt, eta, phi, energy
    #     num_classes=10,
    #     d_model=128,
    #     nhead=4,
    #     num_layers=4,
    #     max_particles=128
    # )
    
    # Modello con feature globali
    model = SimpleJetModelWithGlobal(
        num_particle_features=4,
        num_jet_features=4,  # jet_pt, eta, phi, energy
        num_classes=10,
        d_model=128,
        nhead=4,
        num_layers=4,
        max_particles=128
    ).to(device)
    
    print(f"Modello creato con {sum(p.numel() for p in model.parameters()):,} parametri")
    
    # Optimizer e loss
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=10,              # Numero di epoche per completare il ciclo
        eta_min=1e-5            # Learning rate finale desiderato
    )
    
    # Training loop
    best_val_acc = 0
    for epoch in range(10):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        
        print(f"Epoch {epoch}:")
        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        
        # Salva il miglior modello
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  → Nuovo miglior modello salvato!")
    
    print(f"Miglior accuratezza validation: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
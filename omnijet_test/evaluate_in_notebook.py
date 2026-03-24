# evaluate_in_notebook.py
import torch
import numpy as np
import argparse
from train_simple import SimpleJetModelWithGlobal
from jetclass_dataset import JetClassDataset
from torch.utils.data import DataLoader
import glob
import sys

def evaluate_model_notebook(model_path='best_model.pt', 
                           data_dir='data/JetClass/Pythia/train_100M',
                           batch_size=64):
    """
    Versione per Jupyter notebook della valutazione
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Carica modello
    model = SimpleJetModelWithGlobal(
        num_particle_features=4,
        num_jet_features=4,
        num_classes=10,  # il modello è stato addestrato con 10 classi
        d_model=128,
        nhead=4,
        num_layers=4,
        max_particles=128
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modello caricato da {model_path}")
    
    # USA LE STESSE CLASSI DEL TRAINING!
    training_classes = ['HToBB', 'TTBarLep', 'WToQQ']  # le classi su cui è stato addestrato
    
    test_files = []
    for class_name in training_classes:
        files = glob.glob(f"{data_dir}/*{class_name}*.root")
        # Prendi file DIVERSI da quelli usati nel training
        # Escludi i file usati nel training: ['HToBB_001.root', 'HToBB_006.root', ...]
        excluded = ['001', '006', '002', '004', '007']  # numeri dei file usati nel training
        for f in files:
            if not any(ex in f for ex in excluded):
                test_files.append(f)
                print(f"Trovato test file: {f.split('/')[-1]}")
                break  # prendi solo un file per classe
    
    print(f"\nTotale file di test: {len(test_files)}")
    
    # Crea dataset di test
    test_dataset = JetClassDataset(
        file_paths=test_files,
        max_num_particles=128,
        max_jets_per_file=5000
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Dataset di test: {len(test_dataset)} jet")
    print(f"Classi presenti: {np.unique(test_dataset.class_indices)}")
    
    return model, test_loader, device

def plot_results_notebook(model, test_loader, device):
    """
    Plotta i risultati direttamente nel notebook
    """
    from evaluate_metrics import evaluate_model, plot_confusion_matrix, plot_roc_curves
    
    print("\n📊 VALUTAZIONE MODELLO")
    print("=" * 50)
    
    # Metriche base
    results = evaluate_model(model, test_loader, device)
    
    # Matrice di confusione - ora passa i nomi delle classi corretti
    class_names = ['HBB', 'HCC', 'HGG', 'H4q', 'HQQL', 
                   'ZQQ', 'WQQ', 'TBQQ', 'TBL', 'QCD']
    plot_confusion_matrix(model, test_loader, device, class_names=class_names)
    
    # ROC curves
    plot_roc_curves(model, test_loader, device)
    
    return results
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, loader, device, num_classes=None):
    """
    Valutazione completa del modello
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            particles = batch['particles'].to(device)
            jets = batch['jets'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(particles, jets, mask)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Converti a numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Determina il numero di classi dai dati se non specificato
    if num_classes is None:
        num_classes = len(np.unique(all_labels))
    
    # Metriche
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print("=" * 50)
    print("METRICHE GLOBALI")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'num_classes': num_classes
    }


def plot_confusion_matrix(model, loader, device, class_names=None):
    """
    Plotta la matrice di confusione
    """
    results = evaluate_model(model, loader, device)
    
    # Determina il numero di classi dai dati
    n_classes = len(np.unique(results['labels']))
    print(f"Numero di classi rilevato: {n_classes}")
    
    # Crea nomi delle classi in base al numero reale
    if class_names is None:
        all_class_names = ['HBB', 'HCC', 'HGG', 'H4q', 'HQQL', 
                           'ZQQ', 'WQQ', 'TBQQ', 'TBL', 'QCD']
        # Prendi solo i primi n_classes
        class_names = all_class_names[:n_classes]
    else:
        # Se vengono passati nomi, prendi solo i primi n_classes
        class_names = class_names[:n_classes]
    
    print(f"Usando {len(class_names)} classi: {class_names}")
    
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    # Verifica che la matrice abbia la dimensione corretta
    if cm.shape != (n_classes, n_classes):
        print(f"Attenzione: la matrice {cm.shape} non corrisponde a {n_classes} classi. Ricreo...")
        # Ricrea la matrice con la dimensione corretta
        cm = np.zeros((n_classes, n_classes))
        for true, pred in zip(results['labels'], results['predictions']):
            if true < n_classes and pred < n_classes:
                cm[true, pred] += 1
    
    # Normalizza per riga (percentuali)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # sostituisci NaN con 0
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Matrice assoluta
    sns.heatmap(cm.astype(int), annot=True, fmt='d', ax=ax1, 
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues')
    ax1.set_title('Matrice di Confusione - Conteggi Assoluti')
    ax1.set_xlabel('Predetto')
    ax1.set_ylabel('Vero')
    
    # Matrice normalizzata
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('Matrice di Confusione - Normalizzata')
    ax2.set_xlabel('Predetto')
    ax2.set_ylabel('Vero')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    
    # Stampa accuratezza per classe
    print("\n" + "=" * 50)
    print("ACCURATEZZA PER CLASSE")
    print("=" * 50)
    for i, name in enumerate(class_names):
        if i < len(cm_normalized):
            acc = cm_normalized[i, i]
            print(f"{name}: {acc:.2%}")
        else:
            print(f"{name}: N/D (classe non presente)")
    
    return cm, class_names


def plot_roc_curves(model, loader, device, signal_class=0):
    """
    ROC curve per una classe specifica (es. segnale)
    """
    from sklearn.metrics import roc_curve, auc
    
    results = evaluate_model(model, loader, device)
    
    # Per ogni classe, calcola ROC one-vs-rest
    plt.figure(figsize=(10, 8))
    
    n_classes = results['num_classes']
    
    for i in range(n_classes):
        # One-vs-rest: questa classe vs tutte le altre
        y_true = (results['labels'] == i).astype(int)
        y_score = results['probabilities'][:, i]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, 
                 label=f'Classe {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves.png', dpi=150)
    plt.show()
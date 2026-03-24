#!/usr/bin/env python3
"""
Modulo per aggiungere parametri di controllo che confrontano le proprietà dei jet
con la somma delle proprietà dei loro costituenti.
I parametri di controllo sono definiti come: (jet_value - sum_constituents) / mean_jet_value
Per la carica: (jet_charge - sum_constituent_charge) / 1 (nessuna normalizzazione)
"""

import dask.dataframe as dd
import numpy as np
import pandas as pd


def compute_invariant_mass(pt, eta, phi, mass):
    """
    Calcola la massa invariante da 4-vettori.
    M^2 = (E)^2 - (px)^2 - (py)^2 - (pz)^2
    dove E = sqrt(pt^2 + pz^2 + mass^2) e pz = pt * sinh(eta)
    """
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(pt**2 + pz**2 + mass**2)
    
    mass_sq = e**2 - px**2 - py**2 - pz**2
    
    if mass_sq < 0:
        mass_sq = 0
    
    return np.sqrt(mass_sq)


def add_jet_control_parameters(ddf):
    """
    Versione semplificata con solo i parametri di controllo.
    """
    
    def compute_and_flatten_partition(partition):
        all_rows = []
        
        # Calcola medie per normalizzazione
        jet_pt_list = []
        jet_mass_list = []
        
        for _, row in partition.iterrows():
            jet_pt = row.get('FullReco_JetAK4_PT')
            jet_mass = row.get('FullReco_JetAK4_Mass')
            if jet_pt is not None:
                jet_pt_list.extend(jet_pt)
            if jet_mass is not None:
                jet_mass_list.extend(jet_mass)
        
        mean_jet_pt = np.mean(jet_pt_list) if jet_pt_list else 1.0
        #mean_jet_mass = np.mean(jet_mass_list) if jet_mass_list else 1.0
        mean_jet_mass = 91
        
        for _, row in partition.iterrows():
            jet_pt = row.get('FullReco_JetAK4_PT')
            jet_eta = row.get('FullReco_JetAK4_Eta')
            jet_phi = row.get('FullReco_JetAK4_Phi')
            jet_mass = row.get('FullReco_JetAK4_Mass')
            jet_charge = row.get('FullReco_JetAK4_Charge')
            jet_constituents = row.get('FullReco_JetAK4_Constituents')
            
            pfcand_pt = row.get('FullReco_PFCand_PT')
            pfcand_eta = row.get('FullReco_PFCand_Eta')
            pfcand_phi = row.get('FullReco_PFCand_Phi')
            pfcand_mass = row.get('FullReco_PFCand_Mass')
            pfcand_charge = row.get('FullReco_PFCand_Charge')
            
            if (jet_pt is None or jet_constituents is None or pfcand_pt is None):
                continue
            
            for jet_idx in range(len(jet_pt)):
                if jet_idx >= len(jet_constituents):
                    continue
                
                constituents_idx = jet_constituents[jet_idx]
                if constituents_idx is None or len(constituents_idx) == 0:
                    continue
                
                # Massa invariante del jet
                jet_inv_mass = compute_invariant_mass(
                    jet_pt[jet_idx], jet_eta[jet_idx], 
                    jet_phi[jet_idx], jet_mass[jet_idx]
                )
                
                # Somma costituenti
                sum_pt = 0.0
                sum_inv_mass = 0.0
                sum_charge = 0.0
                
                total_px = 0.0
                total_py = 0.0
                total_pz = 0.0
                total_e = 0.0
                
                for idx in constituents_idx:
                    if idx < len(pfcand_pt):
                        sum_pt += pfcand_pt[idx]
                        sum_charge += pfcand_charge[idx]
                        
                        pt = pfcand_pt[idx]
                        eta = pfcand_eta[idx]
                        phi = pfcand_phi[idx]
                        mass = pfcand_mass[idx]
                        
                        px = pt * np.cos(phi)
                        py = pt * np.sin(phi)
                        pz = pt * np.sinh(eta)
                        e = np.sqrt(pt**2 + pz**2 + mass**2)
                        
                        total_px += px
                        total_py += py
                        total_pz += pz
                        total_e += e
                
                inv_mass_sq = total_e**2 - total_px**2 - total_py**2 - total_pz**2
                if inv_mass_sq < 0:
                    inv_mass_sq = 0
                sum_inv_mass = np.sqrt(inv_mass_sq)
                
                # Parametri di controllo
                pt_control = (jet_pt[jet_idx] - sum_pt) / mean_jet_pt if mean_jet_pt > 0 else np.nan
                mass_control = (jet_inv_mass - sum_inv_mass) / mean_jet_mass if mean_jet_mass > 0 else np.nan
                charge_control = jet_charge[jet_idx] - sum_charge
                
                all_rows.append({
                    'mass_control': mass_control,
                    'pt_control': pt_control,
                    'charge_control': charge_control
                })
        
        return pd.DataFrame(all_rows)
    
    result = ddf.map_partitions(compute_and_flatten_partition, meta={
        'mass_control': 'float64',
        'pt_control': 'float64',
        'charge_control': 'float64'
    })
    
    return result


# Esempio di utilizzo
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Carica i dati
    file_path = "data/tuo_file.parquet"
    
    ddf = dd.read_parquet(file_path, columns=[
        "FullReco_JetAK4_PT",
        "FullReco_JetAK4_Eta",
        "FullReco_JetAK4_Phi",
        "FullReco_JetAK4_Mass",
        "FullReco_JetAK4_Charge",
        "FullReco_JetAK4_Constituents",
        "FullReco_PFCand_PT",
        "FullReco_PFCand_Eta",
        "FullReco_PFCand_Phi",
        "FullReco_PFCand_Mass",
        "FullReco_PFCand_Charge"
    ])
    
    print("Dataset caricato")
    
    # Aggiungi parametri di controllo
    ddf_control = add_jet_control_parameters(ddf)
    
    # Calcola e plotta
    print("\nCalcolo dati...")
    data = ddf_control.compute()
    
    # Pulisci dati
    mass_control = data['mass_control'].dropna()
    pt_control = data['pt_control'].dropna()
    charge_control = data['charge_control'].dropna()
    
    print(f"\nDati validi:")
    print(f"  mass_control: {len(mass_control)} valori")
    print(f"  pt_control: {len(pt_control)} valori")
    print(f"  charge_control: {len(charge_control)} valori")
    
    # Istogrammi
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(mass_control, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('(M_jet - M_constituents) / <M_jet>')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Controllo Massa')
    axes[0].axvline(0, color='red', linestyle='--')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(pt_control, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('(PT_jet - PT_constituents) / <PT_jet>')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Controllo PT')
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(charge_control, bins=50, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Q_jet - Q_constituents')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Controllo Carica')
    axes[2].axvline(0, color='red', linestyle='--')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistiche
    print("\nSTATISTICHE:")
    print(f"  Mass control: mean={mass_control.mean():.4f}, std={mass_control.std():.4f}")
    print(f"  PT control: mean={pt_control.mean():.4f}, std={pt_control.std():.4f}")
    print(f"  Charge control: mean={charge_control.mean():.4f}, std={charge_control.std():.4f}")
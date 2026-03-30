#!/usr/bin/env python3
"""
Modulo per aggiungere parametri di controllo che confrontano le proprietà dei jet
con le proprietà aggregate dei loro costituenti.

Parametri:
- charge_control: Q_jet - Q_costituenti (differenza diretta)
- pt_control: (PT_jet / PT_costituenti_vettoriale) - 1
- mass_control: (M_jet / M_costituenti) - 1
"""

import dask.dataframe as dd
import numpy as np
import pandas as pd
import vector


def compute_invariant_mass(pt, eta, phi, mass):
    """Calcola la massa invariante da 4-vettori."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(pt**2 + pz**2 + mass**2)
    
    mass_sq = e**2 - px**2 - py**2 - pz**2
    if mass_sq < 0:
        mass_sq = 0
    
    return np.sqrt(mass_sq)


def add_jet_control_parameters_puppi(ddf, min_jet_pt = -1, no_22 = False, no_neg_jet_m = False): 
    """
    Aggiunge parametri di controllo appiattiti (una riga per jet).
    
    Parametri:
    - charge_control: Q_jet - Q_costituenti
    - pt_control: (PT_jet / PT_costituenti_vettoriale) - 1
    - mass_control: (M_jet / M_costituenti) - 1
    """
    
    def compute_and_flatten_partition(partition):
        all_rows = []
        
        for _, row in partition.iterrows():
            jet_pt = row.get('FullReco_JetPuppiAK4_PT')
            jet_eta = row.get('FullReco_JetPuppiAK4_Eta')
            jet_phi = row.get('FullReco_JetPuppiAK4_Phi')
            jet_mass = row.get('FullReco_JetPuppiAK4_Mass')
            jet_charge = row.get('FullReco_JetPuppiAK4_Charge')
            jet_constituents = row.get('FullReco_JetPuppiAK4_Constituents')
            
            pfcand_pt = row.get('FullReco_PUPPIPart_PT')
            pfcand_eta = row.get('FullReco_PUPPIPart_Eta')
            pfcand_phi = row.get('FullReco_PUPPIPart_Phi')
            pfcand_mass = row.get('FullReco_PUPPIPart_Mass')
            pfcand_charge = row.get('FullReco_PUPPIPart_Charge')
            pfcand_id = row.get('FullReco_PUPPIPart_fUniqueID')
            pfcand_type = row.get('FullReco_PUPPIPart_PID')
            
            if (jet_pt is None or jet_constituents is None or 
                pfcand_pt is None or pfcand_id is None):
                continue
            
            for jet_idx in range(len(jet_pt)):
                if no_neg_jet_m and jet_mass[jet_idx] < 0:
                    continue
                
                if jet_pt[jet_idx] < min_jet_pt: 
                    continue
                
                if jet_idx >= len(jet_constituents):
                    raise RuntimeError("Errore index")
                
                constituents_idx = jet_constituents[jet_idx]
                if constituents_idx is None or len(constituents_idx) == 0:
                    raise RuntimeError("Errore constituents")
                
                # 1. Massa invariante del jet
                jet_inv_mass = jet_mass[jet_idx]
                
                # 2. Proprietà aggregate dei costituenti
                sum_charge = 0.0
                total_px = 0.0
                total_py = 0.0
                total_pz = 0.0
                total_e = 0.0
                n_constituents = 0 
                
                # Cerca ogni costituente usando il suo ID
                for constituent_id in constituents_idx:
                    pos = np.where(pfcand_id == constituent_id)[0]
                    
                    if len(pos) >= 0:
                        idx = pos[0]
                        
                        #scarto particelle di tipo 22
                        if no_22 and pfcand_type[idx] == 22:
                            continue
                        n_constituents += 1
                        
                        # Carica (scalare)
                        charge_val = pfcand_charge[idx]
                        if isinstance(charge_val, (list, np.ndarray)):
                            charge_val = charge_val[0] if len(charge_val) > 0 else 0
                        sum_charge += charge_val
                        
                        # PT vettoriale: componenti cartesiane
                        pt = pfcand_pt[idx]
                        phi = pfcand_phi[idx]
                        px = pt * np.cos(phi)
                        py = pt * np.sin(phi)
                        total_px += px
                        total_py += py
                        
                        # Massa invariante: 4-vettori completi
                        eta = pfcand_eta[idx]
                        mass = pfcand_mass[idx]
                        pz = pt * np.sinh(eta)
                        e = np.sqrt(pt**2 + pz**2 + mass**2)
                        total_pz += pz
                        total_e += e
                
                # PT vettoriale dei costituenti (magnitudo della somma)
                sum_pt = np.sqrt(total_px**2 + total_py**2)
                
                # Massa invariante dei costituenti
                inv_mass_sq = total_e**2 - total_px**2 - total_py**2 - total_pz**2
                if inv_mass_sq < 0:
                    inv_mass_sq = 0
                sum_inv_mass = np.sqrt(inv_mass_sq)

                # DeltaR
                j_4mom = vector.obj(pt=jet_pt[jet_idx],
                           eta=jet_eta[jet_idx],
                           phi=jet_phi[jet_idx],
                           m=jet_mass[jet_idx])
                const_4mom = vector.obj(px = total_px, py = total_py, pz = total_pz, E = total_e)
                
                # Carica del jet
                jet_charge_val = jet_charge[jet_idx]
                if isinstance(jet_charge_val, (list, np.ndarray)):
                    jet_charge_val = jet_charge_val[0] if len(jet_charge_val) > 0 else 0
                
                # Parametri di controllo
                charge_control = jet_charge_val - sum_charge
                
                # PT control: (jet / constituents) - 1
                if sum_pt > 0:
                    pt_control = (sum_pt / jet_pt[jet_idx]) - 1
                else:
                    pt_control = np.nan
                
                # Mass control: (jet / constituents) - 1
                if sum_inv_mass > 0:
                    mass_control = (sum_inv_mass / jet_inv_mass) - 1
                else:
                    mass_control = np.nan
                
                all_rows.append({
                    'charge_control': charge_control,
                    'pt_control': pt_control,
                    'mass_control': mass_control,
                    'n_constituents': n_constituents,
                    'distance': const_4mom.deltaR(j_4mom)
                })
        return pd.DataFrame(all_rows)
    
    result = ddf.map_partitions(compute_and_flatten_partition, meta={
        'charge_control': 'float64',
        'pt_control': 'float64',
        'mass_control': 'float64',
        'n_constituents': 'float64',
        'distance': 'float64'
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
        "FullReco_PFCand_Charge",
        "FullReco_PFCand_fUniqueID"
    ])
    
    print("Dataset caricato")
    
    # Aggiungi parametri di controllo
    ddf_control = add_jet_control_parameters(ddf)
    
    # Calcola e plotta
    print("\nCalcolo dati...")
    data = ddf_control.compute()
    
    # Pulisci dati
    charge_control = data['charge_control'].dropna()
    pt_control = data['pt_control'].dropna()
    mass_control = data['mass_control'].dropna()
    
    print(f"\nDati validi:")
    print(f"  charge_control: {len(charge_control)} valori")
    print(f"  pt_control: {len(pt_control)} valori")
    print(f"  mass_control: {len(mass_control)} valori")
    
    # Istogrammi
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Charge control (differenza)
    axes[0].hist(charge_control, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Q_jet - Q_costituenti')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Controllo Carica')
    axes[0].axvline(0, color='red', linestyle='--')
    axes[0].grid(True, alpha=0.3)
    
    # PT control (rapporto -1)
    axes[1].hist(pt_control, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('(PT_jet / PT_costituenti) - 1')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Controllo PT')
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].grid(True, alpha=0.3)
    
    # Mass control (rapporto -1)
    axes[2].hist(mass_control, bins=50, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('(M_jet / M_costituenti) - 1')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Controllo Massa')
    axes[2].axvline(0, color='red', linestyle='--')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistiche
    print("\nSTATISTICHE:")
    print(f"  Charge control: mean={charge_control.mean():.6f}, std={charge_control.std():.6f}")
    print(f"  PT control: mean={pt_control.mean():.6f}, std={pt_control.std():.6f}")
    print(f"  Mass control: mean={mass_control.mean():.6f}, std={mass_control.std():.6f}")
    
    # Interpretazione
    print("\nINTERPRETAZIONE:")
    print("  Se pt_control = 0 -> PT_jet = PT_costituenti")
    print("  Se pt_control > 0 -> PT_jet > PT_costituenti (jet ha più PT della somma vettoriale)")
    print("  Se pt_control < 0 -> PT_jet < PT_costituenti (anomalo, potrebbe indicare problemi)")
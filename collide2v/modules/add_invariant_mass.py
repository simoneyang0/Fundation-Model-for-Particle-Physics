#!/usr/bin/env python3
"""
Modulo semplice per aggiungere la massa invariante a un Dask DataFrame.
"""

import dask.dataframe as dd
import numpy as np


def add_invariant_mass(ddf, use_leading=True, n_leading=2):
    """
    Aggiunge una colonna con la massa invariante calcolata dai jet.
    
    Parameters:
    -----------
    ddf : dask.dataframe.DataFrame
        Dask DataFrame con le colonne dei jet (PT, Eta, Phi, Mass)
    use_leading : bool
        Se True, usa solo i primi n_leading jet con PT più alto
        Se False, usa tutti i jet dell'evento
    n_leading : int
        Numero di leading jet da usare (solo se use_leading=True)
    
    Returns:
    --------
    dask.dataframe.DataFrame
        Nuovo Dask DataFrame con colonna 'invariant_mass' aggiunta
    """
    
    def compute_mass(row):
        """Calcola massa invariante per un singolo evento"""
        
        # Estrai gli array
        pt = row['FullReco_JetAK4_PT']
        eta = row['FullReco_JetAK4_Eta']
        phi = row['FullReco_JetAK4_Phi']
        mass = row['FullReco_JetAK4_Mass']
        
        # Controlla se ci sono abbastanza jet
        if pt is None or len(pt) < 2:
            return np.nan
        
        # Seleziona i jet da usare
        if use_leading:
            # Ordina per PT decrescente e prendi i primi n_leading
            sorted_idx = np.argsort(pt)[::-1]
            n = min(n_leading, len(sorted_idx))
            selected_idx = sorted_idx[:n]
        else:
            # Usa tutti i jet
            selected_idx = range(len(pt))
        
        # Se non ci sono almeno 2 jet, restituisci NaN
        if len(selected_idx) < 2:
            return np.nan
        
        # Prendi i primi 2 jet selezionati
        i1, i2 = selected_idx[0], selected_idx[1]
        
        # Calcola 4-vettori
        # Jet 1
        px1 = pt[i1] * np.cos(phi[i1])
        py1 = pt[i1] * np.sin(phi[i1])
        pz1 = pt[i1] * np.sinh(eta[i1])
        e1 = np.sqrt(pt[i1]**2 + pz1**2 + mass[i1]**2)
        
        # Jet 2
        px2 = pt[i2] * np.cos(phi[i2])
        py2 = pt[i2] * np.sin(phi[i2])
        pz2 = pt[i2] * np.sinh(eta[i2])
        e2 = np.sqrt(pt[i2]**2 + pz2**2 + mass[i2]**2)
        
        # Somma
        total_px = px1 + px2
        total_py = py1 + py2
        total_pz = pz1 + pz2
        total_e = e1 + e2
        
        # Massa invariante
        mass_sq = total_e**2 - total_px**2 - total_py**2 - total_pz**2
        
        if mass_sq < 0:
            mass_sq = 0
        
        return np.sqrt(mass_sq)
    
    # Applica la funzione a ogni riga
    mass_series = ddf.apply(compute_mass, axis=1, meta=('invariant_mass', 'float64'))
    
    # Restituisci nuovo DDF con colonna aggiunta
    return ddf.assign(invariant_mass=mass_series)

# Esempio di utilizzo
if __name__ == "__main__":
    # Carica i dati
    file_path = "data/tuo_file.parquet"
    ddf = dd.read_parquet(file_path, columns=[
        "FullReco_JetAK4_PT",
        "FullReco_JetAK4_Eta", 
        "FullReco_JetAK4_Phi",
        "FullReco_JetAK4_Mass"
    ])
    
    # Aggiungi massa invariante usando i leading jet
    ddf_with_mass = add_invariant_mass(ddf, use_leading=True, n_leading=2)
    
    # Oppure usando tutti i jet
    # ddf_with_mass = add_invariant_mass(ddf, use_leading=False)
    
    # Ora puoi fare l'istogramma
    masses = ddf_with_mass['invariant_mass'].compute()
    masses = masses[~np.isnan(masses)]
    
    import matplotlib.pyplot as plt
    plt.hist(masses, bins=50, range=(0, 200))
    plt.xlabel('Massa Invariante (GeV)')
    plt.ylabel('Count')
    plt.show()
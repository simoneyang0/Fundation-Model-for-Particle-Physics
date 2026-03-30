#!/usr/bin/env python3
"""
Modulo per la creazione di istogrammi da file Parquet con Dask.
Gestisce automaticamente colonne annidate e overflow numerici.
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
import dask.dataframe as dd
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
import argparse
import os
import json
from datetime import datetime


class SafeHistogramMaker:
    """
    Classe per creare istogrammi in modo sicuro da Dask DataFrames,
    gestendo colonne annidate, NaN, infiniti e overflow numerici.
    """
    
    def __init__(self, ddf: dd.DataFrame, verbose: bool = True, 
                 cache_flat_data: bool = True):
        """
        Inizializza il SafeHistogramMaker.
        
        Parameters:
        -----------
        ddf : dask.dataframe.DataFrame
            Dask DataFrame contenente i dati
        verbose : bool
            Se True, stampa informazioni di progresso
        cache_flat_data : bool
            Se True, mantiene in cache i dati appiattiti
        """
        self.ddf = ddf
        self.verbose = verbose
        self.cache_flat_data = cache_flat_data
        self.flat_data = {} if cache_flat_data else None
        self._column_types = {}  # Cache per tipi di colonne
        
    def clean_array(self, data: np.ndarray, remove_outliers: bool = True,
                   outlier_percentile: float = 99.9) -> np.ndarray:
        """
        Pulisce l'array rimuovendo valori problematici.
        
        Parameters:
        -----------
        data : np.ndarray
            Array da pulire
        remove_outliers : bool
            Se True, rimuove outliers estremi
        outlier_percentile : float
            Percentile per la rimozione outliers (default: 99.9)
        
        Returns:
        --------
        np.ndarray : Array pulito
        """
        # Converti in float64 per sicurezza
        data = np.asarray(data, dtype=np.float64)
        
        # Rimuovi NaN e infiniti
        data = data[~np.isnan(data)]
        data = data[np.isfinite(data)]
        
        if len(data) == 0:
            return data
        
        # Rimuovi outliers estremi
        if remove_outliers:
            p_low = np.percentile(data, 100 - outlier_percentile)
            p_high = np.percentile(data, outlier_percentile)
            
            if np.isfinite(p_low) and np.isfinite(p_high):
                data = data[(data >= p_low) & (data <= p_high)]
        
        return data
    
    def safe_mean(self, data: np.ndarray) -> float:
        """Calcola la media in modo sicuro."""
        if len(data) == 0:
            return np.nan
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            try:
                return np.mean(data, dtype=np.float64)
            except:
                return np.median(data)
    
    def safe_std(self, data: np.ndarray) -> float:
        """Calcola la deviazione standard in modo sicuro."""
        if len(data) == 0:
            return np.nan
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            try:
                return np.std(data, dtype=np.float64)
            except:
                q75, q25 = np.percentile(data, [75, 25])
                return (q75 - q25) / 1.349
    
    def is_nested_column(self, column: str) -> bool:
        """
        Verifica se una colonna contiene dati annidati.
        """
        if column in self._column_types:
            return self._column_types[column]
        
        try:
            sample = self.ddf[column].head(1).iloc[0]
            is_nested = isinstance(sample, (list, np.ndarray))
            self._column_types[column] = is_nested
            return is_nested
        except Exception as e:
            if self.verbose:
                print(f"  Warning: impossibile determinare tipo per {column}: {e}")
            return False
    
    def flatten_column(self, column: str, max_elements: Optional[int] = None) -> np.ndarray:
        """
        Appiattisce una colonna annidata in modo sicuro.
        
        Parameters:
        -----------
        column : str
            Nome della colonna
        max_elements : int, optional
            Numero massimo di elementi da estrarre (per limitare memoria)
        
        Returns:
        --------
        np.ndarray : Array appiattito
        """
        # Controlla cache
        if self.cache_flat_data and column in self.flat_data:
            return self.flat_data[column]
        
        if self.verbose:
            print(f"Appiattimento colonna: {column}")
        
        all_values = []
        total_elements = 0
        
        # Processa partizione per partizione
        for partition_idx, partition in enumerate(self.ddf[column].to_delayed()):
            try:
                partition_values = partition.compute()
                
                for val in partition_values:
                    if val is None:
                        continue
                    
                    if isinstance(val, (list, np.ndarray)):
                        # Converti in array e filtra valori problematici
                        arr = np.asarray(val, dtype=np.float64)
                        arr = arr[np.isfinite(arr)]
                        if len(arr) > 0:
                            all_values.extend(arr.tolist())
                            total_elements += len(arr)
                    else:
                        # Converti in float e controlla
                        try:
                            fval = float(val)
                            if np.isfinite(fval):
                                all_values.append(fval)
                                total_elements += 1
                        except (ValueError, TypeError):
                            continue
                    
                    # Controlla limite elementi
                    if max_elements and total_elements >= max_elements:
                        if self.verbose:
                            print(f"  Raggiunto limite di {max_elements} elementi")
                        break
                
                if max_elements and total_elements >= max_elements:
                    break
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Attenzione: errore nella partizione {partition_idx}: {e}")
                continue
        
        if len(all_values) == 0:
            result = np.array([])
        else:
            result = np.array(all_values, dtype=np.float64)
        
        # Salva in cache
        if self.cache_flat_data:
            self.flat_data[column] = result
        
        if self.verbose:
            print(f"  Trovati {len(result):,} valori validi")
        
        return result
    
    def get_column_data(self, column: str, max_elements: Optional[int] = None) -> np.ndarray:
        """
        Ottiene i dati di una colonna in modo sicuro.
        
        Parameters:
        -----------
        column : str
            Nome della colonna
        max_elements : int, optional
            Numero massimo di elementi da estrarre
        
        Returns:
        --------
        np.ndarray : Array con i dati puliti
        """
        if self.is_nested_column(column):
            data = self.flatten_column(column, max_elements)
        else:
            # Per colonne non annidate
            try:
                data = self.ddf[column].compute().to_numpy()
                data = np.asarray(data, dtype=np.float64)
            except Exception as e:
                if self.verbose:
                    print(f"Errore nel caricamento di {column}: {e}")
                return np.array([])
        
        # Pulisci sempre i dati
        return self.clean_array(data)
    
    def get_statistics(self, column: str) -> Dict[str, float]:
        """
        Calcola statistiche per una colonna.
        
        Returns:
        --------
        dict : Dizionario con le statistiche
        """
        data = self.get_column_data(column)
        
        if len(data) == 0:
            return {'error': 'No valid data'}
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            stats = {
                'count': len(data),
                'mean': self.safe_mean(data),
                'std': self.safe_std(data),
                'min': np.min(data),
                'max': np.max(data),
                'median': np.percentile(data, 50),
                'p1': np.percentile(data, 1),
                'p5': np.percentile(data, 5),
                'p95': np.percentile(data, 95),
                'p99': np.percentile(data, 99)
            }
            
        return stats
    
    def make_histogram(self, column: str, bins: int = 100,
                       range: Optional[Tuple[float, float]] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       show_stats: bool = True,
                       save_path: Optional[str] = None,
                       title: Optional[str] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Crea un istogramma per una colonna.
        
        Parameters:
        -----------
        column : str
            Nome della colonna
        bins : int
            Numero di bin
        range : tuple, optional
            Range dell'istogramma (min, max)
        figsize : tuple
            Dimensione della figura
        show_stats : bool
            Se True, mostra statistiche sul plot
        save_path : str, optional
            Percorso per salvare l'immagine
        title : str, optional
            Titolo personalizzato
        
        Returns:
        --------
        tuple : (hist, bin_edges) o None se errore
        """
        # Sopprimi i warning durante il calcolo
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            
            # Ottieni i dati
            data = self.get_column_data(column)
            
            if len(data) == 0:
                print(f"Attenzione: Nessun dato valido per {column}")
                return None
            
            if self.verbose:
                print(f"\n--- {column} ---")
                print(f"  Elementi validi: {len(data):,}")
                print(f"  Range dati: [{data.min():.3f}, {data.max():.3f}]")
            
            # Determina range automatico se non specificato
            if range is None:
                p1 = np.percentile(data, 0.5)
                p99 = np.percentile(data, 99.5)
                range = (p1, p99)
            
            # Crea il plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Calcola istogramma
            try:
                hist, bin_edges, _ = ax.hist(data, bins=bins, range=range,
                                            edgecolor='black', alpha=0.7,
                                            color='steelblue')
            except Exception as e:
                print(f"Errore nel plot di {column}: {e}")
                # Tentativo con bins ridotti
                hist, bin_edges, _ = ax.hist(data, bins=min(bins, 50), range=range,
                                            edgecolor='black', alpha=0.7)
            
            # Imposta etichette
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
            
            # Titolo
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f'Distribuzione di {column}')
            
            ax.grid(True, alpha=0.3)
            
            # Aggiungi statistiche
            if show_stats:
                mean_val = self.safe_mean(data)
                std_val = self.safe_std(data)
                median_val = np.percentile(data, 50)
                
                stats_text = (f'n = {len(data):,}\n'
                            f'mean = {mean_val:.3f}\n'
                            f'std = {std_val:.3f}\n'
                            f'median = {median_val:.3f}')
                
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Salva se richiesto
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                if self.verbose:
                    print(f"  Salvato: {save_path}")
            
            plt.show()
            
            return hist, bin_edges
    
    def make_all_histograms(self, bins: int = 50,
                           range_dict: Optional[Dict[str, Tuple[float, float]]] = None,
                           figsize: Tuple[int, int] = (10, 6),
                           output_dir: Optional[str] = None):
        """
        Crea istogrammi per tutte le colonne.
        
        Parameters:
        -----------
        bins : int
            Numero di bin
        range_dict : dict, optional
            Dizionario con range personalizzati per colonna
        figsize : tuple
            Dimensione della figura
        output_dir : str, optional
            Directory per salvare le immagini
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for col in self.ddf.columns:
            col_range = range_dict.get(col) if range_dict else None
            save_path = None
            if output_dir:
                safe_name = col.replace('/', '_').replace(' ', '_')
                save_path = os.path.join(output_dir, f"{safe_name}.png")
            
            self.make_histogram(col, bins=bins, range=col_range,
                              figsize=figsize, save_path=save_path)
    
    def print_summary(self):
        """Stampa un riassunto delle statistiche per tutte le colonne."""
        print("\n" + "="*80)
        print("RIASSUNTO STATISTICHE")
        print("="*80)
        
        summary_data = []
        
        for col in self.ddf.columns:
            stats = self.get_statistics(col)
            
            if 'error' in stats:
                print(f"\n{col}:")
                print("  Nessun dato valido")
                continue
            
            print(f"\n{col}:")
            print(f"  Elementi validi: {stats['count']:,}")
            print(f"  Media: {stats['mean']:.3f}")
            print(f"  Std: {stats['std']:.3f}")
            print(f"  Min: {stats['min']:.3f}")
            print(f"  Max: {stats['max']:.3f}")
            print(f"  Mediana: {stats['median']:.3f}")
            print(f"  Percentili: 1%={stats['p1']:.3f}, 99%={stats['p99']:.3f}")
            
            summary_data.append({
                'column': col,
                'count': stats['count'],
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'median': stats['median'],
                'p1': stats['p1'],
                'p99': stats['p99']
            })
        
        return pd.DataFrame(summary_data)
    
    def save_statistics(self, output_file: str):
        """
        Salva le statistiche in un file CSV.
        
        Parameters:
        -----------
        output_file : str
            Percorso del file di output
        """
        df_stats = self.print_summary()
        if df_stats is not None and len(df_stats) > 0:
            df_stats.to_csv(output_file, index=False)
            if self.verbose:
                print(f"\nStatistiche salvate in: {output_file}")


def create_histogram_maker_from_file(file_path: str, 
                                      columns: Optional[List[str]] = None,
                                      verbose: bool = True) -> SafeHistogramMaker:
    """
    Factory function per creare un SafeHistogramMaker da un file Parquet.
    
    Parameters:
    -----------
    file_path : str
        Percorso del file Parquet
    columns : list, optional
        Lista di colonne da caricare (None = tutte)
    verbose : bool
        Se True, stampa informazioni
    
    Returns:
    --------
    SafeHistogramMaker : Istanza configurata
    """
    if verbose:
        print(f"Caricamento file: {file_path}")
        if columns:
            print(f"Colonne selezionate: {columns}")
    
    ddf = dd.read_parquet(file_path, columns=columns)
    
    if verbose:
        print(f"Dataset caricato: {len(ddf.columns)} colonne, {ddf.npartitions} partizioni")
    
    return SafeHistogramMaker(ddf, verbose=verbose)


def main():
    """Funzione principale per uso da linea di comando."""
    parser = argparse.ArgumentParser(
        description='Crea istogrammi da file Parquet con Dask'
    )
    parser.add_argument('file_path', type=str, help='Percorso del file Parquet')
    parser.add_argument('--columns', '-c', nargs='+', 
                       help='Colonne da processare (default: tutte)')
    parser.add_argument('--bins', '-b', type=int, default=50,
                       help='Numero di bin per istogramma (default: 50)')
    parser.add_argument('--output-dir', '-o', type=str, default='histograms',
                       help='Directory per salvare gli istogrammi (default: histograms)')
    parser.add_argument('--stats-file', '-s', type=str, default='statistics.csv',
                       help='File per salvare le statistiche (default: statistics.csv)')
    parser.add_argument('--no-show', action='store_true',
                       help='Non mostrare i plot (salva solo su file)')
    parser.add_argument('--max-elements', type=int, default=None,
                       help='Numero massimo di elementi per colonna (per limitare memoria)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mostra informazioni dettagliate')
    
    args = parser.parse_args()
    
    # Crea l'analizzatore
    maker = create_histogram_maker_from_file(
        args.file_path,
        columns=args.columns,
        verbose=args.verbose
    )
    
    # Crea directory di output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Crea istogrammi
    print("\n" + "="*80)
    print("CREAZIONE ISTOGRAMMI")
    print("="*80)
    
    maker.make_all_histograms(
        bins=args.bins,
        output_dir=args.output_dir if not args.no_show else args.output_dir,
        figsize=(10, 6)
    )
    
    # Salva statistiche
    maker.save_statistics(args.stats_file)
    
    print("\n" + "="*80)
    print(f"Completato! Istogrammi salvati in: {args.output_dir}")
    print(f"Statistiche salvate in: {args.stats_file}")
    print("="*80)


if __name__ == "__main__":
    main()
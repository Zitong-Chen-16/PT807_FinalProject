from rdkit import Chem
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def is_chiral_from_smiles(smiles):
    """Detect chirality from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    return len(chiral_centers) > 0

def process_smiles_file(df, smiles_column='smiles'):
    """Process a CSV file with SMILES strings."""
    chiral_mols = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        smiles = row[smiles_column]
        if is_chiral_from_smiles(smiles):
            chiral_mols.append((idx, smiles))
    
    return chiral_mols

def save_chiral_molecules(chiral_mols, output_file='chiral_molecules.csv'):
    df = pd.DataFrame(chiral_mols, columns=['Index', 'SMILES'])
    df.to_csv(output_file, index=False)
    print(f"Saved {len(chiral_mols)} chiral molecules to {output_file}")

def main():
    data_dir = Path('../data/geom/')

    smiles_list_file = Path('geom_drugs_smiles.txt')
    
    df_smile = pd.read_csv(data_dir / smiles_list_file, sep=' ', header=None)
    df_smile.columns = ['smiles']

    chiral_molecules = process_smiles_file(df_smile, smiles_column='smiles')
    output_file = 'chiral_molecules.csv'
    save_chiral_molecules(chiral_molecules, output_file=data_dir / output_file)
    print(f'Percentage of molecules that are chiral: {len(chiral_molecules)/df_smile.shape[0]*100}%')

if __name__ == "__main__":
    main()
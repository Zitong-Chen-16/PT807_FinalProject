import numpy as np
from pathlib import Path 

def reflect_dataset(input_file, output_file, axis=0):
    """
    Reflects molecules in the dataset along a specified axis.

    Args:
        input_file (str): Path to the original conformers (.npy file).
        output_file (str): Path to save the reflected conformers.
        axis (int): 0 = x-axis, 1 = y-axis, 2 = z-axis.
    """
    print(f"Loading data from {input_file}...")
    dataset = np.load(input_file)

    print(f"Original dataset shape: {dataset.shape}")
    reflected_dataset = dataset.copy()
    
    # axis+2 because dataset format is (mol_id, atomic_number, x, y, z)
    reflected_dataset[:, axis + 2] *= -1.0

    print(f"Saving reflected dataset to {output_file}...")
    np.save(output_file, reflected_dataset)
    print("Done.")

# Example usage:
if __name__ == "__main__":
    data_dir = Path('../data/geom/')

    input_filename = Path("geom_drugs_chiral_30.npy")
    output_filename = Path("geom_drugs_chiral_30_mirrored.npy")

    reflect_dataset(
        data_dir/input_filename, 
        data_dir/output_filename, 
        axis=0
    )  # reflect along x-axis

import numpy as np
from scipy.sparse.linalg import eigsh

from openfermion import (MolecularData, get_fermion_operator, 
                         jordan_wigner, get_sparse_operator)
from openfermionpyscf import run_pyscf

# ====================
# Pyscf script for H2 molecule. 
# ====================


# 0.5 to 5.0 Å with step 0.1
bond_lengths = np.arange(0.5, 5.1, 0.1) 
energies = []
ham_matrices = []


print("Starting H2 molecule calculations...")
print(f"Bond lengths: {len(bond_lengths)} points from {bond_lengths[0]} to {bond_lengths[-1]} Å")

for idx, d in enumerate(bond_lengths):
    # Define H2 geometry - two hydrogen atoms along z-axis
    geometry = [('H', (0, 0, 0)), ('H', (0, 0, d))]
    
    print(f"\n{'='*50}")
    print(f"Calculating for bond length: {d:.1f} Å (Point {idx+1}/{len(bond_lengths)})")
    print(f"{'='*50}")
    
    try:
        # Create MolecularData object for H2
        molecule = MolecularData(
            geometry=geometry,
            basis='sto-3g',
            multiplicity=1,  # Singlet state
            charge=0,
            description=f"H2_d={d:.2f}"
        )
        
        # Run PySCF calculation
        print("Running PySCF calculation...")
        molecule = run_pyscf(molecule, run_scf=1) 
        
        # Get molecular Hamiltonian in second quantization
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        
        # Convert to FermionOperator
        print("Performing Jordan-Wigner transformation...")        
        fermion_ham = get_fermion_operator(molecular_hamiltonian)
        
        # Get qubit Hamiltonian via Jordan-Wigner
        qubit_ham = jordan_wigner(fermion_ham)
        
        # Get sparse matrix representation
        sparse_mat = get_sparse_operator(qubit_ham)
        ham_matrix = sparse_mat.toarray()
        
        # Diagonalize to get ground state energy
        print("Diagonalizing Hamiltonian...")
        energy, _ = eigsh(sparse_mat, k=1, which='SA')
        ground_state_energy = energy[0]
        energies.append(ground_state_energy)
        ham_matrices.append(ham_matrix)
        
        # Store qubit basis information
        n_qubits = int(np.log2(ham_matrix.shape[0]))
        
        print(f"✓ Calculation successful!")
        print(f"  Ground State Energy: {ground_state_energy:.6f} Hartree")
        print(f"  Matrix shape: {ham_matrix.shape}")
        

    except Exception as e:
        print(f"✗ Error at bond length {d:.1f}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Save results
print("\nSaving results to files...")
np.save('h2_matrix.npy', np.array(ham_matrices, dtype=object))

print("\n" + "="*50)
print(f"Calculation complete!")
print(f"Number of geometries calculated: {len(energies)}/{len(bond_lengths)}")
print(f"Energy range: {min(energies):.6f} to {max(energies):.6f} Hartree")
print(f"Minimum energy at bond length: {bond_lengths[np.argmin(energies)]:.2f} Å")
print("="*50)


# Create a summary file with energies vs bond lengths
with open('h2_energies_summary.txt', 'w') as f:
    f.write("H2 Molecule - Bond Length vs Energy\n")
    f.write("="*40 + "\n")
    f.write("Bond Length (Å)   Energy (Hartree)\n")
    f.write("-"*40 + "\n")
    for i, d in enumerate(bond_lengths):
        if i < len(energies):
            f.write(f"{d:10.2f}        {energies[i]:12.8f}\n")

print(f"Energy summary saved to 'h2_energies.txt'")







# hybrid_experiment.py
from mpi4py import MPI
import numpy as np 
from research_paper_experiments import ResearchPaperExperiments

if __name__ == "__main__":
    experiments = ResearchPaperExperiments()
    
    if experiments.rank == 0:
        print("Running Hybrid MPI + OMP4Py Jacobi Method")
        print(f"Using {experiments.size} MPI processes")
    
    # Run hybrid Jacobi
    result = experiments.jacobi_hybrid(400, 100)
    
    if experiments.rank == 0:
        print("Hybrid computation completed!")
        print(f"Final solution vector norm: {np.linalg.norm(result):.6f}")
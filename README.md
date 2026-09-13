# percolation_resolved_diffusion
Code and data for percolation-resolved pore characterization and methane diffusion analysis in disordered porous materials.

## Directory structure

- `pseudomaterials/` — Example disordered porous structures used in the simulations.
- `blocking_sphere_generation/` — Generates blocking spheres for accessible pore regions.
- `mc_example/` — Example Monte Carlo (MC) setup using `InvertBlockPockets` to initialize methane within the percolating pore network.
- `md_example/` — Example molecular dynamics simulation for methane diffusion.
- `property_calculation/` — Scripts for calculating structural and topology-related properties using Mofography; some functions were revised and shown in "winding_percolation.py". 

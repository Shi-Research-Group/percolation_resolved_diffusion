# Pore-structure properties of a hyper-crosslinked polymer


### This Jupyter notebook demonstrates how to compute pore-structure properties of a hyper-crosslinked polymer (HCP) with mofography. Using the 75X_0DVB_4 structure as an example, it builds a periodic pore graph and computes the largest cavity diameter, pore limiting diameter, N₂ accessible surface area, percolating pore volume, average pore and window sizes, pore connectivity, percolating backbone fraction, and fractional free volume.

The structure is taken from https://doi.org/10.1021/acs.jctc.2c00798

For detailed definitions of these properties, see our paper about
methane diffusion: Zheng, C.; Gopalan, A.; Shi, K. Revisiting Percolation and Molecular Diffusion in Disordered Pore Networks (https://doi.org/10.26434/chemrxiv.15011012/v1).

Requirements

- python==3.11.9
- mofography== 2026.10.9.1
- numpy==1.23.5
- pandas==2.2.2
- ase==3.22.1
- networkx==3.2.1

``` python
import os
os.environ["RAYON_NUM_THREADS"] = "8"

import pandas as pd
import mofography as mgr
from ase.io import read
import numpy as np
```

``` python
# TraPPE sigma (Å) of the united-atom sites in 75X_0DVB_4.cif
trappe_dia = pd.Series(
        {'C1': 4.65,
         'C2': 3.95,
         'C3': 3.88,
         'C4': 3.695,
         'C6': 4.65,
         'C7': 3.95}
        )
```

``` python
spacing = 0.5
mask_thickness = 0.0
h = 0.1
threshold_abs = 0.3
min_region_size = 0.00005
preserve_threshold = 103.0
smooth_threshold = 106.0
max_smooth_radius = 2.0
extra_padding = 2
min_distance = 1.0
minimum_window_separation = 1.5

# probe radii (Å)
probe_radius = 1.865      # CH4 (TraPPE sigma / 2), for percolation analysis
n2_probe_radius = 1.82    # N2, for accessible surface area
```

``` python
def add_volume_to_graph(graph, mat_atoms):

    vol = mat_atoms.get_volume()
    vol_L=vol/(1E27)
    for i in graph.nodes():
        graph.nodes[i]['cell_volume']=vol_L
    return graph

def add_pixel_ratio_to_graph(graph, regions_pbc):

    n=regions_pbc.size
    for i in graph.nodes():
        pixel_counts = np.count_nonzero(regions_pbc == i)
        graph.nodes[i]['pixel_counts']=pixel_counts
        graph.nodes[i]['total_pixel_amounts']=n
        graph.nodes[i]['pixel_ratio']=pixel_counts/n
    return graph
```

``` python
def add_connectivities_to_nodes(
    graph,
    probe_radius,
    pore_radius_attr="pocket_radius",
    pixel_count_attr="pixel_counts",
    edge_radius_attr="window_radius",
):

    for node in graph.nodes():

        edges = {
            (u, v, key): data
            for u, v, key, data in (
                list(graph.in_edges(node, keys=True, data=True))
                + list(graph.out_edges(node, keys=True, data=True))
            )
        }

        connectivity = len(edges)

        if connectivity == 0:
            continue

        radii = np.array([
            float(data[edge_radius_attr])
            for data in edges.values()
        ])

        pore_radius = float(graph.nodes[node][pore_radius_attr])
        pixel_count = float(graph.nodes[node][pixel_count_attr])

        avg_radius = float(np.mean(radii))
        min_radius = float(np.min(radii))

        clearances = radii - probe_radius
        avg_clearance = float(np.mean(clearances))

        graph.nodes[node]["connectivity"] = connectivity
        graph.nodes[node]["pixel_count_times_connectivity"] = pixel_count * connectivity
        graph.nodes[node]["avg_window_radius"] = avg_radius
        graph.nodes[node]["avg_window_pore_ratio"] = avg_radius / pore_radius
        graph.nodes[node]["avg_radius_times_connectivity"] = connectivity * avg_radius
        graph.nodes[node]["min_window_radius"] = min_radius
        graph.nodes[node]["min_radius_times_connectivity"] = connectivity * min_radius
        graph.nodes[node]["avg_window_clearance"] = avg_clearance
        graph.nodes[node]["clearance_times_connectivity"] = connectivity * avg_clearance

    return graph
```

``` python
mat_atoms = read('../mofography/examples/TestCIFs/75X_0DVB_4.cif',store_tags=True)
site_labels = mat_atoms.info['_atom_site_label']
radii = trappe_dia[site_labels].values/ 2.0
dgrid = mgr.dgrid_from_atoms_rust(mat_atoms, spacing=spacing, radii=radii)
```

``` python
results = mgr.do_periodic_segmentation_rust(
                dgrid,
                mat_atoms,
                mask_thickness=mask_thickness,
                h=h,
                spacing=spacing,
                return_dgrid=False,
                threshold_abs=threshold_abs,
                min_distance=min_distance,
                return_padded_regions=True,
                preserve_threshold=preserve_threshold,
                smooth_threshold=smooth_threshold,
                max_smooth_radius=max_smooth_radius,
                extra_padding=extra_padding,
                compute_connections=True,
                min_region_size=min_region_size,
                minimum_window_separation=minimum_window_separation,
                watershed_backend="rust_det",
            )


regions = results["regions"]
maxima_info = results["maxima_info"]
connections = results["connections"]
regions_pbc = results["regions_pbc"]
regions_1d = regions.flatten()
```

**Pore graph.** Each node is a pore in the unit cell, and each edge is
one window between two pores. Only windows between pores that own part
of the original unit cell are kept, so every window is counted once.
Windows that cross the cell boundary are kept as edges with a non-zero
`relative_offset`.

``` python
#construct pore graph
graph=(mgr.make_window_resolved_graph_from_results(results))

graph = add_volume_to_graph(graph,mat_atoms)

graph = add_pixel_ratio_to_graph(graph,regions_pbc)

base_labels = (mgr.build_base_region_labels_from_results(results))
```

``` python
# Largest cavity diameter (Å): diameter of the largest sphere that fits in the pore space, regardless of percolation
lcd = dgrid.max() * 2
```

``` python
# Pore limiting diameter (Å): diameter of the largest spherical probe that can pass through the periodic percolating pore network
plr = mgr.find_pore_limiting_radius_winding(results,graph=graph,base_region_labels=base_labels,include_dangling=True)
pld = plr * 2
```

``` python
# N2-accessible surface area (m²/g): surface area accessible to an N2-sized probe in the percolating pore network
np.random.seed(0)
n2_asa = (mgr.compute_surface_area_mc_winding(
                    results,
                    mat_atoms,
                    probe_radius=n2_probe_radius,
                    n_samples=50000,
                    shell_thickness=1,
                    return_details=False,
                    window_fp=False,
                    winding_kwargs=None,
                )["accessible_surface_area"]
            )
```

``` python
# Methane-accessible percolating pore network
# before pruning: the whole percolating network, including self-loops and dangling branches
# after pruning: only the shortest percolating backbone pathways
percolation_pore_graph_before_pruning = (mgr.extract_percolating_pore_graph(graph,threshold=probe_radius,expand_components=True,include_dangling=True))
percolation_pore_graph_before_pruning = (add_connectivities_to_nodes(percolation_pore_graph_before_pruning,probe_radius=probe_radius))

percolation_pore_graph_after_pruning = (mgr.extract_percolating_pore_graph(graph,threshold=probe_radius,expand_components=False,include_dangling=False))
percolation_pore_graph_after_pruning = (add_connectivities_to_nodes(percolation_pore_graph_after_pruning,probe_radius=probe_radius))
```

``` python
# Percolating pore volume fraction: fraction of the cell volume occupied by the methane-accessible percolating pore network
percolating_pore_volume_fraction=sum(percolation_pore_graph_before_pruning.nodes[node]["pixel_ratio"]for node in percolation_pore_graph_before_pruning.nodes())
```

``` python
# Percolating gravimetric pore volume (cm³/g): percolating pore volume divided by the framework mass
rho=0.82 # bulk density (g/cm³): framework mass / cell volume, from 10.1021/acs.jctc.2c00798
percolating_gravimetric_pore_volume=percolating_pore_volume_fraction/rho
```

``` python
# Average percolating pore radius (Å): mean radius of the pores in the percolating network
average_pore_radius = np.mean([percolation_pore_graph_before_pruning.nodes[n]["pocket_radius"] for n in percolation_pore_graph_before_pruning.nodes()])

# Average accessible window radius (Å): for each pore, the mean radius of the accessible windows connected to it,
# averaged over all pores in the percolating network
average_accessible_window_radius = np.mean([percolation_pore_graph_before_pruning.nodes[n]["avg_window_radius"] for n in percolation_pore_graph_before_pruning.nodes()])

# Average window-to-pore radius ratio: mean of (average radius of the windows connected to a pore / radius of that pore)
w_to_p = np.mean([percolation_pore_graph_before_pruning.nodes[n]["avg_window_radius"] / percolation_pore_graph_before_pruning.nodes[n]["pocket_radius"] for n in percolation_pore_graph_before_pruning.nodes()])
```

``` python
# Average pore connectivity: average number of accessible windows connected to each pore
average_connectivity = np.mean([percolation_pore_graph_before_pruning.nodes[n]["connectivity"]for n in percolation_pore_graph_before_pruning.nodes()])

# Average-window-radius-weighted connectivity = sum_i(c_i * r_w,i) / sum_i(r_w,i),
# where c_i is the connectivity of pore i and r_w,i is the mean radius of its accessible windows
average_window_size_weighted_connectivity = (sum(percolation_pore_graph_before_pruning.nodes[n]["avg_radius_times_connectivity"]for n in percolation_pore_graph_before_pruning.nodes())/sum(percolation_pore_graph_before_pruning.nodes[n]["avg_window_radius"]for n in percolation_pore_graph_before_pruning.nodes()))
```

``` python
# Percolating backbone fraction = n_b / n_p: grid points in the shortest percolating backbone
# divided by grid points in the whole percolating network
percolating_pore_volume_fraction_backbone=sum(percolation_pore_graph_after_pruning.nodes[node]["pixel_ratio"]for node in percolation_pore_graph_after_pruning.nodes())
fB=percolating_pore_volume_fraction_backbone/percolating_pore_volume_fraction
```

``` python
# Fractional free volume = 1 - 1.3 * V_vdW / V_sp: fraction of the material volume not occupied by the framework
Vvdw=np.sum(regions_1d==0)
Vsp=len(regions_1d)
ffv=1-1.3*(Vvdw/Vsp)
```

``` python
summary = pd.DataFrame(
    [
        ["Largest cavity diameter (LCD)", lcd, "Å"],
        ["Pore limiting diameter (PLD)", pld, "Å"],
        ["N₂-accessible surface area", n2_asa, "m²/g"],
        ["Percolating pore volume fraction", percolating_pore_volume_fraction, "–"],
        ["Percolating gravimetric pore volume", percolating_gravimetric_pore_volume, "cm³/g"],
        ["Average percolating pore radius", average_pore_radius, "Å"],
        ["Average accessible window radius", average_accessible_window_radius, "Å"],
        ["Average window-to-pore radius ratio", w_to_p, "–"],
        ["Average pore connectivity", average_connectivity, "–"],
        ["Average-window-radius-weighted connectivity", average_window_size_weighted_connectivity, "–"],
        ["Percolating backbone fraction (fB)", fB, "–"],
        ["Fractional free volume (FFV)", ffv, "–"],
    ],
    columns=["Property", "Value", "Unit"],
)
summary
```

|     | Property                                    | Value      | Unit  |
|-----|---------------------------------------------|------------|-------|
| 0   | Largest cavity diameter (LCD)               | 9.163111   | Å     |
| 1   | Pore limiting diameter (PLD)                | 4.072923   | Å     |
| 2   | N₂-accessible surface area                  | 835.987770 | m²/g  |
| 3   | Percolating pore volume fraction            | 0.257681   | –     |
| 4   | Percolating gravimetric pore volume         | 0.314245   | cm³/g |
| 5   | Average percolating pore radius             | 3.154000   | Å     |
| 6   | Average accessible window radius            | 2.521294   | Å     |
| 7   | Average window-to-pore radius ratio         | 0.815319   | –     |
| 8   | Average pore connectivity                   | 2.692308   | –     |
| 9   | Average-window-radius-weighted connectivity | 2.802774   | –     |
| 10  | Percolating backbone fraction (fB)          | 0.294555   | –     |
| 11  | Fractional free volume (FFV)                | 0.328138   | –     |

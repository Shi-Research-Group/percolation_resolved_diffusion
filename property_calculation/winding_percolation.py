"""
Periodic winding-number percolation helpers for directed pore graphs.

This module targets the deduplicated NetworkX ``MultiDiGraph`` produced by the
PBC-aware segmentation pipeline described in ``winding_percolation_gpt5.1.md``:

* Nodes correspond to unique ``maxima_pbc`` representatives and carry at least
  ``pocket_radius`` (Å) and fractional coordinates ``frac_pos``.
* Edges are stored only once with the canonical ordering ``i <= j`` and provide
  ``relative_offset`` – an integer tuple telling how much node ``j`` must be
  shifted (in box lengths) to reach the connected periodic image from ``i``.
  Self-loops with non-zero offsets represent pores directly connected to their
  own periodic images and therefore can encode valid winding loops.
* The transpose edge (``j -> i`` with ``-relative_offset``) is intentionally
  absent to match CGD/topology file conventions, so any traversal routine must
  treat edges as *undirected* while respecting the stored orientation when
  accumulating winding numbers.

The functions below keep the BFS/DFS logic entirely in graph space – no voxel
replication is required.  They accumulate integer offsets per node, detect when
back-edges close non-contractible cycles, and report per-axis percolation along
with the actual winding clusters.  When testing percolation along one axis we
implicitly allow periodic connectivity in the other two axes, ensuring that
clusters such as the large NU-1000 corner pocket (which has multiple XY images
but a single Z loop) are reported as one physical cluster instead of duplicate
self-loops.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import mofography as mgr

Offset = Tuple[int, int, int]


@dataclass(frozen=True)
class CanonicalEdgeID:
    """Reference to an edge independent of traversal direction."""

    u: int
    v: int
    key: int

    @classmethod
    def from_nodes(cls, a: int, b: int, key: int) -> "CanonicalEdgeID":
        if a <= b:
            return cls(a, b, key)
        return cls(b, a, key)


def _normalize_offset(value: Sequence[int]) -> Offset:
    """Return an integer tuple regardless of numpy scalar types."""
    return tuple(int(x) for x in value)


def _iter_neighbor_edges(
    G: nx.MultiDiGraph, node: int
) -> Iterator[Tuple[int, int, Offset, Offset, bool]]:
    """
    Yield (neighbor, key, canonical_offset, delta, forward) for edges touching ``node``.

    ``forward=True`` means ``node`` matches the stored direction (edge data is
    ``node -> neighbor``).  ``forward=False`` means we are traversing the edge
    in reverse (``neighbor -> node`` in the stored graph) and therefore the
    offset must be negated.
    """

    # Outgoing edges: orientation matches storage.
    for nbr, keyed in G.succ[node].items():
        for key, attrs in keyed.items():
            rel = _normalize_offset(attrs.get("relative_offset", (0, 0, 0)))
            yield nbr, key, rel, rel, True

    # Incoming edges: orientation reversed.
    for nbr, keyed in G.pred[node].items():
        for key, attrs in keyed.items():
            rel = _normalize_offset(attrs.get("relative_offset", (0, 0, 0)))
            if nbr == node:
                # Self-loops already handled by the outgoing iteration above.
                continue
            delta = tuple(-x for x in rel)
            yield nbr, key, rel, delta, False


def _tree_path(u: int, v: int, parent: Dict[int, Optional[int]]) -> List[int]:
    """Return the list of nodes along the unique tree path connecting u and v."""
    path_u = []
    cur = u
    while cur is not None:
        path_u.append(cur)
        cur = parent[cur]

    path_v = []
    cur = v
    while cur is not None:
        path_v.append(cur)
        cur = parent[cur]

    ancestors_u = set(path_u)
    lca = next(node for node in path_v if node in ancestors_u)

    pu = path_u[: path_u.index(lca) + 1]
    pv = path_v[: path_v.index(lca) + 1]

    return pu + pv[::-1][1:]


def percolation_winding_multidigraph(
    G: nx.MultiDiGraph,
) -> Tuple[Dict[str, bool], List[Dict[str, object]], Dict[int, Offset]]:
    """
    Evaluate percolation using winding numbers on a deduplicated MultiDiGraph.

    Returns
    -------
    percolates : dict
        Keys ``'x'``, ``'y'``, ``'z'`` map to booleans indicating whether at
        least one non-contractible loop exists along that axis.
    loops : list of dict
        Each loop dictionary contains ``winding`` (Offset), ``nodes`` (list of
        node IDs along the tree path that closes the loop), ``edge`` (canonical
        ``(u, v, key)``), ``edge_offset`` (stored offset on that edge) and
        ``delta`` (the traversal offset that triggered detection).
    acc_offsets : dict
        Maps node ID to its accumulated integer offset relative to the BFS root
        of its connected component.
    """

    percolates = {"x": False, "y": False, "z": False}
    loops: List[Dict[str, object]] = []
    acc_offsets: Dict[int, Offset] = {}
    parent: Dict[int, Optional[int]] = {}
    visited = set()

    undirected = nx.Graph(G.to_undirected(as_view=True))

    for comp_nodes in nx.connected_components(undirected):
        comp_nodes = set(comp_nodes)
        root = next(iter(comp_nodes))
        if root in visited:
            continue
        visited.add(root)
        parent[root] = None
        acc_offsets[root] = (0, 0, 0)
        queue = deque([root])

        while queue:
            u = queue.popleft()
            u_acc = np.array(acc_offsets[u], dtype=int)

            for nbr, key, canonical_offset, delta, forward in _iter_neighbor_edges(G, u):
                ced = CanonicalEdgeID.from_nodes(u, nbr, key)

                # Ensure traversal stays within the current component
                if nbr not in comp_nodes:
                    continue

                implied = tuple((u_acc + np.array(delta, dtype=int)).tolist())

                if nbr not in visited:
                    visited.add(nbr)
                    parent[nbr] = u
                    acc_offsets[nbr] = implied
                    queue.append(nbr)
                    continue

                current = acc_offsets[nbr]
                if implied == current:
                    continue

                winding_array = np.array(implied, dtype=int) - np.array(current, dtype=int)
                winding = tuple(int(x) for x in winding_array)
                if winding == (0, 0, 0):
                    continue

                cycle_nodes = _tree_path(u, nbr, parent)
                loop_info = {
                    "winding": winding,
                    "nodes": cycle_nodes,
                    "edge": (ced.u, ced.v, ced.key),
                    "edge_offset": canonical_offset,
                    "delta": delta,
                }
                loops.append(loop_info)
                if winding[0] != 0:
                    percolates["x"] = True
                if winding[1] != 0:
                    percolates["y"] = True
                if winding[2] != 0:
                    percolates["z"] = True

    unique_loops = []
    seen = set()
    for loop in loops:
        key = (loop["winding"], tuple(loop["nodes"]), loop["edge"])
        if key in seen:
            continue
        seen.add(key)
        unique_loops.append(loop)

    return percolates, unique_loops, acc_offsets


def detect_percolation_with_threshold(
    G: nx.MultiDiGraph,
    threshold: float,
    node_radius_attr: str = "pocket_radius",
    edge_radius_attr: str = "window_radius",
) -> Tuple[Dict[str, bool], List[Dict[str, object]], Dict[int, Offset], nx.MultiDiGraph]:
    """
    Convenience wrapper that filters nodes/edges below ``threshold`` before
    running :func:`percolation_winding_multidigraph`.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Pore graph following the deduplicated conventions.
    threshold : float
        Minimum radius (Å) for both nodes and edges to participate.
    node_radius_attr, edge_radius_attr : str
        Attribute names storing radii on nodes/edges respectively.
    """

    eligible_nodes = {
        n
        for n, data in G.nodes(data=True)
        if float(data.get(node_radius_attr, 0.0)) >= threshold
    }

    H = nx.MultiDiGraph()
    if not eligible_nodes:
        return {"x": False, "y": False, "z": False}, [], {}, H

    for n in eligible_nodes:
        H.add_node(n, **G.nodes[n])

    for u, v, key, data in G.edges(keys=True, data=True):
        if u not in eligible_nodes or v not in eligible_nodes:
            continue
        edge_radii = data.get("window_radii")
        if edge_radii:
            max_radius = max(float(val) for val in edge_radii)
        else:
            max_radius = float(data.get(edge_radius_attr, 0.0))
        if max_radius < threshold:
            continue
        H.add_edge(u, v, key=key, **data)

    if H.number_of_edges() == 0:
        return {"x": False, "y": False, "z": False}, [], {}, H

    percolates, loops, acc_offsets = percolation_winding_multidigraph(H)
    return percolates, loops, acc_offsets, H


def _maxima_pbc_data(maxima_info: np.ndarray):
    zero_mask = np.all(maxima_info[:, 4:7] == 0, axis=1)
    zero_rows = maxima_info[zero_mask]
    coords = zero_rows[:, 1:4].astype(int)
    maxima_pbc_dict = {tuple(coord): int(row[0]) for coord, row in zip(coords, zero_rows)}
    orig_map = np.array(
        [maxima_pbc_dict[tuple(map(int, row[1:4]))] for row in maxima_info], dtype=int
    )
    return maxima_pbc_dict, orig_map, zero_rows

def collect_periodic_edges_canonical(connections, maxima_info):
    import numpy as np

    label_to_idx = {int(row[0]): idx for idx, row in enumerate(maxima_info)}
    _, original_maxima_mapping, _ = _maxima_pbc_data(np.asarray(maxima_info))

    edge_map: dict[
        tuple[int, int, tuple[int, int, int]], dict[str, float | int | list[float]]
    ] = {}
    edges_debug = []

    for c in connections:
        if not c[1]:
            continue

        i_label, j_label = c[0]
        idx_i = label_to_idx.get(int(i_label))
        idx_j = label_to_idx.get(int(j_label))
        if idx_i is None or idx_j is None:
            continue

        base_i = int(original_maxima_mapping[idx_i])
        base_j = int(original_maxima_mapping[idx_j])
        offset_i = maxima_info[idx_i, 4:7].astype(int)
        offset_j = maxima_info[idx_j, 4:7].astype(int)
        # Use the largest window observed for this connection so threshold filtering
        # later uses the true maximum rather than an average.
        window_radius = float(np.max(c[4]))

        start_base = base_i
        end_base = base_j
        start_offset = offset_i
        end_offset = offset_j
        if end_base < start_base:
            start_base, end_base = end_base, start_base
            start_offset, end_offset = end_offset, start_offset

        relative_offset = tuple(int(x) for x in (end_offset - start_offset))

        # Canonicalize self-loop orientation to collapse +/-offset duplicates.
        if start_base == end_base:
            if relative_offset == (0, 0, 0):
                continue
            neg_offset = tuple(-x for x in relative_offset)
            if neg_offset < relative_offset:
                relative_offset = neg_offset

        key = (start_base, end_base, relative_offset)
        entry = edge_map.setdefault(
            key,
            {
                "window_radii": [],
                "window_positions": [],
                "window_radius_min": window_radius,
                "window_radius_max": window_radius,
                "multiplicity": 0,
            },
        )
        entry["window_radii"].append(window_radius)
        if len(c) > 3 and isinstance(c[3], (list, tuple, np.ndarray)):
            for pos in c[3]:
                try:
                    if len(pos) == 3:
                        entry["window_positions"].append(tuple(float(x) for x in pos))
                except Exception:
                    continue
        entry["window_radius_min"] = min(entry["window_radius_min"], window_radius)
        entry["window_radius_max"] = max(entry["window_radius_max"], window_radius)
        entry["multiplicity"] = int(entry["multiplicity"]) + 1
        edges_debug.append(
            [
                int(i_label),
                int(j_label),
                start_base,
                end_base,
                *start_offset,
                *end_offset,
                *relative_offset,
                window_radius,
            ]
        )

    return edge_map, np.array(edges_debug) if edges_debug else None


def collect_periodic_edges(connections, maxima_info):
    """Backward-compatible wrapper for canonical edge collection."""
    return collect_periodic_edges_canonical(connections, maxima_info)

# def collect_periodic_edges(connections, maxima_info):
#     import numpy as np

#     label_to_idx = {int(row[0]): idx for idx, row in enumerate(maxima_info)}
#     maxima_pbc = np.vstack([m[1:4] for m in maxima_info if np.all(m[4:7] == 0)])
#     maxima_pbc_dict = {tuple(m): i + 1 for i, m in enumerate(maxima_pbc)}

#     is_asymmetric = np.all(maxima_info[:, 4:7] == 0, axis=1)
#     asymmetric_labels = set(int(maxima_info[i, 0]) for i in np.where(is_asymmetric)[0])
#     origina_maxima_mapping = np.array(
#         [maxima_pbc_dict[tuple(m[1:4])] for m in maxima_info], dtype=int
#     )

#     edge_map: dict[tuple[int, int, tuple[int, int, int]], float] = {}
#     edges_debug = []

#     for c in connections:
#         if not c[1]:
#             continue
#         i_label, j_label = c[0]
#         if not (i_label in asymmetric_labels or j_label in asymmetric_labels):
#             continue

#         idx_i = label_to_idx[i_label]
#         idx_j = label_to_idx[j_label]
#         base_i = int(origina_maxima_mapping[idx_i])
#         base_j = int(origina_maxima_mapping[idx_j])
#         offset_i = maxima_info[idx_i, 4:7].astype(int)
#         offset_j = maxima_info[idx_j, 4:7].astype(int)
#         window_radius = float(np.mean(c[4]))

#         start_base = base_i
#         end_base = base_j
#         start_offset = offset_i
#         end_offset = offset_j
#         if base_j < base_i:
#             start_base, end_base = base_j, base_i
#             start_offset, end_offset = offset_j, offset_i

#         relative_offset = tuple(int(x) for x in (end_offset - start_offset))
#         if start_base == end_base and relative_offset == (0, 0, 0):
#             continue

#         key = (start_base, end_base, relative_offset)
#         if key not in edge_map:
#             edge_map[key] = window_radius
#             edges_debug.append(
#                 [
#                     i_label,
#                     j_label,
#                     start_base,
#                     end_base,
#                     *start_offset,
#                     *end_offset,
#                     *relative_offset,
#                     window_radius,
#                 ]
#             )
#         else:
#             edge_map[key] = min(edge_map[key], window_radius)

    return edge_map, np.array(edges_debug) if edges_debug else None
# def make_directed_graph_from_segmentation_results(results, dgrid=None):
#     """
#     Convenience helper to build a MultiDiGraph directly from the deduplicated
#     edge map returned by *make_edges_set_from_connections*.
#     """

#     import networkx as nx
#     import numpy as np
#     maxima_info = results["maxima_info"]
#     connections = results["connections"]
#     edge_map, edges_debug = collect_periodic_edges_canonical(connections, maxima_info)
#     if dgrid is None:
#         dgrid = results['dgrid']
#     dgrid_shape = dgrid.shape
#     frac_coords = (maxima_info[:, 1:4] / dgrid_shape) + maxima_info[:, 4:7]
#     G = nx.MultiDiGraph()
#     for (i, j, rel_offset), window_radius in edge_map.items():
#         G.add_edge(i, j, relative_offset=rel_offset, window_radius=window_radius)

#     for node in G.nodes():
#         G.nodes[node]["frac_pos"] = np.round(frac_coords[node - 1], 2)
#         voxel_idx = tuple(maxima_info[node - 1, 1:4].astype(int))
#         G.nodes[node]["pocket_radius"] = float(dgrid[voxel_idx])

#     return G

def make_directed_graph_from_segmentation_results(results, dgrid=None):
    """
    Convenience helper to build a MultiDiGraph directly from the deduplicated
    edge map returned by *make_edges_set_from_connections*.
    """

    import networkx as nx
    import numpy as np
    maxima_info = results["maxima_info"]
    connections = results["connections"]
    edge_map, edges_debug = collect_periodic_edges_canonical(connections, maxima_info)
    if dgrid is None:
        dgrid = results['dgrid']
    dgrid_shape = dgrid.shape
    frac_coords = (maxima_info[:, 1:4] / dgrid_shape) + maxima_info[:, 4:7]
    G = nx.MultiDiGraph()
    for (i, j, rel_offset), data in edge_map.items():
        window_radii = tuple(float(r) for r in data.get("window_radii", ()))
        window_positions = tuple(tuple(float(x) for x in p) for p in data.get("window_positions", ()))
        if window_radii:
            window_radius_max = max(window_radii)
            window_radius_min = min(window_radii)
        else:
            window_radius_max = float(data.get("window_radius_max", 0.0))
            window_radius_min = float(data.get("window_radius_min", window_radius_max))
        G.add_edge(
            i,
            j,
            relative_offset=rel_offset,
            window_radius=window_radius_max,
            window_radius_max=window_radius_max,
            window_radius_min=window_radius_min,
            window_radii=window_radii,
            window_positions=window_positions,
            multiplicity=int(data.get("multiplicity", 1)),
        )

    for node in G.nodes():
        G.nodes[node]["frac_pos"] = np.round(frac_coords[node - 1], 2)
        voxel_idx = tuple(maxima_info[node - 1, 1:4].astype(int))
        G.nodes[node]["pocket_radius"] = float(dgrid[voxel_idx])

    return G


def build_base_region_labels(maxima_info: np.ndarray, region_labels: set, maxima_pbc_dict: Dict[Tuple[int, int, int], int]):
    base_region_labels: Dict[int, set] = {}
    for row in maxima_info:
        label = int(row[0])
        if label not in region_labels:
            continue
        base_idx = tuple(map(int, row[1:4]))
        base_label = maxima_pbc_dict.get(base_idx)
        if base_label is None:
            continue
        base_region_labels.setdefault(base_label, set()).add(label)
    return base_region_labels


def build_base_region_labels_from_results(results: Dict[str, object]) -> Dict[int, set]:
    maxima_info = np.asarray(results["maxima_info"])
    maxima_pbc_dict, _, _ = _maxima_pbc_data(maxima_info)
    regions = np.asarray(results["regions"])
    region_labels = set(int(x) for x in np.unique(regions) if int(x) != 0)
    return build_base_region_labels(maxima_info, region_labels, maxima_pbc_dict)


def _prepare_graph_and_labels(
    results: Dict[str, object],
    graph: Optional[nx.MultiDiGraph] = None,
    base_region_labels: Optional[Dict[int, set]] = None,
) -> Tuple[nx.MultiDiGraph, Dict[int, set]]:
    if graph is None:
        graph = make_directed_graph_from_segmentation_results(results)
    if base_region_labels is None:
        base_region_labels = build_base_region_labels_from_results(results)
    return graph, base_region_labels


def _axes_from_winding(winding: Tuple[int, int, int]) -> List[str]:
    axes = []
    if winding[0] != 0:
        axes.append("x")
    if winding[1] != 0:
        axes.append("y")
    if winding[2] != 0:
        axes.append("z")
    return axes


def annotate_clusters(loop_list: List[Dict[str, object]]) -> Dict[Tuple[frozenset, Tuple[int, int, int]], int]:
    cluster_ids: Dict[Tuple[frozenset, Tuple[int, int, int]], int] = {}
    next_id = 1
    for loop in loop_list:
        key = (
            frozenset(loop["nodes"]),
            tuple(abs(int(w)) for w in loop["winding"]),
        )
        if key not in cluster_ids:
            cluster_ids[key] = next_id
            next_id += 1
        loop["cluster_id"] = cluster_ids[key]
    return cluster_ids


def percolation_all_axes_winding(
    results: Dict[str, object],
    threshold: float,
    graph: Optional[nx.MultiDiGraph] = None,
    base_region_labels: Optional[Dict[int, set]] = None,
    node_radius_attr: str = "pocket_radius",
    edge_radius_attr: str = "window_radius",
    include_dangling: bool = True,
    expand_components: bool = True,
) -> Tuple[
    Dict[str, List[List[int]]],
    Dict[int, List[int]],
    Dict[int, Dict[str, object]],
]:
    graph, base_region_labels = _prepare_graph_and_labels(results, graph, base_region_labels)
    percolates, loops, acc_offsets, filtered_graph = detect_percolation_with_threshold(
        graph, threshold, node_radius_attr=node_radius_attr, edge_radius_attr=edge_radius_attr
    )
    annotate_clusters(loops)
    graph_undirected = nx.Graph(filtered_graph.to_undirected(as_view=True))
    component_lookup: Dict[int, set] = {}
    if expand_components:
        for comp in nx.connected_components(graph_undirected):
            comp_set = set(comp)
            for node in comp_set:
                component_lookup[node] = comp_set

    axis_clusters: Dict[str, List[List[int]]] = {"x": [], "y": [], "z": []}
    cluster_details: Dict[int, List[int]] = {}
    cluster_axes: Dict[int, set] = {}
    cluster_region_sets: Dict[int, set] = {}
    cluster_node_sets: Dict[int, set] = {}

    for loop in loops:
        cid = loop["cluster_id"]
        cluster_axes.setdefault(cid, set()).update(_axes_from_winding(tuple(loop["winding"])))
        region_set = cluster_region_sets.setdefault(cid, set())
        for base in loop["nodes"]:
            region_set.update(base_region_labels.get(base, ()))
        cluster_node_sets.setdefault(cid, set()).update(loop["nodes"])

    cluster_metadata: Dict[int, Dict[str, object]] = {}
    for cid in cluster_region_sets:
        if expand_components:
            comp_nodes: set = set()
            for node in cluster_node_sets.get(cid, ()):
                comp_nodes.update(component_lookup.get(node, ()))
            comp_nodes = {n for n in comp_nodes if n in filtered_graph}
        else:
            comp_nodes = set(cluster_node_sets.get(cid, ()))
        if not comp_nodes:
            continue

        if expand_components:
            comp_subgraph = graph_undirected.subgraph(comp_nodes)
            subgraph_no_loops = comp_subgraph.copy()
            subgraph_no_loops.remove_edges_from(nx.selfloop_edges(subgraph_no_loops))
            try:
                core_nodes = set(nx.k_core(subgraph_no_loops, k=2).nodes())
            except nx.NetworkXError:
                core_nodes = set(subgraph_no_loops.nodes())
            if not core_nodes:
                core_nodes = comp_nodes
            full_nodes = comp_nodes
            dangling_nodes = full_nodes - core_nodes
        else:
            core_nodes = comp_nodes
            full_nodes = comp_nodes
            dangling_nodes = set()

        edge_set: set = set()
        for u, v, key in filtered_graph.edges(keys=True):
            if u in core_nodes and v in core_nodes:
                if u <= v:
                    edge_set.add((u, v, key))
                else:
                    edge_set.add((v, u, key))
        edge_set_full: set = set()
        for u, v, key in filtered_graph.edges(keys=True):
            if u in full_nodes and v in full_nodes:
                if u <= v:
                    edge_set_full.add((u, v, key))
                else:
                    edge_set_full.add((v, u, key))

        pruned_region_set = set()
        for node in core_nodes:
            pruned_region_set.update(base_region_labels.get(node, ()))
        full_region_set = set()
        for node in full_nodes:
            full_region_set.update(base_region_labels.get(node, ()))

        region_list = sorted(pruned_region_set)
        region_list_full = sorted(full_region_set)
        dangling_nodes_sorted = sorted(dangling_nodes)
        if include_dangling:
            nodes_out = sorted(full_nodes)
            edges_out = sorted(edge_set_full)
            regions_out = region_list_full
        else:
            nodes_out = sorted(core_nodes)
            edges_out = sorted(edge_set)
            regions_out = region_list
        cluster_details[cid] = region_list
        metadata: Dict[str, object] = {
            "regions": regions_out,
            "nodes": nodes_out,
            "edges": edges_out,
            "axes": sorted(cluster_axes.get(cid, ())),
            "regions_full": region_list_full,
            "nodes_full": sorted(full_nodes),
            "edges_full": sorted(edge_set_full),
            "dangling_nodes": dangling_nodes_sorted,
        }
        cluster_metadata[cid] = metadata
        for axis in cluster_axes.get(cid, ()):
            axis_clusters[axis].append(regions_out)

    return axis_clusters, cluster_details, cluster_metadata




def check_percolation_winding(
    results: Dict[str, object],
    threshold: float,
    graph: Optional[nx.MultiDiGraph] = None,
    base_region_labels: Optional[Dict[int, set]] = None,
    include_dangling: bool = True,
    expand_components: bool = True,
) -> bool:
    axis_clusters, _, _ = percolation_all_axes_winding(
        results,
        threshold,
        graph=graph,
        base_region_labels=base_region_labels,
        include_dangling=include_dangling,
        expand_components=expand_components,
    )
    return any(axis_clusters[axis] for axis in axis_clusters)


def _gather_threshold_candidates(graph: nx.MultiDiGraph) -> List[float]:
    node_vals = {float(data.get("pocket_radius", 0.0)) for _, data in graph.nodes(data=True)}
    edge_vals: set[float] = set()
    for _, _, data in graph.edges(data=True):
        radii = data.get("window_radii")
        if radii:
            edge_vals.update(float(val) for val in radii)
        else:
            edge_vals.add(float(data.get("window_radius", 0.0)))
    return sorted({val for val in node_vals.union(edge_vals) if val > 0})


def _default_mask_thickness(results: Dict[str, object]) -> float:
    if "mask_thickness" in results:
        return float(results["mask_thickness"])
    kwargs = results.get("kwargs", {})
    if isinstance(kwargs, dict) and "mask_thickness" in kwargs:
        return float(kwargs["mask_thickness"])
    return 0.0


def _dfsd_threshold(plr: float, results: Dict[str, object], kwargs: Optional[Dict[str, object]]) -> float:
    """Return a slightly relaxed threshold for DFSD region selection."""
    eps = 1e-2
    if kwargs is not None and "dfsd_threshold_eps" in kwargs:
        try:
            eps = max(float(kwargs["dfsd_threshold_eps"]), 0.0)
        except Exception:
            eps = 1e-2
    min_default = _default_mask_thickness(results)
    return max(min_default, float(plr) - eps)


def _cosine_similarity_safe(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _window_frac_from_edge_data(
    edge_data: Dict[str, object], dgrid_shape: Tuple[int, int, int]
) -> Optional[np.ndarray]:
    """Choose one representative window position and convert to wrapped fractional coordinates."""
    positions = edge_data.get("window_positions")
    if not positions:
        return None
    radii = edge_data.get("window_radii") or ()
    pick_idx = 0
    if radii and len(radii) == len(positions):
        pick_idx = int(np.argmax(np.asarray(radii, dtype=float)))
    try:
        pos = np.asarray(positions[pick_idx], dtype=float)
    except Exception:
        return None
    if pos.shape != (3,):
        return None
    shp = np.asarray(dgrid_shape, dtype=float)
    if np.any(shp <= 0):
        return None
    frac = ((np.mod(pos, shp)) + 0.5) / shp
    return np.mod(frac, 1.0)


def _window_shell_fingerprint(
    frac_pos: np.ndarray, atoms, shell_radius: float
) -> Optional[np.ndarray]:
    """Direction-invariant shell fingerprint around a window centre."""
    from ase.geometry import complete_cell

    if atoms is None:
        return None
    try:
        cell = complete_cell(atoms.get_cell()).T
        atom_frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
        atom_z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    except Exception:
        return None
    if atom_frac.size == 0:
        return None
    dr_frac = atom_frac - frac_pos.reshape(1, 3)
    dr_frac -= np.round(dr_frac)
    dr_cart = np.einsum("ni,ij->nj", dr_frac, cell)
    dist = np.linalg.norm(dr_cart, axis=1)
    mask = (dist > 1e-8) & (dist <= float(shell_radius))
    if not np.any(mask):
        return None
    vec = np.zeros(118, dtype=float)
    w = 1.0 / np.maximum(dist[mask], 1e-3)
    for z, ww in zip(atom_z[mask], w):
        if 1 <= int(z) <= 118:
            vec[int(z) - 1] += float(ww)
    nrm = float(np.linalg.norm(vec))
    if nrm <= 1e-12:
        return None
    return vec / nrm


def annotate_graph_edges_with_window_fingerprints(
    graph: nx.MultiDiGraph,
    results: Dict[str, object],
    atoms=None,
    shell_radius: float = 3.5,
) -> nx.MultiDiGraph:
    """Annotate each graph edge with a lightweight window shell fingerprint."""
    if atoms is None and isinstance(results, dict):
        atoms = results.get("mat_atoms")
    if atoms is None:
        return graph
    dgrid = np.asarray(results.get("dgrid"))
    if dgrid.ndim != 3:
        return graph
    shape = tuple(int(x) for x in dgrid.shape)
    for u, v, k, data in graph.edges(keys=True, data=True):
        frac = _window_frac_from_edge_data(data, shape)
        if frac is None:
            graph[u][v][k]["window_fp_ready"] = False
            continue
        fp = _window_shell_fingerprint(frac, atoms, shell_radius=float(shell_radius))
        graph[u][v][k]["window_frac_pos"] = tuple(float(x) for x in frac)
        if fp is None:
            graph[u][v][k]["window_fp_ready"] = False
            continue
        graph[u][v][k]["window_fp_vec"] = tuple(float(x) for x in fp)
        graph[u][v][k]["window_fp_ready"] = True
    return graph


def _incident_edge_records(graph: nx.MultiDiGraph, node: int):
    records: Dict[CanonicalEdgeID, Tuple[int, int, int, Dict[str, object]]] = {}
    for u, v, k, data in graph.out_edges(node, keys=True, data=True):
        cid = CanonicalEdgeID.from_nodes(int(u), int(v), int(k))
        records[cid] = (int(u), int(v), int(k), data)
    for u, v, k, data in graph.in_edges(node, keys=True, data=True):
        cid = CanonicalEdgeID.from_nodes(int(u), int(v), int(k))
        records[cid] = (int(u), int(v), int(k), data)
    return list(records.values())


def _expand_dfsd_labels_with_window_fingerprint(
    *,
    graph: nx.MultiDiGraph,
    base_region_labels: Dict[int, set],
    percolating_labels: set,
    results: Dict[str, object],
    dgrid: np.ndarray,
    plr: float,
    kwargs: Optional[Dict[str, object]],
    atoms=None,
) -> set:
    """Optionally rescue max-cavity labels when near-threshold windows look similar."""
    if kwargs is None or not bool(kwargs.get("window_fp_for_dfsd", False)):
        return percolating_labels
    if not percolating_labels:
        return percolating_labels
    regions = np.asarray(results["regions"])
    valid = regions > 0
    if not np.any(valid):
        return percolating_labels
    max_idx = int(np.argmax(np.where(valid, dgrid, -np.inf)))
    max_label = int(regions.flat[max_idx])
    if max_label <= 0 or max_label in percolating_labels:
        return percolating_labels

    owner_node = None
    for node, labels in base_region_labels.items():
        if max_label in labels:
            owner_node = int(node)
            break
    if owner_node is None:
        return percolating_labels

    percolating_nodes = {
        int(node)
        for node, labels in base_region_labels.items()
        if set(labels).intersection(percolating_labels)
    }
    if not percolating_nodes:
        return percolating_labels

    shell_radius = float(kwargs.get("window_fp_shell_radius", 3.5))
    annotate_graph_edges_with_window_fingerprints(
        graph, results, atoms=atoms or results.get("mat_atoms"), shell_radius=shell_radius
    )

    sim_thresh = float(kwargs.get("window_fp_sim_thresh", 0.88))
    size_tol = float(kwargs.get("window_fp_size_tol", 0.20))
    ref_tol = float(kwargs.get("window_fp_ref_tol", 0.12))
    cand_tol = float(kwargs.get("window_fp_candidate_tol", 0.12))

    ref_edges = []
    for node in percolating_nodes:
        for _u, _v, _k, data in _incident_edge_records(graph, node):
            if float(data.get("window_radius", 0.0)) < float(plr) - ref_tol:
                continue
            if not data.get("window_fp_ready", False):
                continue
            vec = np.asarray(data.get("window_fp_vec"), dtype=float)
            ref_edges.append((vec, float(data.get("window_radius", 0.0))))
    if not ref_edges:
        return percolating_labels

    owner_records = _incident_edge_records(graph, owner_node)
    for u, v, _k, data in owner_records:
        wr = float(data.get("window_radius", 0.0))
        if wr < float(plr) - cand_tol:
            continue
        if not data.get("window_fp_ready", False):
            continue
        cvec = np.asarray(data.get("window_fp_vec"), dtype=float)
        for rvec, rwr in ref_edges:
            if abs(wr - rwr) > size_tol:
                continue
            if _cosine_similarity_safe(cvec, rvec) >= sim_thresh:
                new_labels = set(percolating_labels)
                new_labels.update(base_region_labels.get(int(u), set()))
                new_labels.update(base_region_labels.get(int(v), set()))
                return new_labels
    return percolating_labels


def _window_fp_params(kwargs: Optional[Dict[str, object]]) -> Dict[str, float]:
    kw = kwargs or {}
    return {
        "shell_radius": float(kw.get("window_fp_shell_radius", 3.5)),
        "sim_thresh": float(kw.get("window_fp_sim_thresh", 0.88)),
        "size_tol": float(kw.get("window_fp_size_tol", 0.20)),
        "ref_tol": float(kw.get("window_fp_ref_tol", 0.12)),
        "cand_tol": float(kw.get("window_fp_candidate_tol", 0.12)),
        "subprobe_gap": float(kw.get("window_fp_subprobe_gap", 0.5)),
    }


def _collect_window_fp_reference_edges(
    graph: nx.MultiDiGraph,
    nodes: set,
    min_window_radius: float,
) -> List[Tuple[np.ndarray, float]]:
    """Collect (fingerprint, window_radius) from open windows on the given node set."""
    ref_edges: List[Tuple[np.ndarray, float]] = []
    for node in nodes:
        for _u, _v, _k, data in _incident_edge_records(graph, int(node)):
            wr = float(data.get("window_radius", 0.0))
            if wr < float(min_window_radius):
                continue
            if not data.get("window_fp_ready", False):
                continue
            ref_edges.append(
                (np.asarray(data.get("window_fp_vec"), dtype=float), wr)
            )
    return ref_edges


def _window_fp_matches_reference_edges(
    cvec: np.ndarray,
    wr: float,
    ref_edges: List[Tuple[np.ndarray, float]],
    *,
    sim_thresh: float,
    ref_minus_cand_max: float,
) -> bool:
    """True when candidate matches a reference edge in shell cosine and window size.

    Uses a **one-sided** radius gate: the candidate may be sub-probe relative to
    the reference open mouth, but not by more than *ref_minus_cand_max* (defaults
    to ``window_fp_subprobe_gap``).  This blocks annex bridges like XALDOE
    (wr_ref − wr_cand ≈ 0.7–0.8 Å) while allowing Mechanism B grid undershoot
    (≈ 0.4–0.5 Å below the same mouth).
    """
    for rvec, rwr in ref_edges:
        if (float(rwr) - float(wr)) > float(ref_minus_cand_max):
            continue
        if _cosine_similarity_safe(cvec, rvec) >= float(sim_thresh):
            return True
    return False


def _expand_percolating_labels_with_window_fingerprint(
    *,
    graph: nx.MultiDiGraph,
    base_region_labels: Dict[int, set],
    percolating_labels: set,
    results: Dict[str, object],
    probe_radius: float,
    kwargs: Optional[Dict[str, object]] = None,
    atoms=None,
) -> set:
    """Rescue watershed regions behind sub-probe windows with matching fingerprints.

    When winding percolation at *probe_radius* excludes interior pockets because
    connecting graph windows are slightly below the probe (grid/watershed
    discretization), this BFS adds neighbor nodes whose sub-probe windows share
    a chemical shell fingerprint with windows already open at the probe and
    whose ``window_radius`` is at most ``window_fp_subprobe_gap`` (or
    ``window_fp_ref_minus_cand_tol``) **smaller** than the matched reference
    open edge (one-sided; blocks annex bridges with much smaller throats).

  Controlled by ``window_fp_for_accessible`` in *kwargs* (off by default).
    """
    if kwargs is None or not bool(kwargs.get("window_fp_for_accessible", False)):
        return percolating_labels
    if not percolating_labels:
        return percolating_labels

    params = _window_fp_params(kwargs)
    annotate_graph_edges_with_window_fingerprints(
        graph,
        results,
        atoms=atoms or results.get("mat_atoms"),
        shell_radius=params["shell_radius"],
    )

    percolating_nodes = {
        int(node)
        for node, labels in base_region_labels.items()
        if set(labels).intersection(percolating_labels)
    }
    if not percolating_nodes:
        return percolating_labels

    ref_edges = _collect_window_fp_reference_edges(
        graph,
        percolating_nodes,
        float(probe_radius) - params["ref_tol"],
    )
    if not ref_edges:
        return percolating_labels

    expanded_labels = set(percolating_labels)
    active_nodes = set(percolating_nodes)
    visited_edges: set = set()
    queue: deque = deque(active_nodes)
    probe = float(probe_radius)

    while queue:
        node = int(queue.popleft())
        for u, v, key, data in _incident_edge_records(graph, node):
            edge_id = (min(int(u), int(v)), max(int(u), int(v)), int(key))
            if edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)

            other = int(v) if int(u) == node else int(u)
            wr = float(data.get("window_radius", 0.0))
            if wr >= probe:
                if other not in active_nodes:
                    active_nodes.add(other)
                    expanded_labels.update(base_region_labels.get(other, ()))
                    queue.append(other)
                continue

            gap = probe - wr
            if gap > params["subprobe_gap"]:
                continue
            if not data.get("window_fp_ready", False):
                continue
            cvec = np.asarray(data.get("window_fp_vec"), dtype=float)
            ref_minus_cand_max = float(
                kwargs.get("window_fp_ref_minus_cand_tol", params["subprobe_gap"])
                if kwargs
                else params["subprobe_gap"]
            )
            if not _window_fp_matches_reference_edges(
                cvec,
                wr,
                ref_edges,
                sim_thresh=params["sim_thresh"],
                ref_minus_cand_max=ref_minus_cand_max,
            ):
                continue
            if other not in active_nodes:
                active_nodes.add(other)
                expanded_labels.update(base_region_labels.get(other, ()))
                queue.append(other)

    return expanded_labels


def percolating_region_labels_at_probe_winding(
    results: Dict[str, object],
    probe_radius: float,
    kwargs: Optional[Dict[str, object]] = None,
    graph: Optional[nx.MultiDiGraph] = None,
    base_region_labels: Optional[Dict[int, set]] = None,
    include_dangling: bool = True,
    expand_components: bool = True,
    atoms=None,
) -> set:
    """Region labels that are probe-accessible under winding percolation (+ optional FP rescue)."""
    graph, base_region_labels = _prepare_graph_and_labels(results, graph, base_region_labels)
    axis_clusters, _, _ = percolation_all_axes_winding(
        results,
        float(probe_radius),
        graph=graph,
        base_region_labels=base_region_labels,
        include_dangling=include_dangling,
        expand_components=expand_components,
    )
    labels: set = set()
    for groups in axis_clusters.values():
        for group in groups:
            labels.update(group)
    return _expand_percolating_labels_with_window_fingerprint(
        graph=graph,
        base_region_labels=base_region_labels,
        percolating_labels=labels,
        results=results,
        probe_radius=float(probe_radius),
        kwargs=kwargs,
        atoms=atoms,
    )


def find_pore_limiting_radius_winding(
    results: Dict[str, object],
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
    graph: Optional[nx.MultiDiGraph] = None,
    base_region_labels: Optional[Dict[int, set]] = None,
    include_dangling: bool = True,
    expand_components: bool = True,
) -> Optional[float]:
    graph, base_region_labels = _prepare_graph_and_labels(results, graph, base_region_labels)
    candidates = _gather_threshold_candidates(graph)
    if not candidates:
        return None

    min_default = _default_mask_thickness(results)
    if min_threshold is None:
        min_threshold = min_default
    min_threshold = max(min_threshold, min_default)
    if max_threshold is None:
        max_threshold = candidates[-1]

    filtered = [val for val in candidates if min_threshold <= val <= max_threshold]
    if not filtered:
        return None

    left, right = 0, len(filtered) - 1
    last_good = None
    while left <= right:
        mid = (left + right) // 2
        threshold = filtered[mid]
        axis_clusters, _, _ = percolation_all_axes_winding(
            results,
            threshold,
            graph=graph,
            base_region_labels=base_region_labels,
            include_dangling=include_dangling,
            expand_components=expand_components,
        )
        if any(axis_clusters[axis] for axis in axis_clusters):
            last_good = threshold
            left = mid + 1
        else:
            right = mid - 1
    final_radius = last_good
    if last_good is not None:
        (
            _,
            _,
            cluster_metadata,
        ) = percolation_all_axes_winding(
            results,
            last_good,
            graph=graph,
            base_region_labels=base_region_labels,
            include_dangling=include_dangling,
            expand_components=expand_components,
        )
        percolating_clusters = {
            cid: data
            for cid, data in cluster_metadata.items()
            if data.get("axes")
        }
        results["plr_clusters_winding"] = percolating_clusters
        results["plr_threshold_candidate_winding"] = last_good
        results["plr_radius_winding"] = last_good
    return final_radius


# def compute_percolation_history_from_results_winding(
#     results: Dict[str, object],
#     thresholds: Optional[Sequence[float]] = None,
#     step: float = 0.05,
#     min_threshold: Optional[float] = None,
#     max_threshold: Optional[float] = None,
#     graph: Optional[nx.MultiDiGraph] = None,
#     base_region_labels: Optional[Dict[int, set]] = None,
# ) -> List[Tuple[float, Dict[str, List[List[int]]]]]:
#     graph, base_region_labels = _prepare_graph_and_labels(results, graph, base_region_labels)
#     if thresholds is None:
#         min_default = _default_mask_thickness(results)
#         if min_threshold is None:
#             min_threshold = min_default
#         min_threshold = max(min_threshold, min_default)
#         if max_threshold is None:
#             plr = find_pore_limiting_radius_winding(
#                 results,
#                 min_threshold=min_threshold,
#                 max_threshold=None,
#                 graph=graph,
#                 base_region_labels=base_region_labels,
#             )
#             if plr is None:
#                 return []
#             max_threshold = plr
#         thresholds = np.arange(min_threshold, max_threshold + 1e-9, step)

#     history: List[Tuple[float, Dict[str, List[List[int]]]]] = []
#     for threshold in thresholds:
#         axis_clusters, _ = percolation_all_axes_winding(
#             results, float(threshold), graph=graph, base_region_labels=base_region_labels
#         )
#         if any(axis_clusters[axis] for axis in axis_clusters):
#             history.append((float(threshold), axis_clusters))
#     return history

def update_results_with_percolation_history_winding(
    results: Dict[str, object],
    thresholds: Optional[Sequence[float]] = None,
    step: float = 0.05,
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
    graph: Optional[nx.MultiDiGraph] = None,
    base_region_labels: Optional[Dict[int, set]] = None,
    include_dangling: bool = True,
    expand_components: bool = True,
) -> Dict[str, object]:
    graph, base_region_labels = _prepare_graph_and_labels(results, graph, base_region_labels)
    min_default = _default_mask_thickness(results)
    if min_threshold is None:
        min_threshold = min_default
    min_threshold = max(min_threshold, min_default)
    plr = find_pore_limiting_radius_winding(
        results,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        graph=graph,
        base_region_labels=base_region_labels,
        include_dangling=include_dangling,
        expand_components=expand_components,
    )
    if plr is None:
        from mofography.percolation_results import store_percolation_history

        store_percolation_history(results, "winding", [])
        results["pld_locations"] = np.empty((0, 3), dtype=int)
        results.setdefault("plr_clusters_winding", {})
        return results

    if thresholds is None:
        if max_threshold is None:
            max_threshold = plr
        thresholds = np.arange(min_threshold, max_threshold + 1e-9, step)

    history: List[Tuple[float, Dict[str, List[List[int]]]]] = []
    for threshold in thresholds:
        axis_clusters, _, _ = percolation_all_axes_winding(
            results,
            float(threshold),
            graph=graph,
            base_region_labels=base_region_labels,
            include_dangling=include_dangling,
            expand_components=expand_components,
        )
        if any(axis_clusters[axis] for axis in axis_clusters):
            history.append((float(threshold), axis_clusters))
    if results.get("dgrid") is not None:
        dgrid = np.asarray(results["dgrid"])
        pld_locations = np.argwhere(np.isclose(dgrid, plr, atol=1e-10))
    else:
        pld_locations = np.empty((0, 3), dtype=int)
    from mofography.percolation_results import store_percolation_history

    store_percolation_history(results, "winding", history, pld_locations=pld_locations)
    results["pld_locations"] = pld_locations
    return results



def compute_percolation_history_winding(
    results: Dict[str, object],
    include_dangling: bool = True,
    expand_components: bool = True,
    **kwargs,
) -> Tuple[List[Tuple[float, Dict[str, List[List[int]]]]], np.ndarray, Dict[int, Dict[str, object]]]:
    """Wrapper similar to ``compute_percolation_history2`` but also returns cluster metadata."""
    results = update_results_with_percolation_history_winding(
        results,
        include_dangling=include_dangling,
        expand_components=expand_components,
        **kwargs,
    )
    from mofography.percolation_results import get_percolation_history

    percolation_history = get_percolation_history(results, "winding")
    pld_locations = results.get("pld_locations")
    if pld_locations is None:
        pld_locations = results.get("pld_locations_winding", np.empty((0, 3), dtype=int))
    final_clusters = results.get("plr_clusters_winding", {})
    return percolation_history, pld_locations, final_clusters


def get_pore_dia_mgr_winding(
    mat_atoms=None,
    dgrid: Optional[np.ndarray] = None,
    kwargs: Optional[Dict[str, object]] = None,
    results: Optional[Dict[str, object]] = None,
    include_dangling: bool = True,
    expand_components: bool = True,
) -> Tuple[float, float, float, np.ndarray]:
    if kwargs is None:
        kwargs = {}

    spacing = kwargs.get("spacing", 0.5)
    if results is None:
        if dgrid is None:
            if mat_atoms is None:
                raise ValueError("mat_atoms or results must be provided.")
            dgrid = mgr.dgrid_from_atoms_cpu(mat_atoms, spacing=spacing, block_size=50000)
        results = mgr.do_periodic_segmentation_rust(
            dgrid,
            mat_atoms,
            mask_thickness=kwargs.get("mask_thickness", 1.5),
            h=kwargs.get("h", 0.2),
            spacing=spacing,
            return_dgrid=True,
            threshold_abs=kwargs.get("threshold_abs", 1.5),
            min_distance=kwargs.get("min_distance", 1.5),
            return_padded_regions=True,
            extra_padding=kwargs.get("extra_padding", 2),
        )
    else:
        dgrid = np.asarray(results["dgrid"])

    graph = make_directed_graph_from_segmentation_results(results)
    base_region_labels = build_base_region_labels_from_results(results)

    lcd = float(dgrid.max() * 2)
    plr = find_pore_limiting_radius_winding(
        results,
        graph=graph,
        base_region_labels=base_region_labels,
        include_dangling=include_dangling,
        expand_components=expand_components,
    )
    if plr is None:
        return lcd, 0.0, 0.0, dgrid

    pld = float(plr * 2)
    dfsd_threshold = _dfsd_threshold(plr, results, kwargs)
    axis_clusters, _, _ = percolation_all_axes_winding(
        results,
        dfsd_threshold,
        graph=graph,
        base_region_labels=base_region_labels,
        include_dangling=include_dangling,
        expand_components=expand_components,
    )
    percolating_labels = set()
    for groups in axis_clusters.values():
        for labels in groups:
            percolating_labels.update(labels)
    percolating_labels = _expand_dfsd_labels_with_window_fingerprint(
        graph=graph,
        base_region_labels=base_region_labels,
        percolating_labels=percolating_labels,
        results=results,
        dgrid=dgrid,
        plr=float(plr),
        kwargs=kwargs,
        atoms=mat_atoms,
    )

    dfsd = 0.0
    if percolating_labels:
        regions = np.asarray(results["regions"])
        mask = np.isin(regions, list(percolating_labels))
        if mask.any():
            dfsd = float(dgrid[mask].max() * 2)

    return lcd, pld, dfsd, dgrid


def _has_suspicious_self_loop_asymmetry(
    edge_map: dict,
    maxima_info: np.ndarray,
    dgrid: np.ndarray,
    threshold: float = 0.3,
) -> bool:
    """Return True if any node's self-loop radii spread suggests watershed spill.

    Deterministic watershed (rust_det) can spill 1-2 voxels across
    thin boundaries, inflating self-loop window radii in some directions.  When
    the spread of self-loop radii for a single node exceeds *threshold* × pocket
    radius, the segmentation is flagged as unreliable.

    Parameters
    ----------
    edge_map : dict
        Output of ``collect_periodic_edges``.
    maxima_info : ndarray
        Segmentation maxima_info array (columns: label, i, j, k, di, dj, dk).
    dgrid : ndarray
        Distance grid used for segmentation.
    threshold : float
        Maximum allowed spread/pocket_radius ratio (default 0.3).

    Returns
    -------
    bool
        True if at least one node exceeds the asymmetry threshold.
    """
    from collections import defaultdict

    node_self: dict[int, list[float]] = defaultdict(list)
    for (s, e, off), data in edge_map.items():
        if s == e and tuple(off) != (0, 0, 0):
            node_self[s].append(float(data["window_radius_min"]))

    for node, radii in node_self.items():
        if len(radii) < 2:
            continue
        spread = max(radii) - min(radii)
        mi_rows = maxima_info[maxima_info[:, 0] == node]
        if len(mi_rows) == 0:
            continue
        voxel = tuple(mi_rows[0, 1:4].astype(int))
        if not all(0 <= v < s for v, s in zip(voxel, dgrid.shape)):
            continue
        pocket_r = float(dgrid[voxel])
        if pocket_r > 0 and spread / pocket_r > threshold:
            return True
    return False


def get_pore_dia_mgr_winding2(
    mat_atoms=None,
    dgrid: Optional[np.ndarray] = None,
    kwargs: Optional[Dict[str, object]] = None,
    results: Optional[Dict[str, object]] = None,
    include_dangling: bool = True,
    expand_components: bool = True,
    auto_rerun: bool = True,
) -> Tuple[float, float, float, np.ndarray]:
    """Compute pore diameters using winding percolation with spill guardrail.

    Identical to ``get_pore_dia_mgr_winding`` but adds automatic re-segmentation
    when deterministic watershed spill is detected.  Spill is flagged when any
    node's self-loop radii spread exceeds 30% of its pocket radius.

    When *auto_rerun* is True and spill is detected:
    - Re-segments with ``mask_thickness=1.5`` using the same backend and dgrid.
    - Recomputes PLD from the guarded segmentation.
    - Stores ``_spill_corrected``, ``_original_pld``, ``_corrected_pld`` in
      *results* (if mutable) for diagnostics.

    Parameters
    ----------
    mat_atoms : ase.Atoms, optional
        ASE Atoms object. Required if *results* is None.
    dgrid : ndarray, optional
        Pre-computed distance grid.
    kwargs : dict, optional
        Keyword arguments for ``do_periodic_segmentation_rust``.
    results : dict, optional
        Pre-computed segmentation results.
    include_dangling : bool
        Whether to include dangling edges.
    expand_components : bool
        Whether to expand components.
    auto_rerun : bool
        If True, silently re-segment with guard mask when spill is detected.
        Requires ``results['dgrid']`` and ``results.get('mat_atoms')`` (or
        *mat_atoms* passed separately).  Raises ``ValueError`` if neither is
        available.

    Returns
    -------
    tuple[float, float, float, ndarray]
        (lcd, pld, dfsd, dgrid)
    """
    if kwargs is None:
        kwargs = {}

    spacing = kwargs.get("spacing", 0.5)
    if results is None:
        if dgrid is None:
            if mat_atoms is None:
                raise ValueError("mat_atoms or results must be provided.")
            dgrid = mgr.dgrid_from_atoms_cpu(mat_atoms, spacing=spacing, block_size=50000)
        results = mgr.do_periodic_segmentation_rust(
            dgrid,
            mat_atoms,
            mask_thickness=kwargs.get("mask_thickness", 1.5),
            h=kwargs.get("h", 0.2),
            spacing=spacing,
            return_dgrid=True,
            threshold_abs=kwargs.get("threshold_abs", 1.5),
            min_distance=kwargs.get("min_distance", 1.5),
            return_padded_regions=True,
            extra_padding=kwargs.get("extra_padding", 2),
        )
    else:
        dgrid = np.asarray(results["dgrid"])

    graph = make_directed_graph_from_segmentation_results(results)
    base_region_labels = build_base_region_labels_from_results(results)

    lcd = float(dgrid.max() * 2)
    plr = find_pore_limiting_radius_winding(
        results,
        graph=graph,
        base_region_labels=base_region_labels,
        include_dangling=include_dangling,
        expand_components=expand_components,
    )
    if plr is None:
        return lcd, 0.0, 0.0, dgrid

    pld = float(plr * 2)

    # --- Spill guardrail ---
    if auto_rerun:
        edge_map, _ = collect_periodic_edges_canonical(
            results["connections"], results["maxima_info"]
        )
        params = results.get("input_params", {})
        backend = params.get("watershed_backend", "rust_det")
        mask = float(params.get("mask_thickness", 0.05))

        if backend in ("rust_det", "rust_det_parallel", "skimage") and mask < 1.5:
            if _has_suspicious_self_loop_asymmetry(
                edge_map, results["maxima_info"], dgrid
            ):
                # Need dgrid + atoms to rerun
                atoms = results.get("mat_atoms") or mat_atoms
                if atoms is None:
                    raise ValueError(
                        f"Watershed spill detected (backend={backend}, "
                        f"mask={mask:.2f}) but cannot auto-correct: "
                        "neither results['mat_atoms'] nor mat_atoms is available. "
                        "Either store mat_atoms in results or pass it to this "
                        "function, or set auto_rerun=False."
                    )
                dgrid_for_rerun = dgrid
                if "dgrid" not in results:
                    raise ValueError(
                        "Watershed spill detected but 'dgrid' missing from results. "
                        "Call do_periodic_segmentation_rust with return_dgrid=True, "
                        "or pass dgrid directly."
                    )

                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    "Spill guardrail triggered (backend=%s, mask=%.2f). "
                    "Re-segmenting with mask_thickness=1.5...",
                    backend, mask,
                )

                guard_results = mgr.do_periodic_segmentation_rust(
                    dgrid_for_rerun,
                    atoms,
                    mask_thickness=1.5,
                    h=params.get("h", 0.2),
                    spacing=params.get("spacing", spacing),
                    return_dgrid=True,
                    threshold_abs=params.get("threshold_abs", 1.5),
                    min_distance=params.get("min_distance", 1.5),
                    return_padded_regions=True,
                    extra_padding=params.get("extra_padding", 2),
                    watershed_backend=backend,
                )

                guard_graph = make_directed_graph_from_segmentation_results(guard_results)
                guard_labels = build_base_region_labels_from_results(guard_results)
                guard_plr = find_pore_limiting_radius_winding(
                    guard_results,
                    graph=guard_graph,
                    base_region_labels=guard_labels,
                    include_dangling=include_dangling,
                    expand_components=expand_components,
                )

                if guard_plr is not None:
                    guard_pld = float(guard_plr * 2)
                    original_pld = pld
                    if guard_pld < pld - 0.1:
                        pld = guard_pld
                        # Recompute dfsd from guarded results
                        guard_dgrid = np.asarray(guard_results["dgrid"])
                        dfsd_threshold = _dfsd_threshold(guard_plr, guard_results, kwargs)
                        axis_clusters, _, _ = percolation_all_axes_winding(
                            guard_results,
                            dfsd_threshold,
                            graph=guard_graph,
                            base_region_labels=guard_labels,
                            include_dangling=include_dangling,
                            expand_components=expand_components,
                        )
                        percolating_labels = set()
                        for groups in axis_clusters.values():
                            for labels in groups:
                                percolating_labels.update(labels)
                        percolating_labels = _expand_dfsd_labels_with_window_fingerprint(
                            graph=guard_graph,
                            base_region_labels=guard_labels,
                            percolating_labels=percolating_labels,
                            results=guard_results,
                            dgrid=guard_dgrid,
                            plr=float(guard_plr),
                            kwargs=kwargs,
                            atoms=atoms,
                        )
                        dfsd = 0.0
                        if percolating_labels:
                            regions = np.asarray(guard_results["regions"])
                            rmask = np.isin(regions, list(percolating_labels))
                            if rmask.any():
                                dfsd = float(guard_dgrid[rmask].max() * 2)
                        try:
                            results["_spill_corrected"] = True
                            results["_original_pld"] = original_pld
                            results["_corrected_pld"] = pld
                        except Exception:
                            pass  # results might be read-only
                        logger.info(
                            "Spill corrected: PLD %.2f -> %.2f",
                            original_pld, pld,
                        )
                        return lcd, pld, dfsd, dgrid
                    else:
                        logger.info(
                            "Spill guardrail: no significant PLD change (%.2f vs %.2f)",
                            original_pld, guard_pld,
                        )

    # --- Compute DFSD (original path, or when no rerun needed) ---
    dfsd_threshold = _dfsd_threshold(plr, results, kwargs)
    axis_clusters, _, _ = percolation_all_axes_winding(
        results,
        dfsd_threshold,
        graph=graph,
        base_region_labels=base_region_labels,
        include_dangling=include_dangling,
        expand_components=expand_components,
    )
    percolating_labels = set()
    for groups in axis_clusters.values():
        for labels in groups:
            percolating_labels.update(labels)
    percolating_labels = _expand_dfsd_labels_with_window_fingerprint(
        graph=graph,
        base_region_labels=base_region_labels,
        percolating_labels=percolating_labels,
        results=results,
        dgrid=dgrid,
        plr=float(plr),
        kwargs=kwargs,
        atoms=mat_atoms,
    )

    dfsd = 0.0
    if percolating_labels:
        regions = np.asarray(results["regions"])
        mask = np.isin(regions, list(percolating_labels))
        if mask.any():
            dfsd = float(dgrid[mask].max() * 2)

    from mofography.percolation_results import store_pore_dia

    store_pore_dia(
        results,
        "winding",
        lcd,
        pld,
        dfsd,
        extra={"plr": float(plr)},
    )
    return lcd, pld, dfsd, dgrid



def _nearest_atom_distance(
    frac_pos: np.ndarray, atoms, cell_matrix: np.ndarray
) -> float:
    """Return distance from fractional position to nearest atom (PBC-aware)."""
    atom_frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    dr_frac = atom_frac - frac_pos.reshape(1, 3)
    dr_frac -= np.round(dr_frac)
    dr_cart = np.einsum("ni,ij->nj", dr_frac, cell_matrix)
    dist = np.linalg.norm(dr_cart, axis=1)
    return float(dist.min())


def _prune_spill_edges(
    graph: nx.MultiDiGraph,
    results: Dict[str, object],
    atoms,
    spill_margin: float = 2.0,
) -> nx.MultiDiGraph:
    """Remove edges whose window centers are too far from any framework atom.

    A real constriction must have framework atoms within ``wr + max_vdw`` of
    the window center.  Spill-over windows float in empty pore space with
    nearest atoms far away.  The check is:

        nearest_atom_dist > wr_min + spill_margin  =>  spill artifact

    Parameters
    ----------
    graph : MultiDiGraph
        Pore graph with edge data containing ``window_radius_min`` and
        ``window_positions``.
    results : dict
        Segmentation results (used for dgrid shape).
    atoms : ase.Atoms
        Framework atoms.
    spill_margin : float
        Extra tolerance in Angstroms beyond the window radius.  Accounts for
        VDW radii and grid discretization.  Default 2.0 Å covers
        max(CRC_vdw) ≈ 1.7 Å + 0.3 Å headroom.

    Returns
    -------
    graph : MultiDiGraph
        The *same* graph object with spill edges removed.
    """
    from ase.geometry import complete_cell

    dgrid = np.asarray(results.get("dgrid"))
    if dgrid.ndim != 3 or atoms is None:
        return graph
    shape = tuple(int(x) for x in dgrid.shape)
    cell_matrix = complete_cell(atoms.get_cell()).T

    edges_to_remove = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        wr_min = float(data.get("window_radius_min", 0))
        frac = _window_frac_from_edge_data(data, shape)
        if frac is None:
            continue
        nearest = _nearest_atom_distance(np.asarray(frac), atoms, cell_matrix)
        if nearest > wr_min + spill_margin:
            edges_to_remove.append((u, v, k))

    for u, v, k in edges_to_remove:
        graph.remove_edge(u, v, k)

    import logging
    logger = logging.getLogger(__name__)
    if edges_to_remove:
        logger.info(
            "Spill-edge pruning: removed %d edges (no resegmentation)",
            len(edges_to_remove),
        )

    return graph


def get_pore_dia_mgr_winding3(
    mat_atoms=None,
    dgrid: Optional[np.ndarray] = None,
    kwargs: Optional[Dict[str, object]] = None,
    results: Optional[Dict[str, object]] = None,
    include_dangling: bool = True,
    expand_components: bool = True,
    spill_margin: float = 2.0,
) -> Tuple[float, float, float, np.ndarray]:
    """Compute pore diameters with spill-edge pruning (no resegmentation).

    Identical to ``get_pore_dia_mgr_winding`` but instead of re-segmenting
    with a thicker mask, this version detects and **removes** spill-over edges
    from the pore graph.  A real constriction must have framework atoms near
    the window center; spill artifacts float in empty pore space.

    Detection: for each edge, compute PBC-aware distance from the window
    center to the nearest framework atom.  If
    ``nearest_atom_dist > wr_min + spill_margin``, the edge is a spill
    artifact and is pruned.

    Parameters
    ----------
    mat_atoms : ase.Atoms, optional
        ASE Atoms object. Required if *results* is None.
    dgrid : ndarray, optional
        Pre-computed distance grid.
    kwargs : dict, optional
        Keyword arguments for ``do_periodic_segmentation_rust``.
    results : dict, optional
        Pre-computed segmentation results.
    include_dangling : bool
        Whether to include dangling edges.
    expand_components : bool
        Whether to expand components.
    spill_margin : float
        Extra tolerance (Å) beyond window radius for spill detection.
        Default 2.0 Å covers max(CRC_vdw) + headroom.

    Returns
    -------
    tuple[float, float, float, ndarray]
        (lcd, pld, dfsd, dgrid)
    """
    if kwargs is None:
        kwargs = {}

    spacing = kwargs.get("spacing", 0.5)
    if results is None:
        if dgrid is None:
            if mat_atoms is None:
                raise ValueError("mat_atoms or results must be provided.")
            dgrid = mgr.dgrid_from_atoms_cpu(mat_atoms, spacing=spacing, block_size=50000)
        results = mgr.do_periodic_segmentation_rust(
            dgrid,
            mat_atoms,
            mask_thickness=kwargs.get("mask_thickness", 0.05),
            h=kwargs.get("h", 0.2),
            spacing=spacing,
            return_dgrid=True,
            threshold_abs=kwargs.get("threshold_abs", 1.5),
            min_distance=kwargs.get("min_distance", 1.5),
            return_padded_regions=True,
            extra_padding=kwargs.get("extra_padding", 2),
            watershed_backend=kwargs.get("watershed_backend", "rust_det"),
        )
    else:
        dgrid = np.asarray(results["dgrid"])

    # Resolve atoms for spill detection
    atoms = mat_atoms
    if atoms is None and isinstance(results, dict):
        atoms = results.get("mat_atoms")

    graph = make_directed_graph_from_segmentation_results(results)
    base_region_labels = build_base_region_labels_from_results(results)

    # --- Spill-edge pruning (before PLR computation) ---
    if atoms is not None:
        graph = _prune_spill_edges(graph, results, atoms, spill_margin=spill_margin)

    lcd = float(dgrid.max() * 2)
    plr = find_pore_limiting_radius_winding(
        results,
        graph=graph,
        base_region_labels=base_region_labels,
        include_dangling=include_dangling,
        expand_components=expand_components,
    )
    if plr is None:
        return lcd, 0.0, 0.0, dgrid

    pld = float(plr * 2)

    # --- Compute DFSD ---
    dfsd_threshold = _dfsd_threshold(plr, results, kwargs)
    axis_clusters, _, _ = percolation_all_axes_winding(
        results,
        dfsd_threshold,
        graph=graph,
        base_region_labels=base_region_labels,
        include_dangling=include_dangling,
        expand_components=expand_components,
    )
    percolating_labels = set()
    for groups in axis_clusters.values():
        for labels in groups:
            percolating_labels.update(labels)
    percolating_labels = _expand_dfsd_labels_with_window_fingerprint(
        graph=graph,
        base_region_labels=base_region_labels,
        percolating_labels=percolating_labels,
        results=results,
        dgrid=dgrid,
        plr=float(plr),
        kwargs=kwargs,
        atoms=atoms,
    )

    dfsd = 0.0
    if percolating_labels:
        regions = np.asarray(results["regions"])
        mask = np.isin(regions, list(percolating_labels))
        if mask.any():
            dfsd = float(dgrid[mask].max() * 2)

    return lcd, pld, dfsd, dgrid

CORNER_LOOP_GRID = np.array(
    [
        [0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0],
    ],
    dtype=int,
)


STRIPED_WRAP_GRID = np.array(
    [
        [0, 1, 0, 1, 0,  0, 1, 0, 0],
        [0, 1, 1, 1, 0,  0, 1, 0, 0],
        [0, 0, 0, 0, 0,  0, 1, 0, 0],
        [1, 1, 1, 1, 0,  0, 1, 1, 1],
        [0, 0, 0, 1, 0,  0, 0, 0, 0],
        [1, 1, 0, 1, 0,  0, 1, 1, 1],
        [0, 1, 0, 1, 0,  0, 1, 0, 0],
        [0, 1, 0, 1, 0,  0, 1, 0, 0],
        [0, 1, 0, 1, 0,  0, 1, 0, 0],
    ],
    dtype=int,
)


def make_periodic_graph_from_binary_grid(
    grid: np.ndarray,
    allow_diagonals: bool = False,
    node_radius: float = 1.0,
    edge_radius: float = 1.0,
) -> nx.MultiDiGraph:
    """
    Convert a binary 2D lattice into the directed periodic graph expected by the
    winding-number percolation routines.

    Parameters
    ----------
    grid : np.ndarray
        2D array where ``1`` (or truthy) entries represent pore voxels.  The
        lattice is treated as periodic along both axes.
    allow_diagonals : bool
        Include diagonal neighbors (8-connectivity) when ``True``.  Otherwise,
        only orthogonal neighbors (4-connectivity) are considered.
    node_radius, edge_radius : float
        Values stored on each node/edge so that downstream helpers expecting
        radii attributes continue to work.
    """

    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("Binary grid must be a 2D array.")

    rows, cols = grid.shape
    node_ids: Dict[Tuple[int, int], int] = {}
    G = nx.MultiDiGraph()

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 1:
                continue
            node_id = len(node_ids)
            node_ids[(r, c)] = node_id
            G.add_node(
                node_id,
                frac_pos=(r / rows, c / cols, 0.0),
                pocket_radius=node_radius,
            )

    base_dirs = [(0, 1), (1, 0)]
    if allow_diagonals:
        base_dirs.extend([(1, 1), (1, -1)])

    for (r, c), node_id in node_ids.items():
        for dr, dc in base_dirs:
            raw_r = r + dr
            raw_c = c + dc
            wrap_r = -1 if raw_r < 0 else (1 if raw_r >= rows else 0)
            wrap_c = -1 if raw_c < 0 else (1 if raw_c >= cols else 0)
            nbr_r = raw_r % rows
            nbr_c = raw_c % cols
            nbr_id = node_ids.get((nbr_r, nbr_c))
            if nbr_id is None:
                continue

            rel = (wrap_r, wrap_c, 0)
            u, v = node_id, nbr_id
            if u <= v:
                final_rel = rel
            else:
                u, v = v, u
                final_rel = tuple(-x for x in rel)
            G.add_edge(u, v, relative_offset=final_rel, window_radius=edge_radius)

    return G


def percolation_winding_on_binary_grid(
    grid: np.ndarray,
    allow_diagonals: bool = False,
    verbose: bool = False,
) -> Tuple[Dict[str, bool], List[Dict[str, object]], Dict[int, Offset]]:
    """
    Apply :func:`percolation_winding_multidigraph` to a binary periodic lattice.

    Parameters
    ----------
    grid : np.ndarray
        Binary 2D lattice treated as periodic in x/y.
    allow_diagonals : bool
        When ``True`` includes diagonal neighbors as valid connections.
    verbose : bool
        Print a human-readable summary when enabled.
    """

    G = make_periodic_graph_from_binary_grid(grid, allow_diagonals=allow_diagonals)
    percolates, loops, acc_offsets = percolation_winding_multidigraph(G)

    if verbose:
        print("Binary grid (1 = pore, 0 = empty):")
        print(np.asarray(grid, dtype=int))
        if loops:
            print("Detected winding loops:")
            for loop in loops:
                print(f"  winding={loop['winding']} nodes={loop['nodes']} edge={loop['edge']}")
        else:
            print("Detected winding loops: none (all cycles are contractible).")
        print("Per-axis percolation flags:", percolates)

    return percolates, loops, acc_offsets


#-----------------------------------------------------------------------------------------------------------------------------

def collect_periodic_edges_canonical_all_internal_windows(
    connections, maxima_info
):
    import numpy as np

    label_to_idx = {
        int(row[0]): idx 
        for idx, row in enumerate(maxima_info)
    }

    _, original_maxima_mapping, _ = _maxima_pbc_data(
        np.asarray(maxima_info)
    )

    edge_map = {}
    edges_debug = []

    for c in connections:

        if not c[1]:
            continue

        if c[2] != "internal":
            continue

        i_label, j_label = c[0]

        idx_i = label_to_idx.get(int(i_label))
        idx_j = label_to_idx.get(int(j_label))

        if idx_i is None or idx_j is None:
            continue

        base_i = int(original_maxima_mapping[idx_i])
        base_j = int(original_maxima_mapping[idx_j])

        offset_i = maxima_info[idx_i,4:7].astype(int)
        offset_j = maxima_info[idx_j,4:7].astype(int)


        start_base = base_i
        end_base = base_j

        start_offset = offset_i
        end_offset = offset_j

        if end_base < start_base:
            start_base, end_base = end_base, start_base
            start_offset, end_offset = end_offset, start_offset


        relative_offset = tuple(
            int(x) for x in (end_offset-start_offset)
        )


        if start_base == end_base:
            if relative_offset == (0,0,0):
                continue

            neg_offset = tuple(-x for x in relative_offset)

            if neg_offset < relative_offset:
                relative_offset = neg_offset


        positions = np.asarray(c[3])
        radii = np.asarray(c[4]).reshape(-1)


        if positions.ndim == 1:
            positions = positions.reshape(1,-1)


        # one edge per window
        for window_id, (pos, radius) in enumerate(
            zip(positions, radii)
        ):

            if len(pos) != 3:
                continue


            key = (
                start_base,
                end_base,
                relative_offset,
                window_id
            )


            edge_map[key] = {
                "window_radius": float(radius),
                "window_position": tuple(
                    float(x) for x in pos
                ),
            }


            edges_debug.append(
                [
                    int(i_label),
                    int(j_label),
                    start_base,
                    end_base,
                    *start_offset,
                    *end_offset,
                    *relative_offset,
                    window_id,
                    float(radius)
                ]
            )


    return (
        edge_map,
        np.array(edges_debug) if edges_debug else None
    )


def make_directed_graph_from_segmentation_results_trimmed(results, dgrid=None):

    import networkx as nx
    import numpy as np

    maxima_info = results["maxima_info"]
    connections = results["connections"]

    edge_map, edges_debug = (
        collect_periodic_edges_canonical_all_internal_windows(
            connections,
            maxima_info
        )
    )

    if dgrid is None:
        dgrid = results["dgrid"]

    dgrid_shape = dgrid.shape

    frac_coords = (
        maxima_info[:, 1:4] / dgrid_shape
        + maxima_info[:, 4:7]
    )

    G = nx.MultiDiGraph()


    # one graph edge = one window
    for (i, j, rel_offset, window_id), data in edge_map.items():

        G.add_edge(
            i,
            j,

            # periodic information
            relative_offset=rel_offset,

            # window information
            window_id=window_id,
            window_radius=float(
                data["window_radius"]
            ),
            window_position=(
                data["window_position"]
            ),
        )


    # node attributes
    for node in G.nodes():

        G.nodes[node]["frac_pos"] = np.round(
            frac_coords[node - 1],
            2
        )

        voxel_idx = tuple(
            maxima_info[node - 1,1:4].astype(int)
        )

        G.nodes[node]["pocket_radius"] = float(
            dgrid[voxel_idx]
        )


    return G


def extract_percolating_pore_graph(
    G: nx.MultiDiGraph,
    threshold: float,
    node_radius_attr: str = "pocket_radius",
    edge_radius_attr: str = "window_radius",
    expand_components: bool = True,
    include_dangling: bool = True,
):
    """
    Extract percolating pore network from a PBC pore MultiDiGraph.

    Graph convention:
    -----------------
    Node:
        PBC pore node

    Edge:
        Individual window

    Node attribute:
        pocket_radius

    Edge attribute:
        window_radius
        relative_offset


    Parameters
    ----------
    G : nx.MultiDiGraph
        Original PBC pore graph.

    threshold : float
        Minimum pore/window radius.

    expand_components : bool
        If True:
            keep the entire connected component containing
            a winding loop.

        If False:
            keep only nodes appearing in winding loops.

    include_dangling : bool
        If True:
            keep dangling pores inside the percolating component.

        If False:
            remove dangling branches using k-core.


    Returns
    -------
    P : nx.MultiDiGraph
        Extracted percolating pore graph.

    H : nx.MultiDiGraph
        Threshold-filtered accessible graph.

    """

    import networkx as nx


    # =====================================================
    # Step 1: threshold filtering
    # =====================================================

    H = nx.MultiDiGraph()


    eligible_nodes = {
        n
        for n, data in G.nodes(data=True)
        if float(data.get(node_radius_attr, 0.0))
        >= threshold
    }


    if not eligible_nodes:
        return nx.MultiDiGraph()


    # add nodes
    for n in eligible_nodes:
        H.add_node(
            n,
            **G.nodes[n]
        )


    # add windows
    for u, v, key, data in G.edges(
        keys=True,
        data=True
    ):

        if u not in eligible_nodes or v not in eligible_nodes:
            continue


        window_radius = float(
            data.get(edge_radius_attr, 0.0)
        )

        if window_radius < threshold:
            continue


        H.add_edge(
            u,
            v,
            key=key,
            **data
        )


    if H.number_of_edges() == 0:
        return nx.MultiDiGraph()



    # =====================================================
    # Step 2: detect winding percolation
    # =====================================================

    percolates, loops, acc_offsets = (
        percolation_winding_multidigraph(H)
    )


    if not any(percolates.values()):
        return nx.MultiDiGraph()



    # =====================================================
    # Step 3: collect winding nodes
    # =====================================================

    winding_nodes = set()

    for loop in loops:
        winding_nodes.update(
            loop["nodes"]
        )


    if not winding_nodes:
        return nx.MultiDiGraph()



    # =====================================================
    # Step 4: expand to percolating components
    # =====================================================

    UG = nx.Graph(
        H.to_undirected(as_view=True)
    )


    if expand_components:

        keep_nodes = set()

        for comp in nx.connected_components(UG):

            comp = set(comp)

            if comp.intersection(winding_nodes):
                keep_nodes.update(comp)

    else:

        keep_nodes = winding_nodes



    # =====================================================
    # Step 5: remove dangling branches if requested
    # =====================================================

    if not include_dangling:

        sub = UG.subgraph(keep_nodes).copy()

        # Self-loops should not affect ordinary branch degree
        sub.remove_edges_from(
            list(nx.selfloop_edges(sub))
        )

        # Iteratively remove non-winding leaf nodes
        while True:

            dangling_nodes = [
                n
                for n in sub.nodes()
                if n not in winding_nodes
                and sub.degree(n) <= 1
            ]

            if not dangling_nodes:
                break

            sub.remove_nodes_from(dangling_nodes)

        keep_nodes = set(sub.nodes())



    # =====================================================
    # Step 6: build final graph
    # =====================================================

    percolation_pore_graph = H.subgraph(
        keep_nodes
    ).copy()


    return percolation_pore_graph
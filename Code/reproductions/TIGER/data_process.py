"""Data-processing for TIGER reproduction.

Direct port of upstream ``data_process.py`` (Blair1213/TIGER). Algorithm
preserved; only cosmetic adjustments and explicit ``cache_dir`` parameter so
intermediate JSON files (subgraph caches, PageRank tables, molecular-graph
caches) are kept under project ``Code/data/TIGER/<dataset>/`` rather than
upstream's ``./data/<dataset>/``.

Key paper-section mapping:
  * :func:`smile_to_graph` / :func:`single_smile_to_graph` — MG channel
    pre-processing (paper Sec "Molecular Graph Channel"). Atom features =
    67-d (44 element one-hot + 11 H-count + 11 implicit valence + 1
    aromatic). Bonds encoded via ``e_map['bond_type']``. Shortest-path
    relation augmentation adds relation IDs >=23 for non-adjacent atoms.
  * :func:`subtreeExtractor` — paper Sec "k-subtree-based Extractor" (Sec
    "Biomedical Knowledge Graph Channel").
  * :func:`probExtractor` — paper Sec "Probability-based Extractor".
  * :func:`rwExtractor` — paper Sec "DeepWalk-based Extractor".
  * Shortest-path augmentation inside each extractor: relation IDs >= num_rel
    are reserved for the SP-augmented edges (sp_value as relation offset).

NOTE upstream quirks preserved verbatim:
  * Original ``probExtractor`` ``print(subsets)`` debug noise — REMOVED here
    (would flood logs).
  * ``os.mkdir(paths)`` → ``os.makedirs(paths, exist_ok=True)`` (otherwise
    fails when parent ``data/<dataset>/`` doesn't exist).
  * ``np.asmatrix`` (deprecated NumPy >=1.20) → ``np.asarray`` in
    :func:`google_matrix` (same behaviour for our use case; we never rely on
    matrix-class semantics).
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from torch import Tensor
from torch_geometric.utils import degree, subgraph

from randomwalk import Node2vec

# Bond-type / stereo vocabulary mirrors upstream e_map (used to look up
# rel_index for each molecular bond).
e_map = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [False, True],
}


def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic["X"] = (max_value + min_value) / 2.0
    return dic


def atom_features(atom):
    """44-d element one-hot + 11-d H-count + 11-d implicit valence + 1-d
    aromatic = 67-d feature. Matches upstream ``atom_features`` exactly."""
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
                "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn",
                "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au",
                "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "X",
            ],
        )
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + [atom.GetIsAromatic()]
    ), atom.GetDegree()


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}:")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(cache_dir: str, ligands: dict) -> Tuple[dict, int, int]:
    """Build the cache mapping drug-id → (c_size, features, edge_index,
    rel_index, sp_edge_index, sp_value, sp_rel, max_degree).

    Drugs with single atom (c_size=0 in :func:`single_smile_to_graph`) are
    dropped — matches paper "remove the data items that cannot be converted
    into graphs from SMILES strings" caveat.
    """
    # Per CLAUDE.md §"复现代码 (Reproduction) 规范" §2: the upstream-shipped
    # mol_sp.json lives at ``<cache_dir>/necessary/mol_sp__official.json``;
    # any locally-built version would be ``mol_sp__mine.json`` (priority:
    # official > mine; error if neither). Legacy flat ``<cache_dir>/mol_sp.json``
    # is tried last for back-compat with older checkouts.
    smile_graph: dict = {}
    necessary_dir = os.path.join(cache_dir, "necessary")
    official_path = os.path.join(necessary_dir, "mol_sp__official.json")
    mine_path = os.path.join(necessary_dir, "mol_sp__mine.json")
    legacy_path = os.path.join(cache_dir, "mol_sp.json")
    cache_path = None
    source_tag = None
    for candidate, tag in (
        (official_path, "OFFICIAL"),
        (mine_path, "MINE (official missing)"),
        (legacy_path, "LEGACY (flat path, pre-2026-05-18 layout)"),
    ):
        if os.path.exists(candidate):
            cache_path = candidate
            source_tag = tag
            break

    if cache_path is not None:
        print(f"[mol_sp] USING {source_tag}: {cache_path}", file=__import__("sys").stderr)
        with open(cache_path, "r") as f:
            smile_graph = json.load(f)
        max_rel = 0
        max_degree = 0
        for s in smile_graph.keys():
            max_rel = max(smile_graph[s][6]) if max(smile_graph[s][6]) > max_rel else max_rel
            max_degree = (
                smile_graph[s][7] if smile_graph[s][7] > max_degree else max_degree
            )
        return smile_graph, max_rel, max_degree

    smiles_max_node_degree = []
    num_rel_mol_update = 0
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]))
        c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, deg = single_smile_to_graph(
            lg
        )
        if c_size == 0:
            continue
        if max(s_value) > num_rel_mol_update:
            num_rel_mol_update = max(s_value)
        smile_graph[d] = (
            c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, deg,
        )
        smiles_max_node_degree.append(deg)

    # No cache hit on any of {__official, __mine, legacy}: build from
    # scratch and write to ``__mine`` per CLAUDE.md §"复现代码规范" §2
    # naming (any version WE build is `__mine`, never overwriting
    # `__official`).
    os.makedirs(necessary_dir, exist_ok=True)
    with open(mine_path, "w") as f:
        json.dump(smile_graph, f)
    print(
        f"[mol_sp] BUILT NEW (no official/mine cache) → wrote {mine_path}",
        file=__import__("sys").stderr,
    )

    return smile_graph, num_rel_mol_update, max(smiles_max_node_degree)


def single_smile_to_graph(smile: str):
    """SMILES → graph tuple. Returns 0-tuple for single-atom molecules
    (no bonds → no MG channel input)."""
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    degrees = []
    for atom in mol.GetAtoms():
        feature, deg = atom_features(atom)
        features.append((feature / sum(feature)).tolist())
        degrees.append(deg)

    mol_index = []
    for bond in mol.GetBonds():
        mol_index.append(
            [
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                e_map["bond_type"].index(str(bond.GetBondType())),
            ]
        )
        mol_index.append(
            [
                bond.GetEndAtomIdx(),
                bond.GetBeginAtomIdx(),
                e_map["bond_type"].index(str(bond.GetBondType())),
            ]
        )

    if len(mol_index) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    mol_index = np.array(sorted(mol_index))
    mol_edge_index = mol_index[:, :2]
    mol_rel_index = mol_index[:, 2]

    # SP-augmented edges (paper Sec "Relation-aware Self-Attention" — connect
    # distant nodes via their SP relation).
    s_edge_index_value = calculate_shortest_path(mol_edge_index)
    s_edge_index = s_edge_index_value[:, :2]
    s_value = s_edge_index_value[:, 2]
    # IMPORTANT — match upstream verbatim: ``s_rel = s_value`` is an ALIAS,
    # not a copy. Subsequent in-place writes mutate both arrays, and the
    # second ``np.where`` runs against the ALREADY-MUTATED ``s_value``. This
    # is upstream behaviour (presumably what produced the paper numbers); we
    # preserve it for bit-faithfulness even though it is almost certainly a
    # bug — see ``_paper-and-GitHub/Blair1213__TIGER/data_process.py:161-163``.
    s_rel = s_value  # ALIAS (intentional; see note above)
    s_rel[np.where(s_value == 1)] = mol_rel_index
    s_rel[np.where(s_value != 1)] += 23

    assert len(s_edge_index) == len(s_value)
    assert len(s_edge_index) == len(s_rel)

    return (
        c_size,
        features,
        mol_edge_index.tolist(),
        mol_rel_index.tolist(),
        s_edge_index.tolist(),
        s_value.tolist(),
        s_rel.tolist(),
        max(degrees),
    )


def calculate_shortest_path(edge_index):
    """All-pairs shortest path lengths over a directed graph built from
    ``edge_index`` (shape [E, 2]). Returns sorted (src, dst, length) triplets."""
    s_edge_index_value = []
    g = nx.DiGraph()
    g.add_edges_from(edge_index.tolist())
    paths = nx.all_pairs_shortest_path_length(g)
    for node_i, node_ij in paths:
        for node_j, length_ij in node_ij.items():
            s_edge_index_value.append([node_i, node_j, length_ij])
    s_edge_index_value.sort()
    return np.array(s_edge_index_value)


def read_interactions(path: str, drug_dict: dict):
    """Read paper-format ``ddi.txt`` (lines = ``drug1 drug2 rel label`` with
    pos/neg = 1:1 balanced). Returns (interactions array, drug set).

    Asserts ``negative_num == positive_num`` (matches upstream;
    upstream-provided dataset files satisfy this constraint).
    """
    interactions = []
    all_drug_in_ddi = []
    positive_drug_inter_dict: dict = {}
    positive_num = 0
    negative_num = 0
    with open(path, "r") as f:
        for line in f.readlines():
            drug1_id, drug2_id, rel, label = line.strip().split(" ")[:4]
            if drug1_id in drug_dict and drug2_id in drug_dict:
                all_drug_in_ddi.append(drug1_id)
                all_drug_in_ddi.append(drug2_id)
                if float(label) > 0:
                    positive_num += 1
                else:
                    negative_num += 1
                if drug1_id in positive_drug_inter_dict:
                    if drug2_id not in positive_drug_inter_dict[drug1_id]:
                        positive_drug_inter_dict[drug1_id].append(drug2_id)
                        interactions.append([int(drug1_id), int(drug2_id), int(rel), int(label)])
                else:
                    positive_drug_inter_dict[drug1_id] = [drug2_id]
                    interactions.append([int(drug1_id), int(drug2_id), int(rel), int(label)])

    print(f"  positive interactions: {positive_num}")
    print(f"  negative interactions: {negative_num}")
    assert negative_num == positive_num, (
        "TIGER paper assumes balanced pos:neg = 1:1 in ddi.txt"
    )
    return np.array(interactions, dtype=int), set(all_drug_in_ddi)


def read_network(path: str):
    """Parse paper-format ``networks.txt`` (first line = header, then
    ``head tail rel`` per line). Returns (num_node, edge_index, rel_index,
    num_rel)."""
    edge_index = []
    rel_index = []
    flag = 0
    with open(path, "r") as f:
        for line in f.readlines():
            if flag == 0:
                flag = 1
                continue
            flag += 1
            head, tail, rel = line.strip().split(" ")[:3]
            edge_index.append([int(head), int(tail)])
            rel_index.append(int(rel))

    num_node = int(np.max(np.array(edge_index)))
    num_rel = max(rel_index) + 1
    print(f"  unique relations in network: {len(list(set(rel_index)))}")
    return num_node, edge_index, rel_index, num_rel


def read_smiles(path: str) -> dict:
    """Parse paper-format ``drug_smiles.txt`` (header + ``id\\tsmiles``)."""
    print(f"Read {path}!")
    flag = 0
    out: dict = {}
    with open(path, "r") as f:
        for line in f.readlines():
            if flag == 0:
                flag += 1
                continue
            id_, sequence = line.strip().split("\t")
            if id_ not in out:
                out[id_] = sequence
    return out


def generate_node_subgraphs(
    dataset: str,
    drug_id,
    network_edge_index,
    network_rel_index,
    num_rel: int,
    args,
    cache_root: str,
):
    """Dispatch to one of the 3 extractors per ``args.extractor``. Cache the
    JSON output under ``cache_root/<dataset>/<method>/``."""
    method = args.extractor
    edge_index = torch.from_numpy(np.array(network_edge_index).T)  # [2, E]
    rel_index = torch.from_numpy(np.array(network_rel_index))

    row, col = edge_index
    reverse_edge_index = torch.stack((col, row), 0)
    undirected_edge_index = torch.cat((edge_index, reverse_edge_index), 1)

    paths = os.path.join(cache_root, str(dataset), str(method)) + os.sep
    os.makedirs(paths, exist_ok=True)

    if method == "khop-subtree":
        return subtreeExtractor(
            drug_id, undirected_edge_index, rel_index, paths, num_rel,
            fixed_num=args.fixed_num, khop=args.khop,
        )
    if method == "probability":
        pagerank_paths = os.path.join(paths, "pageRank.json")
        return probExtractor(
            drug_id, undirected_edge_index, rel_index, paths, num_rel,
            fixed_num=args.fixed_num, pagerank_path=pagerank_paths,
        )
    if method == "randomWalk":
        return rwExtractor(
            drug_id, undirected_edge_index, rel_index, paths, num_rel,
            sub_num=args.graph_fixed_num, length=args.fixed_num,
        )
    raise ValueError(f"unknown extractor: {method}")


def subtreeExtractor(drug_id, edge_index, rel_index, shortest_paths, num_rel, fixed_num, khop):
    """k-subtree subgraph extractor (paper Sec "k-subtree-based Extractor"):
    BFS to depth ``khop``, with at most ``fixed_num`` children sampled per
    node (without replacement).
    """
    all_degree = []
    num_rel_update = []
    subgraphs: dict = {}

    json_path = (
        shortest_paths + "subtree_fixed_" + str(fixed_num) + "_hop_" + str(khop) + "sp.json"
    )
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            subgraphs = json.load(f)
        max_rel = 0
        max_degree = 0
        for s in subgraphs.keys():
            max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
            max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree
        return subgraphs, max_degree, max_rel

    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    for d in drug_id:
        subset, sub_edge_index, sub_rel_index, mapping_list = k_hop_subgraph(
            int(d), khop, edge_index, undirected_rel_index, fixed_num, relabel_nodes=True
        )
        row, col = sub_edge_index
        all_degree.append(torch.max(degree(col)).item())

        new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()
        node_idx = subset.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:
                continue
            s_edge_index.append(sp_edge_index[i].tolist())
            s_value.append(sp_value[i])
            s_rel.append(sp_value[i] + num_rel)

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)
        num_rel_update.append(np.max(s_rel))

        subgraphs[d] = (
            node_idx, new_s_edge_index, new_s_rel, mapping_list,
            s_edge_index, s_value, s_rel, torch.max(degree(col)).item(),
        )

    with open(json_path, "w") as f:
        json.dump(subgraphs, f, default=convert)

    return subgraphs, max(all_degree), max(num_rel_update)


def probExtractor(drug_id, edge_index, rel_index, shortest_paths, num_rel, fixed_num, pagerank_path):
    """PageRank-probability subgraph extractor (paper Sec "Probability-based
    Extractor"): sample ``fixed_num`` nodes weighted by PageRank vector
    seeded at the drug node.
    """
    json_path = shortest_paths + "prob_fix_" + str(fixed_num) + "_sp.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            subgraphs = json.load(f)
        max_rel = 0
        max_degree = 0
        for s in subgraphs.keys():
            max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
            max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree
        return subgraphs, max_degree, max_rel

    g = nx.DiGraph()
    g.add_edges_from(edge_index.transpose(1, 0).tolist())

    if not os.path.exists(pagerank_path):
        pagerank = np.array(google_matrix(g), dtype="float32")
        page_dict: dict = {}
        for d in drug_id:
            page_dict[d] = list(pagerank[list(g.nodes()).index(int(d))])
        with open(pagerank_path, "w") as f:
            json.dump(page_dict, f)
    else:
        with open(pagerank_path, "r") as f:
            page_dict = json.load(f)

    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    num_rel_update = []
    max_degree = []
    subgraphs = {}
    for d in drug_id:
        subsets = [int(d)]

        neighbors = np.random.choice(
            a=list(g.nodes()),
            size=fixed_num,
            replace=False,
            p=page_dict[d],
        )
        subsets.extend(neighbors)
        subsets = list(set(subsets))

        mapping_list = [False for _ in subsets]
        mapping_idx = subsets.index(int(d))
        mapping_list[mapping_idx] = True

        sub_edge_index, sub_rel_index = subgraph(
            subsets, edge_index, undirected_rel_index, relabel_nodes=True
        )
        _row_sub, col_sub = sub_edge_index
        new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:
                continue
            s_edge_index.append(sp_edge_index[i].tolist())
            s_value.append(sp_value[i])
            s_rel.append(sp_value[i] + num_rel)

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)
        num_rel_update.append(int(np.max(s_rel)))
        max_degree.append(torch.max(degree(col_sub)).item())

        subgraphs[d] = (
            subsets, new_s_edge_index, new_s_rel, mapping_list,
            s_edge_index, s_value, s_rel, torch.max(degree(col_sub)).item(),
        )

    with open(json_path, "w") as f:
        json.dump(subgraphs, f, default=convert)

    return subgraphs, max(max_degree), max(num_rel_update)


def rwExtractor(drug_id, edge_index, rel_index, shortest_paths, num_rel, sub_num, length):
    """DeepWalk subgraph extractor (paper Sec "DeepWalk-based Extractor"):
    run :class:`Node2vec` with ``dw=True``. NOTE the underlying
    :class:`randomwalk.walker.BasicWalker` returns ``list(set(walks))`` —
    sequence order destroyed, true subgraph size may be <``length``.
    """
    json_path = shortest_paths + "rw_num_" + str(sub_num) + "_length_" + str(length) + "sp.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            subgraphs = json.load(f)
        max_rel = 0
        max_degree = 0
        for s in subgraphs.keys():
            max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
            max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree
        return subgraphs, max_degree, max_rel

    my_graph = nx.Graph()
    my_graph.add_edges_from(edge_index.transpose(1, 0).numpy().tolist())
    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    num_rel_update = []
    max_degree = []
    subgraphs = {}
    for d in drug_id:
        subsets = Node2vec(
            start_nodes=[int(d)], graph=my_graph,
            path_length=length, num_paths=sub_num, workers=6, dw=True,
        ).get_walks()
        mapping_id = subsets.index(int(d))
        mapping_list = [False for _ in range(len(subsets))]
        mapping_list[mapping_id] = True

        sub_edge_index, sub_rel_index = subgraph(
            subsets, edge_index, undirected_rel_index, relabel_nodes=True
        )
        _row_sub, col_sub = sub_edge_index
        new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:
                continue
            s_edge_index.append(sp_edge_index[i].tolist())
            s_value.append(sp_value[i])
            s_rel.append(sp_value[i] + num_rel)

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)
        num_rel_update.append(int(np.max(s_rel)))
        max_degree.append(torch.max(degree(col_sub)).item())

        subgraphs[d] = (
            subsets, new_s_edge_index, new_s_rel, mapping_list,
            s_edge_index, s_value, s_rel, torch.max(degree(col_sub)).item(),
        )

    with open(json_path, "w") as f:
        json.dump(subgraphs, f, default=convert)

    return subgraphs, max(max_degree), max(num_rel_update)


def k_hop_subgraph(
    node_idx,
    num_hops: int,
    edge_index,
    rel_index,
    fixed_num,
    relabel_nodes: bool = False,
    num_nodes=None,
    flow: str = "source_to_target",
):
    """k-hop BFS with optional per-hop fan-out cap. Matches upstream
    behaviour incl. ``np.random.seed(42)`` for deterministic sampling."""
    np.random.seed(42)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ["source_to_target", "target_to_source"]
    if flow == "target_to_source":
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        if fixed_num is None:
            subsets.append(col[edge_mask])
        elif col[edge_mask].size(0) > fixed_num:
            neighbors = np.random.choice(a=col[edge_mask].numpy(), size=fixed_num, replace=False)
            subsets.append(torch.LongTensor(neighbors))
        else:
            subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[: node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx_remap = row.new_full((num_nodes,), -1)
        node_idx_remap[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx_remap[edge_index]

    rel_index = rel_index[edge_mask] if rel_index is not None else None

    mapping_mask = [False for _ in range(len(subset))]
    mapping_mask[inv] = True

    return subset, edge_index, rel_index, mapping_mask


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    if isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    return max(edge_index.size(0), edge_index.size(1))


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def min_max(data: list) -> list:
    min_value = min(data)
    max_value = max(data)
    norm_data = []
    for d in data:
        norm_data.append((d - min_value + 0.00001) / (max_value - min_value))
    return [d / sum(norm_data) for d in norm_data]


def google_matrix(
    G, alpha: float = 0.85, personalization=None, nodelist=None, weight: str = "weight", dangling=None
):
    """PageRank Google matrix builder (upstream copy with ``np.asmatrix``
    replaced by ``np.asarray`` — semantics preserved for our use case)."""
    if nodelist is None:
        nodelist = list(G)

    M = np.asarray(nx.to_numpy_array(G, nodelist=nodelist, weight=weight), dtype="float32")
    N = len(G)
    if N == 0:
        return M

    if personalization is None:
        p = np.repeat(1.0 / N, N).astype("float32")
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype="float32")
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()

    if dangling is None:
        dangling_weights = p
    else:
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype="float32")
        dangling_weights /= dangling_weights.sum()
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]

    for node in dangling_nodes:
        M[node] = dangling_weights

    M /= M.sum(axis=1, keepdims=True).astype("float32")

    return np.multiply(alpha, M, dtype="float32") + np.multiply(1 - alpha, p, dtype="float32")

import os
import ot
import time
import torch
import datetime
import numpy as np
import pandas as pd
import scipy.sparse as sp

from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import OneHotEncoder

# Assuming these are available in your .utils as provided
from .utils import (
    fused_gromov_wasserstein_incent, 
    to_dense_array, 
    extract_data_matrix, 
    jensenshannon_divergence_backend, 
    pairwise_msd
)

def topo_semantic_signatures(slice_obj: AnnData, radius: float, steps: list = [1, 5, 10]) -> np.ndarray:
    """
    Computes Multi-Scale Topo-Semantic Signatures for breaking spatial symmetry.
    Uses a random walk diffusion on the spatial radius-graph over immutable cell-type labels.
    """
    coords = slice_obj.obsm['spatial']
    labels = slice_obj.obs['cell_type_annot'].values.reshape(-1, 1)
    
    # 1. One-hot encode cell types
    encoder = OneHotEncoder(sparse_output=False)
    L = encoder.fit_transform(labels)
    
    # 2. Build spatial adjacency matrix using the user-defined radius
    # mode='connectivity' gives a binary adjacency matrix
    A = radius_neighbors_graph(coords, radius=radius, mode='connectivity', include_self=True)
    
    # 3. Create Random Walk Transition Matrix P = D^-1 A
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv.dot(A)
    
    # 4. Compute diffusion states at multiple scales
    signatures = []
    current_state = L
    max_step = max(steps)
    
    for step in range(1, max_step + 1):
        current_state = P.dot(current_state) # P^t L
        if step in steps:
            # Normalize to ensure it's a valid probability distribution across cell types
            row_sums_state = current_state.sum(axis=1, keepdims=True)
            row_sums_state[row_sums_state == 0] = 1.0
            norm_state = current_state / row_sums_state
            signatures.append(norm_state)
            
    # Concatenate multi-scale features
    multi_scale_signature = np.concatenate(signatures, axis=1)
    
    # Final normalization over the concatenated feature space for JS Divergence
    row_sums_final = multi_scale_signature.sum(axis=1, keepdims=True)
    row_sums_final[row_sums_final == 0] = 1.0
    
    return multi_scale_signature / row_sums_final


def cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep=None, use_gpu=False, nx=ot.backend.NumpyBackend(), beta=0.8, overwrite=False):
    """Original cosine distance function for gene expression."""
    A_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, use_rep)))
    B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceB, use_rep)))

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()
   
    s_A = A_X + 0.01
    s_B = B_X + 0.01

    fileName = f"{filePath}/cosine_dist_gene_expr_{sliceA_name}_{sliceB_name}.npy"
    
    if os.path.exists(fileName) and not overwrite:
        print("Loading precomputed Cosine distance of gene expression for slice A and slice B")
        cosine_dist_gene_expr = np.load(fileName)
    else:
        print("Calculating cosine dist of gene expression for slice A and slice B")
        if isinstance(s_A, torch.Tensor):
            s_A = s_A.detach().cpu().numpy()
        else:
            s_A = np.asarray(s_A)
        if isinstance(s_B, torch.Tensor):
            s_B = s_B.detach().cpu().numpy()
        else:
            s_B = np.asarray(s_B)
        cosine_dist_gene_expr = cosine_distances(s_A, s_B)

        print("Saving cosine dist of gene expression for slice A and slice B")
        np.save(fileName, cosine_dist_gene_expr)

    return cosine_dist_gene_expr


def pairwise_align(
    sliceA: AnnData, 
    sliceB: AnnData, 
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    filePath: str,
    use_rep: Optional[str] = None, 
    G_init = None, 
    a_distribution = None, 
    b_distribution = None, 
    norm: bool = False, 
    numItermax: int = 6000, 
    backend = ot.backend.TorchBackend(),
    use_gpu: bool = False, 
    return_obj: bool = False,
    verbose: bool = False, 
    gpu_verbose: bool = True, 
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite = False,
    neighborhood_dissimilarity: str='jsd',
    dummy_cell: bool = True,
    **kwargs) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], float, float, float, float]]:
    
    start_time = time.time()

    if not os.path.exists(filePath):
        os.makedirs(filePath)

    logFile = open(f"{filePath}/log.txt", "w")
    logFile.write(f"pairwise_align_INCENT (Topo-Semantic Truncated)\n")
    currDateTime = datetime.datetime.now()
    logFile.write(f"{currDateTime}\n")
    logFile.write(f"sliceA_name: {sliceA_name}, sliceB_name: {sliceB_name}\n")
    logFile.write(f"alpha: {alpha}, beta: {beta}, gamma: {gamma}, radius: {radius}\n")

    if use_gpu:
        if isinstance(backend,ot.backend.TorchBackend):
            if torch.cuda.is_available():
                if gpu_verbose: print("gpu is available, using gpu.")
            else:
                if gpu_verbose: print("gpu is not available, resorting to torch cpu.")
                use_gpu = False
        else:
            if torch.cuda.is_available():
                if gpu_verbose:
                    print("Switching to `ot.backend.TorchBackend()` for GPU execution.")
                backend = ot.backend.TorchBackend()
            else:
                print("We currently only have gpu support for Pytorch, please set backend = ot.backend.TorchBackend(). Reverting to selected backend cpu.")
                use_gpu = False
    else:
        if gpu_verbose: print("Using selected backend cpu. If you want to use gpu, set use_gpu = True.")

    if not torch.cuda.is_available():
        use_gpu = False

    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Found empty `AnnData`:\n{s}.")

    nx = backend

    # Filter to shared genes and cell types
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    sliceA = sliceA[:, shared_genes]
    sliceB = sliceB[:, shared_genes]

    shared_cell_types = pd.Index(sliceA.obs['cell_type_annot']).unique().intersection(pd.Index(sliceB.obs['cell_type_annot']).unique())
    sliceA = sliceA[sliceA.obs['cell_type_annot'].isin(shared_cell_types)]
    sliceB = sliceB[sliceB.obs['cell_type_annot'].isin(shared_cell_types)]
    
    # ── 1. Truncated Spatial Distances ──────────────────────────────
    coordinatesA = sliceA.obsm['spatial'].copy()
    coordinatesB = sliceB.obsm['spatial'].copy()
    
    # Calculate raw euclidean
    D_A_raw = ot.dist(coordinatesA, coordinatesA, metric='euclidean')
    D_B_raw = ot.dist(coordinatesB, coordinatesB, metric='euclidean')
    
    # Auto-infer Truncation Threshold (tau) based on the diameter of the smaller slice
    tau = min(np.max(D_A_raw), np.max(D_B_raw))
    logFile.write(f"[Truncated Gromov] Applying spatial distance truncation at tau = {tau:.4f}\n")
    
    # Apply thresholding
    D_A_np = np.clip(D_A_raw, 0, tau)
    D_B_np = np.clip(D_B_raw, 0, tau)
    
    D_A = nx.from_numpy(D_A_np)
    D_B = nx.from_numpy(D_B_np)

    if isinstance(nx,ot.backend.TorchBackend):
        D_A = D_A.double()
        D_B = D_B.double()
        if use_gpu:
            D_A = D_A.cuda()
            D_B = D_B.cuda()

    # ── 2. Gene Expression & Cell Type Penalty (M1) ───────────────────────
    cosine_dist_gene_expr = cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep=use_rep, use_gpu=use_gpu, nx=nx, beta=beta, overwrite=overwrite)

    _lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    _lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)
    M_celltype = (_lab_A[:, None] != _lab_B[None, :]).astype(np.float64)
    M1_combined = (1 - beta) * cosine_dist_gene_expr + beta * M_celltype
    M1 = nx.from_numpy(M1_combined)

    # ── 3. Topo-Semantic Signatures (M2) ──────────────────────────────────
    file_suffix = f"topo_signature_{radius}"
    if os.path.exists(f"{filePath}/{file_suffix}_{sliceA_name}.npy") and not overwrite:
        print("Loading precomputed Topo-Semantic distribution of slice A")
        neighborhood_distribution_sliceA = np.load(f"{filePath}/{file_suffix}_{sliceA_name}.npy")
    else:
        print("Calculating Topo-Semantic distribution of slice A")
        neighborhood_distribution_sliceA = topo_semantic_signatures(sliceA, radius=radius)
        # np.save(f"{filePath}/{file_suffix}_{sliceA_name}.npy", neighborhood_distribution_sliceA)

    if os.path.exists(f"{filePath}/{file_suffix}_{sliceB_name}.npy") and not overwrite:
        print("Loading precomputed Topo-Semantic distribution of slice B")
        neighborhood_distribution_sliceB = np.load(f"{filePath}/{file_suffix}_{sliceB_name}.npy")
    else:
        print("Calculating Topo-Semantic distribution of slice B")
        neighborhood_distribution_sliceB = topo_semantic_signatures(sliceB, radius=radius)
        # np.save(f"{filePath}/{file_suffix}_{sliceB_name}.npy", neighborhood_distribution_sliceB)

    if ('numpy' in str(type(neighborhood_distribution_sliceA))) and use_gpu:
        neighborhood_distribution_sliceA = torch.from_numpy(neighborhood_distribution_sliceA)
    if ('numpy' in str(type(neighborhood_distribution_sliceB))) and use_gpu:
        neighborhood_distribution_sliceB = torch.from_numpy(neighborhood_distribution_sliceB)

    if use_gpu:
        neighborhood_distribution_sliceA = neighborhood_distribution_sliceA.cuda()
        neighborhood_distribution_sliceB = neighborhood_distribution_sliceB.cuda()

    if neighborhood_dissimilarity == 'jsd':
        print("Calculating JSD of Topo-Semantic distribution for slice A and slice B")
        js_dist_neighborhood = jensenshannon_divergence_backend(neighborhood_distribution_sliceA, neighborhood_distribution_sliceB)
        if isinstance(js_dist_neighborhood, torch.Tensor):
            js_dist_neighborhood = js_dist_neighborhood.detach().cpu().numpy()
        M2 = nx.from_numpy(js_dist_neighborhood)
    
    elif neighborhood_dissimilarity == 'cosine':
        ndA = np.asarray(neighborhood_distribution_sliceA.cpu() if use_gpu else neighborhood_distribution_sliceA)
        ndB = np.asarray(neighborhood_distribution_sliceB.cpu() if use_gpu else neighborhood_distribution_sliceB)
        numerator = ndA @ ndB.T
        denom = np.linalg.norm(ndA, axis=1)[:, None] * np.linalg.norm(ndB, axis=1)[None, :]
        cosine_dist_neighborhood = 1 - numerator / denom
        M2 = nx.from_numpy(cosine_dist_neighborhood)

    elif neighborhood_dissimilarity == 'msd':
        ndA = np.asarray(neighborhood_distribution_sliceA.cpu() if use_gpu else neighborhood_distribution_sliceA)
        ndB = np.asarray(neighborhood_distribution_sliceB.cpu() if use_gpu else neighborhood_distribution_sliceB)
        msd_neighborhood = pairwise_msd(ndA, ndB)
        M2 = nx.from_numpy(msd_neighborhood)

    # ── 4. Dummy Cell Augmentation ─────────────────────────────────────────
    _has_dummy_src = False
    _has_dummy_tgt = False

    if dummy_cell:
        from collections import Counter
        ns, nt = sliceA.shape[0], sliceB.shape[0]
        counts_A = Counter(_lab_A)
        counts_B = Counter(_lab_B)
        all_types = set(counts_A.keys()) | set(counts_B.keys())
        _budget = sum(max(counts_A.get(k, 0), counts_B.get(k, 0)) for k in all_types)
        _w_dummy_src = _budget - ns   
        _w_dummy_tgt = _budget - nt   
        _has_dummy_src = _w_dummy_src > 0
        _has_dummy_tgt = _w_dummy_tgt > 0

        def _to_np(x):
            try: return x.cpu().detach().numpy().astype(np.float64)
            except Exception: return np.array(x, dtype=np.float64)

        _ns_aug = ns + (1 if _has_dummy_src else 0)
        _nt_aug = nt + (1 if _has_dummy_tgt else 0)

        if _has_dummy_src:
            D_A_np = _to_np(D_A)
            D_A_aug = np.zeros((_ns_aug, _ns_aug), dtype=np.float64)
            D_A_aug[:ns, :ns] = D_A_np
            D_A = nx.from_numpy(D_A_aug)
            if isinstance(nx, ot.backend.TorchBackend): D_A = D_A.double()

        if _has_dummy_tgt:
            D_B_np = _to_np(D_B)
            D_B_aug = np.zeros((_nt_aug, _nt_aug), dtype=np.float64)
            D_B_aug[:nt, :nt] = D_B_np
            D_B = nx.from_numpy(D_B_aug)
            if isinstance(nx, ot.backend.TorchBackend): D_B = D_B.double()

        M1_np = _to_np(M1)
        M2_np = _to_np(M2)
        _type_M1_max, _type_M2_max = {}, {}
        for _type_k in all_types:
            _S = np.where(_lab_A == _type_k)[0]
            _T = np.where(_lab_B == _type_k)[0]
            if len(_S) > 0 and len(_T) > 0:
                _type_M1_max[_type_k] = float(M1_np[np.ix_(_S, _T)].max())
                _type_M2_max[_type_k] = float(M2_np[np.ix_(_S, _T)].max())
            else:
                _type_M1_max[_type_k] = float(M1_np.max())
                _type_M2_max[_type_k] = float(M2_np.max())

        _death_M1 = np.array([_type_M1_max[_lab_A[i]] for i in range(ns)], dtype=np.float64)
        _birth_M1 = np.array([_type_M1_max[_lab_B[j]] for j in range(nt)], dtype=np.float64)
        _death_M2 = np.array([_type_M2_max[_lab_A[i]] for i in range(ns)], dtype=np.float64)
        _birth_M2 = np.array([_type_M2_max[_lab_B[j]] for j in range(nt)], dtype=np.float64)

        epsilon = 1e-6

        M1_aug = np.zeros((_ns_aug, _nt_aug), dtype=np.float64)
        M1_aug[:ns, :nt] = M1_np
        if _has_dummy_tgt: M1_aug[:ns, nt] = _death_M1 + epsilon
        if _has_dummy_src: M1_aug[ns, :nt] = _birth_M1 + epsilon
        if _has_dummy_src and _has_dummy_tgt: M1_aug[ns, nt] = np.max(M1_np) + epsilon
        M1 = nx.from_numpy(M1_aug)

        M2_aug = np.zeros((_ns_aug, _nt_aug), dtype=np.float64)
        M2_aug[:ns, :nt] = M2_np
        if _has_dummy_tgt: M2_aug[:ns, nt] = _death_M2 + epsilon
        if _has_dummy_src: M2_aug[ns, :nt] = _birth_M2 + epsilon
        if _has_dummy_src and _has_dummy_tgt: M2_aug[ns, nt] = np.max(M2_np) + epsilon
        M2 = nx.from_numpy(M2_aug)

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        M1 = M1.cuda()
        M2 = M2.cuda()
        if dummy_cell and (_has_dummy_src or _has_dummy_tgt):
            D_A = D_A.cuda()
            D_B = D_B.cuda()
    
    # ── 5. Distributions & Normalization ───────────────────────────────────
    if a_distribution is None:
        if dummy_cell:
            if _has_dummy_src:
                a_vals = np.full(ns + 1, 1.0 / _budget, dtype=np.float64)
                a_vals[-1] = float(_w_dummy_src) / _budget
                a = nx.from_numpy(a_vals)
            else:
                a = nx.ones((ns,), type_as=np.array([1.0], dtype=np.float64)) / _budget
        else:
            a = nx.ones((sliceA.shape[0],))/sliceA.shape[0]
    else:
        a = nx.from_numpy(a_distribution)
        
    if b_distribution is None:
        if dummy_cell:
            if _has_dummy_tgt:
                b_vals = np.full(nt + 1, 1.0 / _budget, dtype=np.float64)
                b_vals[-1] = float(_w_dummy_tgt) / _budget
                b = nx.from_numpy(b_vals)
            else:
                b = nx.ones((nt,), type_as=np.array([1.0], dtype=np.float64)) / _budget
        else:
            b = nx.ones((sliceB.shape[0],))/sliceB.shape[0]
    else:
        b = nx.from_numpy(b_distribution)

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        a = a.cuda()
        b = b.cuda()
    
    if norm:
        D_A /= nx.min(D_A[D_A>0])
        D_B /= nx.min(D_B[D_B>0])

    # ── 6. Execute FGW Solver ──────────────────────────────────────────────
    if G_init is not None:
        if dummy_cell and (_has_dummy_src or _has_dummy_tgt):
            _gi = np.array(G_init, dtype=np.float64)
            _gi_aug = np.zeros((_ns_aug, _nt_aug), dtype=np.float64)
            _gi_aug[:ns, :nt] = _gi
            G_init = _gi_aug
        G_init = nx.from_numpy(G_init)
        if isinstance(nx,ot.backend.TorchBackend):
            G_init = G_init.double()
            if use_gpu: G_init = G_init.cuda()

    if dummy_cell:
        _ns_log, _nt_log = sliceA.shape[0], sliceB.shape[0]
        G = np.ones((_ns_log, _nt_log)) / (_ns_log * _nt_log)
    else:
        G = np.ones((a.shape[0], b.shape[0])) / (a.shape[0] * b.shape[0])

    if neighborhood_dissimilarity == 'jsd': initial_obj_neighbor = np.sum(js_dist_neighborhood*G)
    elif neighborhood_dissimilarity == 'msd': initial_obj_neighbor = np.sum(msd_neighborhood*G)
    elif neighborhood_dissimilarity == 'cosine': initial_obj_neighbor = np.sum(cosine_dist_neighborhood*G)

    initial_obj_gene = np.sum(cosine_dist_gene_expr*G)

    _fgw_extra = {'numItermaxEmd': 500_000} if dummy_cell else {}
    pi, logw = fused_gromov_wasserstein_incent(
        M1, M2, D_A, D_B, a, b, 
        G_init=G_init, loss_fun='square_loss', 
        alpha=alpha, gamma=gamma, log=True, 
        numItermax=numItermax, verbose=verbose, 
        use_gpu=use_gpu, **_fgw_extra
    )
    pi = nx.to_numpy(pi)

    # ── 7. Post-Processing & Returns ───────────────────────────────────────
    if dummy_cell and (_has_dummy_src or _has_dummy_tgt):
        pi_full = pi.copy()
        if _has_dummy_src and _has_dummy_tgt: pi = pi_full[:ns, :nt]
        elif _has_dummy_src: pi = pi_full[:ns, :]      
        elif _has_dummy_tgt: pi = pi_full[:, :nt]      

        pi_sum = pi.sum()
        if pi_sum > 0: pi = pi / pi_sum

    if neighborhood_dissimilarity == 'jsd':
        max_indices = np.argmax(pi, axis=1)
        jsd_error = np.zeros(max_indices.shape)
        for i in range(len(max_indices)):
            jsd_error[i] = pi[i][max_indices[i]] * js_dist_neighborhood[i][max_indices[i]]
        final_obj_neighbor = np.sum(jsd_error)
    elif neighborhood_dissimilarity == 'msd':
        final_obj_neighbor = np.sum(msd_neighborhood*pi)
    elif neighborhood_dissimilarity == 'cosine':
        max_indices = np.argmax(pi, axis=1)
        cos_error = np.zeros(max_indices.shape)
        for i in range(len(max_indices)):
            cos_error[i] = pi[i][max_indices[i]] * cosine_dist_neighborhood[i][max_indices[i]]
        final_obj_neighbor = np.sum(cos_error)

    final_obj_gene = np.sum(cosine_dist_gene_expr * pi)

    logFile.write(f"Runtime: {str(time.time() - start_time)} seconds\n")
    logFile.close()

    if isinstance(backend,ot.backend.TorchBackend) and use_gpu:
        torch.cuda.empty_cache()

    if return_obj:
        return pi, initial_obj_neighbor, initial_obj_gene, final_obj_neighbor, final_obj_gene
    
    return pi

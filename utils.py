import enum
import numpy as np
from pandas.core.algorithms import isin
import scanpy as sc
import os.path as osp
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

def get_mask(X):
    assert np.sum(X < 0) == 0
    X = np.array(X)
    mask = np.zeros(shape=(X.shape[0], X.shape[1]))
    mask[X == 0] = 0
    mask[X != 0] = 1
    return mask

def normalize(adata, library_size=True, normalize_input=True, log_input=True):

    if library_size or normalize_input or log_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if library_size:
        sc.pp.normalize_per_cell(adata)
        adata.obs['library_size'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['library_size'] = 1.0

    if log_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def get_zero_mask(X, n_neighbors=20, radius=1.0, large_memory=True, large_memory_batch=1024, probability=False, low_expression_threshold=0.2, low_expression_percentage=0.8):
    """
    Obtain the zero mask for a given matrix so that some zeros will remain unchanged. Those zero items will be specified a value one in the final mask matrix for being selected just like nonzero items.
    """
    assert np.sum(X < 0) == 0
    _origin_non_zero_mask = get_mask(X) # cell x gene
    _mask = np.copy(_origin_non_zero_mask)
    low_exp_count_thresholds = []
    for g in range(X.shape[1]):
        one_gene_expression = X[:, g]
        non_zero_idx = np.nonzero(one_gene_expression)[0]
        low_exp_count_threshold = np.sort(one_gene_expression[non_zero_idx])[np.floor(low_expression_threshold * len(non_zero_idx)).astype(np.int32)]
        low_exp_count_thresholds.append(low_exp_count_threshold)
    low_exp_count_thresholds = np.array(low_exp_count_thresholds).reshape(1, -1)
    if large_memory:
        for b in tqdm(range(0, X.shape[0], large_memory_batch)):
            _non_zero_mask = np.expand_dims(_origin_non_zero_mask[b:np.minimum(b + large_memory_batch, X.shape[0])], 1) # batch x 1 x gene
            _non_zero_mask = np.repeat(_non_zero_mask, X.shape[0], 1) # batch x cell x gene
            for i in tqdm(range(len(_non_zero_mask))):
                _non_full_X = np.multiply(_non_zero_mask[i], X) # same cell x gene; get gene expression for all cells with respect to the non-zero part of the current cell
                _neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, n_jobs=1).fit(_non_full_X)
                _indices = np.squeeze(_neigh.kneighbors(_non_full_X[[b + i]], return_distance=False), 0) # 1 x n_neighbors -> n_neighbors
                if probability is False:
                    _mask[b + i, np.nonzero(np.mean(X[_indices] <= low_exp_count_thresholds, 0) >= low_expression_percentage)[0]] = 1
                else:
                    _mask[b + i, :] = np.maximum(_mask[b + i, :], np.minimum(np.mean(X[_indices] <= low_exp_count_thresholds, 0), 1)) # adaptive
    else: # legacy
        for i in tqdm(range(len(_origin_non_zero_mask))):
            _non_zero_mask = _origin_non_zero_mask[[i]].copy() # 1 x gene
            _non_zero_mask = np.repeat(_non_zero_mask, _origin_non_zero_mask.shape[0], 0) # same cell x gene
            _non_full_X = np.multiply(_non_zero_mask, X) # same cell x gene; get gene expression for all cells with respect to the non-zero part of the current cell
            _neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, n_jobs=1).fit(_non_full_X)
            _indices = np.squeeze(_neigh.kneighbors(_non_full_X[[i]], return_distance=False), 0) # 1 x n_neighbors -> n_neighbors
            if probability is False:
                _mask[i, np.nonzero(np.sum(_origin_non_zero_mask[_indices], 0))] = 1
            else:
                _mask[i, :] = np.maximum(_mask[i, :], np.mean(_origin_non_zero_mask[_indices], 0))
    return _mask

def get_top_mask(X, top=10):
    assert np.sum(X < 0) == 0
    X = np.array(X)
    mask = np.zeros(shape=(X.shape[0], X.shape[1]))
    mask[X == 0] = 0
    sort_idx = np.argsort(X, axis=1)[:, :top]
    mask[np.arange(len(mask)).reshape(-1, 1), sort_idx] = 1
    # assert np.sum(mask) == X.shape[0] * top
    return mask

def get_local_zero_mask(X, n_neighbors=20, radius=1.0, n_negative_neighbors=True):
    assert np.sum(X < 0) == 0
    _origin_non_zero_mask = get_top_mask(X) # cell x gene
    _mask = np.copy(_origin_non_zero_mask)
    if n_negative_neighbors is not None:
        neg_idx = []
    for i in tqdm(range(len(_origin_non_zero_mask))):
        _non_zero_mask = _origin_non_zero_mask[[i]].copy() # 1 x gene
        _non_zero_mask = np.repeat(_non_zero_mask, _origin_non_zero_mask.shape[0], 0) # same cell x gene
        _non_full_X = np.multiply(_non_zero_mask, X) # same cell x gene
        _neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, n_jobs=1).fit(_non_full_X)
        if n_negative_neighbors is not None:
            pair_dist = pairwise_distances(_non_full_X[[i]], _non_full_X).squeeze().argsort()[::-1]
            neg_idx.append(pair_dist)
        _indices = np.squeeze(_neigh.kneighbors(_non_full_X[[i]], return_distance=False), 0) # 1 x n_neighbors -> n_neighbors
        # assert _indices.shape == (n_neighbors, )
        # assert np.sum(_origin_non_zero_mask[_indices], 0).shape == (_origin_non_zero_mask.shape[1], )
        _mask[i, np.nonzero(np.sum(_origin_non_zero_mask[_indices], 0))] = 1
    neg_idx = np.array(neg_idx)
    if n_negative_neighbors is not None:
        return _mask, neg_idx
    else:
        return _mask

def get_rand_mask_for_val(X, rng, dropout=0.5, nonzero_mask=None):
    if nonzero_mask is None:
        nonzero_mask = get_mask(X)
    dropout = float(dropout)
    mask = np.ones(shape=(X.shape[0], X.shape[1]))
    assert np.sum(X < 0) == 0 # the condition of the first judgement
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if nonzero_mask[i, j] == 0: 
                mask[i, j] = 0
            elif rng.random() < dropout and np.sum(np.multiply(nonzero_mask[:, j], mask[:, j])) > nonzero_mask[i, j] and np.sum(np.multiply(nonzero_mask[i, :], mask[i, :])) > nonzero_mask[i, j]:
                # ensure no gene becomes all zero across all cells and no cell becomes all zero across all genes
                mask[i, j] = 0
                if np.var(np.multiply(X[:, j], mask[:, j])) == 0:
                    mask[i, j] = 1
    return mask

def get_rand_mask(X, rng, dropout=0.5):
    dropout = float(dropout)
    mask = np.ones(shape=(X.shape[0], X.shape[1]))
    assert np.sum(X < 0) == 0 # the condition of the first judgement
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] == 0: 
                mask[i, j] = 0
            elif rng.random() < dropout and np.sum(np.multiply(X[:, j], mask[:, j])) > X[i, j] and np.sum(np.multiply(X[i, :], mask[i, :])) > X[i, j]:
                # ensure no gene becomes all zero across all cells and no cell becomes all zero across all genes
                mask[i, j] = 0
                if np.var(np.multiply(X[:, j], mask[:, j])) == 0:
                    mask[i, j] = 1
    return mask

def get_rand_mask_efficient(X, rng, dropout=0.5, select_zero=False):
    assert np.sum(X < 0) == 0 # the condition of the first judgement
    dropout = float(dropout)
    while True:
        mask = np.ones(shape=(X.shape[0], X.shape[1]))
        random_mat = rng.random(size=X.shape)
        if not select_zero:
            mask[np.nonzero(X == 0)] = 0
        mask[np.nonzero(random_mat < dropout)] = 0
        multi = np.multiply(X, mask)
        if (np.sum(multi, 0) > 0).all() and (np.sum(multi, 1) > 0).all():
            break
    return mask

def get_split_percent(split_pct, X_adata):
    return int(float(split_pct) * len(X_adata))

def get_bench_mask(observe_mask, tmask):
    if observe_mask is not None and tmask is not None:
        return tmask - observe_mask
    else:
        return None

def load_output_data(data_dir, dataset_dir, output_dir, method_name, dropout, seed, format='df'):
    if format == 'df':
        imX_df = pd.read_csv(osp.join(data_dir, dataset_dir, output_dir, method_name + ".drop{}.seed{}.name.csv.gz".format(dropout, seed)), index_col=0, header=0)
    elif format == 'adata':
        imX_df = sc.read_csv(osp.join(data_dir, dataset_dir, output_dir, method_name + ".drop{}.seed{}.name.csv.gz".format(dropout, seed)), first_column_names=True)
    imX_df = check_and_correct_R_name(imX_df)
    return imX_df

def load_input_data(data_dir, dataset_dir, highly_genes, target_format, load_origin=True):
    if load_origin:
        if target_format == "log1p":
            X_adata_origin_full = sc.read_csv(osp.join(data_dir, dataset_dir, f"{highly_genes}.X.name.csv.gz"), first_column_names=True)
        elif target_format == "count":
            X_adata_origin_full = sc.read_csv(osp.join(data_dir, dataset_dir, f"{highly_genes}.X.count.name.csv.gz"), first_column_names=True)
            return X_adata_origin_full
        else:
            raise

def check_and_correct_R_name(df):
    if isinstance(df, pd.DataFrame):
        if isinstance(df.index[0], str):
            df_idx_0 = set([s[0] for s in df.index])
            if len(df_idx_0 - set("X")) == 0:
                df.index = [s[1:] for s in df.index]
        if isinstance(df.columns[0], str):
            df_columns_0 = set([s[0] for s in df.columns])
            if len(df_columns_0 - set("X")) == 0:
                df.columns = [s[1:] for s in df.columns]
    else:
        if isinstance(df.obs.index[0], str):
            df_idx_0 = set([s[0] for s in df.obs.index])
            if len(df_idx_0 - set("X")) == 0:
                df.obs.index = [s[1:] for s in df.obs.index]
        if isinstance(df.var.index[0], str):
            df_columns_0 = set([s[0] for s in df.var.index])
            if len(df_columns_0 - set("X")) == 0:
                df.var.index = [s[1:] for s in df.var.index]
    return df

def correct_name(name, corrected_name="IGSimpute"):
    if name == "recover.our":
        return corrected_name
    else:
        return name

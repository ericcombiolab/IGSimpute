from numpy.random import default_rng
import pandas as pd
from os import path as osp
import scanpy as sc
import os

def load_preprocessed(X_count, X_unscaled, cell_names, gene_names, type_name_df=None, z_normalize=True):
    adata_cnt = sc.AnnData(X_count)
    adata = sc.AnnData(X_unscaled)
    adata.obs['cell_name'] = cell_names
    adata_cnt.obs['cell_name'] = cell_names
    if type_name_df is not None:
        adata.uns["num2type"] = type_name_df["num2type"]
        adata_cnt.uns["num2type"] = type_name_df["num2type"]

    adata.var['gene_name'] = gene_names
    adata_cnt.var['gene_name'] = gene_names
    adata_unscaled = adata.copy()
    if z_normalize:
        sc.pp.scale(adata)
    adata = adata.copy()
    return adata, adata_unscaled, adata_cnt

def preprocess(X, cell_names, gene_names, quality_control=True, normalize_input=True, log1p_input=True, highly_genes=None, z_normalize=True, dataset_dir=".", rng=default_rng(), generate_files=True, chunk_size=256):
    if generate_files:
        idx = rng.choice(X.shape[0], X.shape[0], replace=False)
        adata = sc.AnnData(X[idx])
        adata.obs['cell_name'] = cell_names[idx]
    adata.var['gene_name'] = gene_names

    if quality_control is True:
        sc.pp.filter_genes(adata, min_counts=10)
    elif quality_control == "minimum":
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.filter_genes(adata, min_counts=1)
    elif quality_control == "tm_droplet":
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.filter_cells(adata, min_genes=500)
        sc.pp.filter_cells(adata, min_counts=1000)
    adata_cnt = adata.copy()
    if normalize_input:
        sc.pp.normalize_per_cell(adata)
    if log1p_input:
        sc.pp.log1p(adata)
    if highly_genes > 0. and highly_genes < 1.:
        sc.pp.highly_variable_genes(adata, n_top_genes=int(highly_genes * adata.X.shape[1]), flavor="seurat", subset=True)
    elif int(highly_genes) > 1 and int(highly_genes) <= adata.X.shape[1]:
        sc.pp.highly_variable_genes(adata, n_top_genes=int(highly_genes), flavor="seurat", subset=True)
    if generate_files:
        if 'highly_variable' in adata.var:
            gene_names = [adata.var['gene_name'][i] for i in adata.var['highly_variable'].index]
            adata_cnt = adata_cnt[:, adata.var['highly_variable'].index].copy()
        else:
            gene_names = adata.var['gene_name']
    adata_unscaled = adata.copy()
    
    if z_normalize:
        sc.pp.scale(adata)
    adata = adata.copy()
    cell_names = adata.obs['cell_name']
    
    if generate_files:
        # format: content.form
        # e.g. (X.count.T).(name.space.gz)
        tmp_out = adata_unscaled.to_df()
        tmp_out.to_csv(osp.join(dataset_dir, "{}.X.csv.gz".format(highly_genes)), sep=',', header=False, index=False, chunksize=chunk_size)
        tmp_out.index = cell_names
        tmp_out.columns = gene_names
        tmp_out.to_csv(osp.join(dataset_dir, "{}.X.name.csv.gz".format(highly_genes)), chunksize=chunk_size)
        tmp_out = adata_unscaled.to_df().T
        tmp_out.to_csv(osp.join(dataset_dir, "{}.X.T.csv.gz".format(highly_genes)), sep=',', header=False, index=False, chunksize=chunk_size)
        tmp_out.index = gene_names
        tmp_out.columns = cell_names
        tmp_out.to_csv(osp.join(dataset_dir, "{}.X.T.name.csv.gz".format(highly_genes)), chunksize=chunk_size)
        tmp_out = adata_cnt.to_df()
        tmp_out.to_csv(osp.join(dataset_dir, "{}.X.count.csv.gz".format(highly_genes)), sep=',', header=False, index=False, chunksize=chunk_size)
        tmp_out.index = cell_names
        tmp_out.columns = gene_names
        tmp_out.to_csv(osp.join(dataset_dir, "{}.X.count.name.csv.gz".format(highly_genes)), chunksize=chunk_size)
        tmp_out = adata_cnt.to_df().T
        tmp_out.to_csv(osp.join(dataset_dir, "{}.X.count.T.csv.gz".format(highly_genes)), sep=',', header=False, index=False, chunksize=chunk_size)
        tmp_out.index = gene_names
        tmp_out.columns = cell_names
        tmp_out.to_csv(osp.join(dataset_dir, "{}.X.count.T.name.csv.gz".format(highly_genes)), chunksize=chunk_size)

    return adata, adata_unscaled, adata_cnt

def load_h5ad(data_dir, dataset_dir, exp_file_name, highly_genes, rng=default_rng(), generate_files=False, seed=0):
    if generate_files:
        sc_input_adata = sc.read_h5ad(osp.join(data_dir, dataset_dir, exp_file_name))
        full_gene_list = sc_input_adata.var.index.tolist()
        sc_gene_names = full_gene_list
        gene_name_idx_dict = dict(zip(sc_input_adata.var.index.values, range(len(sc_input_adata.var.index.values))))
        sc_gene_idxs = [gene_name_idx_dict[k] for k in sc_gene_names]
        sc_expression = sc_input_adata.X.toarray()[:, sc_gene_idxs]
        if not osp.isdir(osp.join(data_dir, dataset_dir)):
            os.mkdir(osp.join(data_dir, dataset_dir))
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_input_adata.obs_names, sc_gene_names, quality_control=True, normalize_input=False,
                                                    highly_genes=highly_genes, dataset_dir=osp.join(data_dir, dataset_dir), rng=rng, generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(osp.join(data_dir, dataset_dir, '{}.X.count.name.csv.gz'.format(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(osp.join(data_dir, dataset_dir, '{}.X.name.csv.gz'.format(highly_genes)), index_col=0).values
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_cnt_df.index, sc_cnt_df.columns)
    return adata, adata_unscaled, adata_cnt

def load_csv(data_dir, dataset_dir, exp_file_name, highly_genes, rng=default_rng(), generate_files=False, seed=0):
    if generate_files:
        sc_input_adata = sc.read_csv(osp.join(data_dir, dataset_dir, exp_file_name), first_column_names=True)
        full_gene_list = sc_input_adata.var.index.tolist()
        sc_gene_names = full_gene_list
        gene_name_idx_dict = dict(zip(sc_input_adata.var.index.values, range(len(sc_input_adata.var.index.values))))
        sc_gene_idxs = [gene_name_idx_dict[k] for k in sc_gene_names]
        sc_expression = sc_input_adata.X.toarray()[:, sc_gene_idxs]
        if not osp.isdir(osp.join(data_dir, dataset_dir)):
            os.mkdir(osp.join(data_dir, dataset_dir))
        adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_input_adata.obs_names, sc_gene_names, quality_control=True, normalize_input=False,
                                                    highly_genes=highly_genes, dataset_dir=osp.join(data_dir, dataset_dir), rng=rng, generate_files=generate_files)
    else:
        sc_cnt_df = pd.read_csv(osp.join(data_dir, dataset_dir, '{}.X.count.name.csv.gz'.format(highly_genes)), index_col=0)
        sc_unscaled = pd.read_csv(osp.join(data_dir, dataset_dir, '{}.X.name.csv.gz'.format(highly_genes)), index_col=0).values
        adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_cnt_df.index, sc_cnt_df.columns)
    return adata, adata_unscaled, adata_cnt

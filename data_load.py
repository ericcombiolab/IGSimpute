import os

import numpy as np
from numpy.random import default_rng
import pandas as pd
import scanpy as sc
from utils import get_mask
from os import path as osp
from dataset_load import preprocess, load_preprocessed, load_h5ad, load_csv

def load_data(data_dir, dataset_dir, exp_file_name, highly_genes, n_neighbors=10, rng=default_rng(), generate_files=False, seed=0, ref_dropout=0.2, low_expression_threshold=0.20, low_expression_percentage=0.80):
    if dataset_dir.startswith("tm_droplet_"):
        if generate_files:
            sc_input_adata = sc.read_h5ad(osp.join(data_dir, 'czbiohub-tabula-muris', 'TM_droplet_mat.h5ad'))
            sc_anno_df = pd.read_csv(osp.join(data_dir, 'czbiohub-tabula-muris', 'TM_droplet_metadata.csv'), low_memory=False)
            if not dataset_dir.endswith("all"):
                sc_anno_df = sc_anno_df.loc[sc_anno_df['tissue'] == '_'.join(dataset_dir.split('_')[2:]), :]
            sc_expression = sc_input_adata[sc_anno_df.index, :].X.toarray()
            sc_gene_names = sc_input_adata.var.index
            if not osp.isdir(osp.join(data_dir, dataset_dir)):
                os.mkdir(osp.join(data_dir, dataset_dir))
            adata, adata_unscaled, adata_cnt = preprocess(sc_expression, sc_anno_df.index, sc_gene_names, quality_control='tm_droplet',
                                                        highly_genes=highly_genes, dataset_dir=osp.join(data_dir, dataset_dir), rng=rng, generate_files=generate_files)

        else:
            sc_cnt_df = pd.read_csv(osp.join(data_dir, dataset_dir, '{}.X.count.name.csv.gz'.format(highly_genes)), index_col=0)
            sc_unscaled = pd.read_csv(osp.join(data_dir, dataset_dir, '{}.X.name.csv.gz'.format(highly_genes)), index_col=0).values
            adata, adata_unscaled, adata_cnt = load_preprocessed(sc_cnt_df.values, sc_unscaled, sc_cnt_df.index, sc_cnt_df.columns)

    else:
        # adata, adata_unscaled, adata_cnt = getattr(dataset_load, "load_{}".format(dataset_dir))(data_dir, highly_genes, rng, generate_files, seed)
        if exp_file_name.endswith("h5ad"):
            adata, adata_unscaled, adata_cnt = load_h5ad(data_dir, highly_genes, exp_file_name, rng, generate_files, seed)
        else:
            adata, adata_unscaled, adata_cnt = load_csv(data_dir, highly_genes, exp_file_name, rng, generate_files, seed)
    try:
        try:
            zero_mask_df = pd.read_csv(os.path.join(data_dir, dataset_dir, "{}.zeromask{}.{:.2f}threshold.{}masked{}.name.csv.gz".format(highly_genes, n_neighbors, low_expression_threshold, seed)), index_col=0)
            zero_mask = zero_mask_df.values
            zero_mask[zero_mask >= np.round(low_expression_percentage, 2)] = 1
            zero_mask[zero_mask < np.round(low_expression_percentage, 2)] = 0
        except FileNotFoundError:
            print('no zeromask file')
            raise
        # except FileNotFoundError:
        #     try:
        #         zero_mask_df = pd.read_csv(os.path.join(data_dir, dataset_dir, "{}.zeromask{}.{:.2f}threshold.name.csv.gz".format(highly_genes, n_neighbors, low_expression_threshold)), index_col=0)
        #         zero_mask = zero_mask_df.values
        #     except FileNotFoundError:
        #         print('no zeromask file')
        #         raise
    except:
        # zero_mask = get_mask(sc_cnt_df.values)
        zero_mask = get_mask(adata_cnt.X)
    return adata, adata_unscaled, adata_cnt, zero_mask


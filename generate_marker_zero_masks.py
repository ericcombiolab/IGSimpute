import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import os.path as osp
from sklearn.neighbors import NearestNeighbors
from utils import load_input_data, get_zero_mask
from data_load import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench-dataset', type=str, default="human_pancreas", dest='dataset_dir')
    parser.add_argument('--data-dir', type=str, default="../cgi_datasets", dest='data_dir')
    parser.add_argument('--highly-variable-genes', type=float, default=0.1, dest='highly_genes_num')
    parser.add_argument('--low-expression-threshold', type=float, default=0.20, dest='low_expression_threshold')
    parser.add_argument('--low-expression-percentage', type=float, default=0.80, dest='low_expression_percentage')
    parser.add_argument('--target-format', type=str, default="count", dest='target_format')

    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    data_dir = args.data_dir
    highly_genes_num = args.highly_genes_num
    low_expression_threshold = args.low_expression_threshold
    low_expression_percentage = args.low_expression_percentage
    target_format = args.target_format
    n_neighbors = 10

    X_adata = load_input_data(data_dir, dataset_dir, highly_genes_num, target_format)
    zero_mask= get_zero_mask(X_adata.X, n_neighbors=n_neighbors, low_expression_threshold=np.round(low_expression_threshold, 2), probability=True, large_memory_batch=128)
    zero_mask_df = pd.DataFrame(zero_mask, index=X_adata.obs.index, columns=X_adata.var.index)
    zero_mask_df.to_csv(osp.join(data_dir, dataset_dir, "{}.zeromask{}.{:.2f}threshold.name.csv.gz".format(highly_genes_num, n_neighbors, low_expression_threshold)), chunksize=256)

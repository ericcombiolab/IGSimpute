#!/usr/bin/env python

import argparse
import os
from numpy.random import default_rng

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from att_network import *
from data_load import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256, dest='batch_size')
    parser.add_argument('--bench-dataset', type=str, default="tm_droplet_Heart_and_Aorta", dest='dataset_dir')
    parser.add_argument('--data-dir', type=str, default="../cgi_datasets", dest='data_dir')
    parser.add_argument('--rec-loss-weight', type=str, default=None, dest='lambda_c')
    parser.add_argument('--dim', type=float, default=400, dest='dim')
    parser.add_argument('--encoder-dropout-rate', type=str,default=None, dest='lambda_d')
    parser.add_argument('--epochs', type=int, default=100, dest='epochs')
    parser.add_argument('--exp-file-name', type=str, default="X.csv", dest='exp_file_name')
    parser.add_argument('--generate-files', action='store_true', dest='generate_files')
    parser.add_argument('--gpu_node', type=str, default="0", dest='gpu_option')
    parser.add_argument('--ggl-loss-weight', type=str, default=None, dest='lambda_a')
    parser.add_argument('--gsl-L1-weight', type=str, default=None, dest='lambda_b')
    parser.add_argument('--highly-variable-genes', type=float, default=0.1, dest='highly_genes_num')
    parser.add_argument('--low-expression-percentage', type=float, default=0.80, dest='low_expression_percentage')
    parser.add_argument('--low-expression-threshold', type=float, default=0.20, dest='low_expression_threshold')
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate')
    parser.add_argument('--output-dir', type=str, default="output", dest='output_dir')
    parser.add_argument('--output-file-prefix', type=str, default="our", dest='IGSimpute')
    parser.add_argument('--seed', type=int, default=0, dest="seed")
    parser.add_argument('--split-percentage', type=str, default="0.8", dest='split_pct')
    parser.add_argument('--sub-sampling-num', type=str, default=None, dest='sub_sampling_num')
    parser.add_argument('--valid-dropout', type=float, default=0.2, dest='valid_dropout')

    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    data_dir = args.data_dir
    highly_genes_num = args.highly_genes_num
    output_file_prefix = args.output_file_prefix
    output_dir = args.output_dir
    seed = args.seed
    batch_size = args.batch_size
    dim = args.dim
    epochs = args.epochs
    exp_file_name = args.exp_file_name
    generate_files = args.generate_files
    gpu_option = args.gpu_option
    L2 = args.L2
    lambda_a = 0 if args.lambda_a is None or args.lambda_a == "None" else float(args.lambda_a)
    lambda_b = 0 if args.lambda_b is None or args.lambda_b == "None" else float(args.lambda_b)
    lambda_c = 0 if args.lambda_c is None or args.lambda_c == "None" else float(args.lambda_c)
    lambda_d = 0 if args.lambda_d is None or args.lambda_d == "None" else float(args.lambda_d)
    learning_rate = args.learning_rate
    low_expression_threshold = args.low_expression_threshold
    low_expression_percentage = args.low_expression_percentage
    split_pct = args.split_pct
    sub_sampling_num = 0 if args.sub_sampling_num is None or args.sub_sampling_num == "None" else int(args.sub_sampling_num)
    valid_dropout = args.valid_dropout
    compression = '.gz'
    n_neighbors = 10
    chunk_size = 256
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    rng = default_rng(seed)
    seed = args.seed
    adata, adata_unscaled, adata_cnt, post_zero_mask = load_data(data_dir, dataset_dir, exp_file_name, highly_genes_num, n_neighbors, rng, generate_files, seed, low_expression_threshold=low_expression_threshold, low_expression_percentage=low_expression_percentage)
    if epochs > 0:
        if sub_sampling_num != 0:
            subsample_idx = rng.choice(adata.shape[0], sub_sampling_num, replace=False)
            adata = adata[subsample_idx, :]
            adata_unscaled = adata_unscaled[subsample_idx, :]
            adata_cnt = adata_cnt[subsample_idx, :]
            post_zero_mask = post_zero_mask[subsample_idx, :]
        else:
            pass
        valid_split = int(float(split_pct) * len(adata.X))
        if dim <= 1 and dim > 0:
            dim = int(adata.X.shape[1] * dim)
        elif dim > 1:
            dim = int(dim)
        else:
            raise
        dims = [adata.X.shape[1], dim]
        model = Model(data_dir, dataset_dir, output_dir, dims, learning_rate, batch_size, lambda_a, lambda_b, lambda_c, lambda_d, epochs, seed)
        model.train(adata, adata_unscaled, adata_cnt, post_zero_mask, valid_split, valid_dropout, rng, gpu_option)

        # # for imputation
        model.recover_imX_df.to_csv(os.path.join(data_dir, dataset_dir, output_dir, output_file_prefix + ".name.csv") + compression, chunksize=chunk_size)
        model.imX_df.to_csv(os.path.join(data_dir, dataset_dir, output_dir, output_file_prefix + ".KNN.name.csv") + compression, chunksize=chunk_size)

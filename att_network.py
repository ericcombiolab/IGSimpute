import math
import os
from os import path as osp
import random
from xmlrpc.client import boolean
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.compat.v1.train import Saver, latest_checkpoint, cosine_decay, cosine_decay_restarts, MomentumOptimizer
from tensorflow.contrib.memory_stats import BytesLimit, MaxBytesInUse, BytesInUse
from tensorflow.keras.layers import Dense, GaussianNoise, Dropout, BatchNormalization
from tensorflow.keras.initializers import glorot_uniform, Ones, he_uniform, RandomUniform
from tensorflow.contrib.opt import AdamWOptimizer, MomentumWOptimizer, extend_with_decoupled_weight_decay
from utils import get_mask, get_rand_mask, get_rand_mask_for_val, get_zero_mask, normalize
from tqdm import tqdm, trange

class Model(object):
    def __init__(self, data_dir, dataset_dir, output_dir, dims, learning_rate, batch_size, lambda_a, lambda_b, lambda_c, lambda_d, epochs=2000, seed=0, n_cores=-1, noise_sd = 1.5):
        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.dims = dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.pretrain_epochs = epochs
        self.n_cores = n_cores
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.lambda_c = lambda_c
        self.lambda_d = lambda_d
        self.seed = seed

        self.optimizer = AdamWOptimizer(weight_decay=0.0001, learning_rate=self.learning_rate)
        self.training_flag = tf.compat.v1.placeholder(dtype=boolean, shape=())
        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.non_zero_mask = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.unscale_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.x_count = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.target = self.x_count

        with tf.compat.v1.variable_scope("sc"):
            self.gene_b = tf.nn.relu(tf.compat.v1.get_variable(name="gb", shape=[self.dims[0]], dtype=tf.float32, initializer=glorot_uniform(seed=self.seed)))
            self.GRN = tf.compat.v1.get_variable(name='grn', shape=[self.dims[0], self.dims[0]], dtype=tf.float32, initializer=glorot_uniform(seed=self.seed))
        
            self.noise = GaussianNoise(self.noise_sd, name='noise')
            self.noised_x = self.noise(self.x)
            self.select_enc_dense1 = Dense(units=self.dims[-1], kernel_initializer=glorot_uniform(seed=self.seed), name='s_enc1')
            self.select_enc1 = tf.nn.relu(self.select_enc_dense1(self.noised_x))
            self.select_h_dense = Dense(units=self.dims[-1] / 2, kernel_initializer=glorot_uniform(seed=self.seed), name='s_h')
            self.select_h = tf.nn.relu(self.select_h_dense(self.select_enc1))
            self.select_dec_dense1 = Dense(units=self.dims[-1], kernel_initializer=glorot_uniform(seed=self.seed), name='s_dec1')
            self.select_dec1 = tf.nn.relu(self.select_dec_dense1(self.select_h))
            self.select_m_dense = Dense(units=self.dims[0], kernel_initializer=glorot_uniform(seed=self.seed), name='s_m')
            self.select_m = tf.nn.sigmoid(self.select_m_dense(self.select_dec1))
            self.dropped_select_m_drop = Dropout(
                rate=self.lambda_d, noise_shape=None, seed=self.seed, name="dropout_select_m"
            )
            self.dropped_select_m = self.dropped_select_m_drop(self.select_m, training=self.training_flag)
            self.selected_h = tf.multiply(self.noised_x, self.dropped_select_m)
            self.h_dense = Dense(units=self.dims[-1], kernel_initializer=glorot_uniform(seed=self.seed), name='encoder_hidden')
            self.h = self.h_dense(self.selected_h)
            self.auto_decode_X_dense = Dense(units=self.dims[0], activation='softplus', kernel_initializer=glorot_uniform(seed=self.seed), name='auto_decode_X')
            self.auto_decode_X = self.auto_decode_X_dense(self.h)
            self.alp = tf.math.sigmoid(tf.compat.v1.get_variable(name="alp", shape=[self.dims[0]], dtype=tf.float32, initializer=glorot_uniform(seed=self.seed)))
        self.imX = self.alp * tf.matmul(tf.multiply(self.target, self.non_zero_mask) + tf.multiply(self.auto_decode_X, 1 - self.non_zero_mask), tf.multiply(self.GRN, 1 - tf.constant(np.eye(self.dims[0], dtype=np.float32)))) + (1 - self.alp) * self.gene_b
        self.ae_loss = tf.reduce_sum(input_tensor=tf.multiply(tf.divide(tf.square(self.auto_decode_X-self.target), 
                                                                        tf.reshape(tf.reduce_sum(input_tensor=self.non_zero_mask, axis=1), (-1,1))),
                                                            self.non_zero_mask))
        self.grn_loss = tf.reduce_sum(input_tensor=tf.multiply(tf.divide(tf.square(self.imX-self.target),
                                                                         tf.reshape(tf.reduce_sum(input_tensor=self.non_zero_mask, axis=1), (-1,1))),
                                                               self.non_zero_mask))
        self.pretrain_ae_loss = self.lambda_c * self.ae_loss
        self.train_ae_loss = self.lambda_c * self.ae_loss
        self.normalized_grn_loss = self.lambda_a * 1 * self.grn_loss
        self.imp_loss = self.pretrain_ae_loss + self.normalized_grn_loss
        self.imp2_loss = self.train_ae_loss + self.normalized_grn_loss
        self.mask_L1_loss = self.lambda_b * tf.reduce_sum(self.select_m)
        self.mask_loss = self.mask_L1_loss
        self.reconstruction_loss = self.imp_loss + self.mask_loss
        self.close_total_loss = self.imp2_loss + self.mask_loss
        self.init_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.pretrain_op = self.optimizer.minimize(self.reconstruction_loss)
        self.close_op = self.optimizer.minimize(self.close_total_loss)
        self.is_current_training = True

    def set_training(self, is_training=True):
        if self.is_current_training == is_training:
            pass
        else:
            self.is_current_training = is_training
            self.dropped_select_m.training = is_training
            self.noised_x.training = is_training

    def train(self, adata, adata_unscaled, adata_cnt, post_zero_mask, valid_split, valid_dropout, rng, gpu_option):

        X = adata.X[:valid_split].astype(np.float32)
        unscale_X = adata_unscaled[:valid_split].X.astype(np.float32)
        count_X = adata_cnt.X[:valid_split].astype(np.float32)
        Y = adata.obs["cell_groups"][:valid_split]
        # do not consider biological zeros
        # !!!!!
        nonzero_mask = get_mask(unscale_X)
        # nonzero_mask = post_zero_mask[:valid_split]
        # !!!!!
        assert (nonzero_mask.sum(1) == 0).sum() == 0
        # # consider biological zeros
        # nonzero_mask = post_zero_mask[:valid_split]

        if valid_split != len(adata.X):
            valid_X = adata.X[valid_split:].astype(np.float32)
            valid_unscale_X = adata_unscaled[valid_split:].X.astype(np.float32)
            valid_count_X = adata_cnt.X[valid_split:].astype(np.float32)
            # do not consider biological zeros
            # !!!!!
            valid_nonzero_mask = get_mask(valid_unscale_X)
            # valid_nonzero_mask = post_zero_mask[valid_split:]
            # !!!!!
            valid_observe_mask = get_rand_mask(valid_unscale_X, rng, valid_dropout)

            assert (valid_nonzero_mask.sum(1) == 0).sum() == 0
            assert (valid_observe_mask.sum(1) == 0).sum() == 0

            # valid_nonzero_mask = post_zero_mask[valid_split:]
            # valid_observe_mask = get_rand_mask_for_val(valid_unscale_X, rng, valid_dropout, valid_nonzero_mask)
            
            valid_bench_mask = valid_nonzero_mask - valid_observe_mask        

            valid_input_X = np.multiply(valid_observe_mask, valid_X) + np.multiply(valid_bench_mask, np.min(valid_X, 0, keepdims=True))
            valid_input_unscale_X = np.multiply(valid_observe_mask, valid_unscale_X)
            valid_input_count_X = np.multiply(valid_observe_mask, valid_count_X)
            valid_Y = adata.obs["cell_groups"][valid_split:]

        cells_name = adata.obs['cell_name']
        genes_name = adata.var['gene_name']
        batch_size = self.batch_size


        if X.shape[0] < batch_size:
            batch_size = X.shape[0]


        # print("end the data proprocess")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option

        config_ = tf.compat.v1.ConfigProto()
        # config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        config_.intra_op_parallelism_threads = 0
        config_.inter_op_parallelism_threads = 0
        self.sess = tf.compat.v1.Session(config=config_)
        init = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())

        self.sess.run(init)
        # self.iteration_per_epoch = math.ceil(float(len(X)) / float(batch_size))
        self.iteration_per_epoch = math.ceil(len(X) / batch_size)

        self.stored_train_ae_loss = []
        self.stored_normalized_grn_loss = []
        self.stored_mask_loss = []
        self.stored_imputation_mse = []
        self.stored_valid_mse = []
        # Stage: Imputation
        index_for_sampling = range(X.shape[0])
        min_valid_error = np.inf
        min_valid_error_epoch = 0
        early_stop_cnt = 0
        # print("begin model pretrain(Imputation)")
        # tic = time.perf_counter()
        for i in range(2):
            for j in range(self.iteration_per_epoch):
                # if i == 1 and j == 1:
                #     print(self.sess.run(BytesLimit())/1024/1024/1024)
                #     print(self.sess.run(MaxBytesInUse())/1024/1024/1024)
                #     print(self.sess.run(BytesInUse())/1024/1024/1024)
                batch_idx = rng.choice(index_for_sampling, batch_size)

                self.set_training()
                _, pretrain_ae_loss, normalized_grn_loss, mask_loss = self.sess.run(
                            [self.pretrain_op, self.pretrain_ae_loss, self.normalized_grn_loss, self.mask_loss],
                            feed_dict={
                                self.training_flag: True,
                                self.x: X[batch_idx],
                                self.unscale_x: unscale_X[batch_idx],
                                self.non_zero_mask: nonzero_mask[batch_idx],
                                self.x_count: count_X[batch_idx],
                                })

        saver = Saver()
        for i in trange(self.pretrain_epochs):
            for j in range(self.iteration_per_epoch):
                # if i == 5 and j == 1:
                #     print(i, j, self.sess.run(MaxBytesInUse())/1024/1024/1024)
                #     print(i, j, self.sess.run(BytesInUse())/1024/1024/1024)
                    # raise
                batch_idx = rng.choice(index_for_sampling, batch_size)

                self.set_training()
                _, train_ae_loss, normalized_grn_loss, mask_loss = self.sess.run(
                            [self.close_op, self.train_ae_loss, self.normalized_grn_loss, self.mask_loss],
                            feed_dict={
                                self.training_flag: True,
                                self.x: X[batch_idx],
                                self.unscale_x: unscale_X[batch_idx],
                                self.non_zero_mask: nonzero_mask[batch_idx],
                                self.x_count: count_X[batch_idx],
                                })


                self.stored_train_ae_loss.append(train_ae_loss)
                self.stored_normalized_grn_loss.append(normalized_grn_loss)
                self.stored_mask_loss.append(mask_loss)

            self.set_training(False)

            if valid_split != len(adata.X):
                if "gene-gene" in self.lambda_h:
                    valid_imX = self.sess.run(
                        [self.auto_decode_X],
                        feed_dict={
                            self.training_flag: False,
                            self.x: valid_input_X,
                            self.unscale_x: valid_input_unscale_X,
                            self.x_count: valid_input_count_X,
                            self.non_zero_mask: valid_observe_mask,
                            })
                else:
                    valid_imX = self.sess.run(
                        [self.imX],
                        feed_dict={
                            self.training_flag: False,
                            self.x: valid_input_X,
                            self.unscale_x: valid_input_unscale_X,
                            self.x_count: valid_input_count_X,
                            self.non_zero_mask: valid_observe_mask,
                            })
                valid_imX = np.squeeze(valid_imX)

                curr_valid_error = np.divide(np.sum(np.square(np.multiply(valid_count_X - valid_imX, valid_bench_mask))), np.sum(valid_bench_mask))
                self.stored_valid_mse.append(curr_valid_error)
                if min_valid_error > curr_valid_error:
                    min_valid_error = curr_valid_error
                    min_valid_error_epoch = i
                    if "gene-gene" in self.lambda_h:
                        corresponding_imX, corresponding_select_m, corresponding_h, corresponding_GRN, corresponding_gene_b, corresponding_alp = self.sess.run([self.auto_decode_X, self.select_m, self.h, self.GRN, self.gene_b, self.alp],
                                    feed_dict={
                                        self.training_flag: False,
                                        self.x: np.concatenate((X, valid_X), axis=0),
                                        self.unscale_x: np.concatenate([unscale_X, valid_unscale_X], axis=0),
                                        self.x_count: np.concatenate([count_X, valid_count_X], axis=0),
                                        self.non_zero_mask: np.concatenate([nonzero_mask, valid_nonzero_mask], axis=0)
                                    })
                    else:
                        corresponding_imX, corresponding_select_m, corresponding_h, corresponding_GRN, corresponding_gene_b, corresponding_alp = self.sess.run([self.imX, self.select_m, self.h, self.GRN, self.gene_b, self.alp],
                                    feed_dict={
                                        self.training_flag: False,
                                        self.x: np.concatenate((X, valid_X), axis=0),
                                        self.unscale_x: np.concatenate([unscale_X, valid_unscale_X], axis=0),
                                        self.x_count: np.concatenate([count_X, valid_count_X], axis=0),
                                        self.non_zero_mask: np.concatenate([nonzero_mask, valid_nonzero_mask], axis=0)
                                    })
                if min_valid_error < curr_valid_error:
                    early_stop_cnt += 1
                    if early_stop_cnt >= 1000:
                        saver.save(self.sess, osp.join(self.data_dir, self.dataset_dir, self.output_dir, 'last.{}'.format(self.seed)))
                        break
                else:
                    early_stop_cnt = 0
            else:
                if "gene-gene" in self.lambda_h:
                    corresponding_imX, corresponding_select_m, corresponding_h, corresponding_GRN, corresponding_gene_b, corresponding_alp = self.sess.run([self.auto_decode_X, self.select_m, self.h, self.GRN, self.gene_b, self.alp],
                                feed_dict={
                                    self.training_flag: False,
                                    self.x: X,
                                    self.unscale_x: unscale_X,
                                    self.x_count: count_X,
                                    self.non_zero_mask: nonzero_mask
                                })
                else:
                    continue
                    fetch_batch_size = min(16384, len(X))
                    corresponding_imX = np.empty((X.shape[0], X.shape[1]))
                    corresponding_select_m = np.empty((X.shape[0], X.shape[1]))
                    corresponding_h = np.empty((X.shape[0], self.dims[-1]))
                    corresponding_GRN, corresponding_gene_b, corresponding_alp = self.sess.run([self.GRN, self.gene_b, self.alp],
                                feed_dict={
                                    self.training_flag: False,
                                })
                    print(i, "before", self.sess.run(MaxBytesInUse())/1024/1024/1024)
                    print(i, "before", self.sess.run(BytesInUse())/1024/1024/1024)
                    for i in range(math.ceil(float(len(X)) / float(fetch_batch_size))):
                        fetch_batch_start = i * fetch_batch_size
                        fetch_batch_end = min((i + 1) * fetch_batch_size, len(X))
                        corresponding_imX[fetch_batch_start:fetch_batch_end], corresponding_select_m[fetch_batch_start:fetch_batch_end], corresponding_h[fetch_batch_start:fetch_batch_end] = self.sess.run([self.imX, self.select_m, self.h],
                                    feed_dict={
                                        self.training_flag: False,
                                        self.x: X[fetch_batch_start:fetch_batch_end],
                                        self.unscale_x: unscale_X[fetch_batch_start:fetch_batch_end],
                                        self.x_count: count_X[fetch_batch_start:fetch_batch_end],
                                        self.non_zero_mask: nonzero_mask[fetch_batch_start:fetch_batch_end]
                                    })

                    print(i, "after", self.sess.run(MaxBytesInUse())/1024/1024/1024)
                    print(i, "after", self.sess.run(BytesInUse())/1024/1024/1024)
        # toc = time.perf_counter()
        # GB_mem = self.sess.run(MaxBytesInUse())/1024/1024/1024
        # duration = toc - tic
        # print("Max memory usage: {}GB".format(GB_mem))
        # print("Time: {}s".format(duration))
        # with open(osp.join("/mnt/f/OneDrive - Hong Kong Baptist University/year1_1/cgi_datasets/tm_droplet_all", "running_perf.csv"), 'a+') as running_perf:
        # with open(osp.join("/home/comp/20481195", "running_perf.csv"), 'a+') as running_perf:
        #     running_perf.write("{},{},{},{},{}\n".format(len(count_X), self.pretrain_epochs, batch_size, GB_mem, duration))
        # raise
        self.stored_train_ae_loss = np.array(self.stored_train_ae_loss)
        self.stored_normalized_grn_loss = np.array(self.stored_normalized_grn_loss)
        self.stored_count_grn_loss = np.array(self.stored_count_grn_loss)
        self.stored_mask_loss = np.array(self.stored_mask_loss)
        self.stored_close_loss = np.array(self.stored_close_loss)
        self.stored_imputation_mse = np.array(self.stored_imputation_mse)
        self.stored_neg_sample_loss = np.array(self.stored_neg_sample_loss)
        if valid_split != len(adata.X):
            self.stored_valid_mse = np.array(self.stored_valid_mse)
            print(min_valid_error_epoch, min_valid_error)
        
        self.set_training(False)

        ### original ###
        corresponding_imX[corresponding_imX < 0] = 0
        corresponding_raw_imX = corresponding_imX

        self.raw_imX_df = pd.DataFrame(np.array(corresponding_raw_imX), index=cells_name, columns=genes_name)

        if valid_split == len(adata.X):
            valid_count_X = np.zeros((0, count_X.shape[1]))
            valid_nonzero_mask = np.zeros((0, nonzero_mask.shape[1]))
        corresponding_recover_imX = np.multiply(np.concatenate([count_X, valid_count_X], axis=0), 
                np.concatenate([nonzero_mask, valid_nonzero_mask], axis=0)) + \
            np.multiply(corresponding_imX, 
                1 - np.concatenate([nonzero_mask, valid_nonzero_mask], axis=0))

        self.recover_imX_df = pd.DataFrame(np.array(corresponding_recover_imX), index=cells_name, columns=genes_name)

        corresponding_imX = np.multiply(np.concatenate([count_X, valid_count_X], axis=0), 
                post_zero_mask) + \
            np.multiply(corresponding_imX, 
                1 - post_zero_mask)
        
        self.h_df = pd.DataFrame(np.array(np.squeeze(corresponding_h)), index=cells_name)
        self.imX_df = pd.DataFrame(np.array(corresponding_imX), index=cells_name, columns=genes_name)

        ### original ###

        # ### knn ###
        # corresponding_imX[corresponding_imX < 0] = 0
        # corresponding_raw_imX = corresponding_imX

        # self.raw_imX_df = pd.DataFrame(np.array(corresponding_raw_imX), index=cells_name, columns=genes_name)
        # nonzero_mask = get_mask(unscale_X)
        # valid_nonzero_mask = get_mask(valid_unscale_X)
        # if valid_split == len(adata.X):
        #     valid_count_X = np.zeros((0, count_X.shape[1]))
        #     valid_nonzero_mask = np.zeros((0, nonzero_mask.shape[1]))
        # corresponding_recover_imX = np.multiply(np.concatenate([count_X, valid_count_X], axis=0), 
        #         np.concatenate([nonzero_mask, valid_nonzero_mask], axis=0)) + \
        #     np.multiply(corresponding_imX, 
        #         1 - np.concatenate([nonzero_mask, valid_nonzero_mask], axis=0))

        # self.recover_imX_df = pd.DataFrame(np.array(corresponding_recover_imX), index=cells_name, columns=genes_name)
        # corresponding_imX = np.multiply(np.concatenate([count_X, valid_count_X], axis=0), 
        #         post_zero_mask) + \
        #     np.multiply(corresponding_imX, 
        #         1 - post_zero_mask)
        
        # self.h_df = pd.DataFrame(np.array(np.squeeze(corresponding_h)), index=cells_name)
        # self.imX_df = pd.DataFrame(np.array(corresponding_imX), index=cells_name, columns=genes_name)
        # ### knn ###

        
        self.select_m_df = pd.DataFrame(np.array(corresponding_select_m), index=cells_name, columns=genes_name)
        self.grn_df = pd.DataFrame(np.array(np.squeeze(corresponding_GRN)), index=genes_name, columns=genes_name)
        self.gene_b_df = pd.DataFrame(np.array(corresponding_gene_b), index=genes_name).T
        self.alp_df = pd.DataFrame(np.array(corresponding_alp), index=genes_name).T




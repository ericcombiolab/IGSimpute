import math
import os
from os import path as osp
from collections import defaultdict
from xmlrpc.client import boolean

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1.train import Saver
from tensorflow.contrib.memory_stats import MaxBytesInUse, BytesInUse
from tensorflow.keras.layers import Dense, GaussianNoise, Dropout, BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.contrib.opt import AdamWOptimizer
from utils import get_mask, get_rand_mask, get_rand_mask_efficient
from tqdm import trange
from numba import njit

@njit(parallel=True, fastmath=True)
def valid_mse(a, b, c, b_sum):
    return np.divide(np.sum(np.square(np.multiply(a - c, b))), b_sum)

def generate_alphas(m_steps=50,
                    method='riemann_trapezoidal'):
  """
  Args:
    m_steps(Tensor): A 0D tensor of an int corresponding to the number of linear
      interpolation steps for computing an approximate integral. Default is 50.
    method(str): A string representing the integral approximation method. The 
      following methods are implemented:
      - riemann_trapezoidal(default)
      - riemann_left
      - riemann_midpoint
      - riemann_right
  Returns:
    alphas(Tensor): A 1D tensor of uniformly spaced floats with the shape 
      (m_steps,).
  """
#   m_steps_float = tf.cast(m_steps, float) # cast to float for division operations.
  m_steps_float = float(m_steps) # cast to float for division operations.

  if method == 'riemann_trapezoidal':
    alphas = np.linspace(0.0, 1.0, m_steps+1) # needed to make m_steps intervals.
  elif method == 'riemann_left':
    alphas = np.linspace(0.0, 1.0 - (1.0 / m_steps_float), m_steps)
  elif method == 'riemann_midpoint':
    alphas = np.linspace(1.0 / (2.0 * m_steps_float), 1.0 - 1.0 / (2.0 * m_steps_float), m_steps)
  elif method == 'riemann_right':    
    alphas = np.linspace(1.0 / m_steps_float, 1.0, m_steps)
  else:
    raise AssertionError("Provided Riemann approximation method is not valid.")

  return alphas

def generate_path_inputs(baseline,
                         input,
                         alphas):
  """Generate m interpolated inputs between baseline and input features.
  Args:
    baseline(Tensor): A 1D tensor of floats with the shape 
      (#gene).
    input(Tensor): A 1D tensor of floats with the shape 
      (#gene).
    alphas(Tensor): A 1D tensor of uniformly spaced floats with the shape 
      (m_steps,).
  Returns:
    path_inputs(Tensor): A 2D tensor of floats with the shape 
      (m_steps, #gene).
  """
  # Expand dimensions for vectorized computation of interpolations.
  alphas_x = alphas[:, np.newaxis]
  baseline_x = np.expand_dims(baseline, axis=0)
  input_x = np.expand_dims(input, axis=0) 
  delta = input_x - baseline_x
  path_inputs = baseline_x +  alphas_x * delta
  
  return path_inputs

# def compute_gradients(model, path_inputs, target_gene_idx):
#   """Compute gradients of model predicted probabilties with respect to inputs.
#   Args:
#     mode(tf.keras.Model): Trained Keras model.
#     path_inputs(Tensor): A 2D tensor of floats with the shape 
#       (m_steps, #gene).
#     target_gene_idx(Tensor): A 0D tensor of float corresponding to the imputed gene.
#   Returns:
#     gradients(Tensor): A 2D tensor of floats with the shape 
#       (m_steps, #gene).
#   """
#   with tf.GradientTape() as tape:
#     tape.watch(path_inputs)
#     predictions = model(path_inputs)
#     # Note: IG requires softmax probabilities; converting Inception V1 logits.
#     # outputs = tf.nn.softmax(predictions, axis=-1)[:, target_gene_idx]      
#     outputs = predictions[:, target_gene_idx]      
#   gradients = tape.gradient(outputs, path_inputs)

#   return gradients

def compute_gradients(model, path_inputs, target_gene_idx):
  """Compute gradients of model predicted probabilties with respect to inputs.
  Args:
    mode(tf.keras.Model): Trained Keras model.
    path_inputs(Tensor): A 2D tensor of floats with the shape 
      (m_steps, #gene).
    target_gene_idx(Tensor): A 0D tensor of float corresponding to the imputed gene.
  Returns:
    gradients(Tensor): A 2D tensor of floats with the shape 
      (m_steps, #gene).
  """
  with tf.GradientTape() as tape:
    tape.watch(path_inputs)
    predictions = model(path_inputs)
    # Note: IG requires softmax probabilities; converting Inception V1 logits.
    # outputs = tf.nn.softmax(predictions, axis=-1)[:, target_gene_idx]      
    outputs = predictions[:, target_gene_idx]      
  gradients = tape.gradient(outputs, path_inputs)

  return gradients

def integral_approximation(gradients, 
                           method='riemann_trapezoidal'):
  """Compute numerical approximation of integral from gradients.

  Args:
    gradients(Tensor): A 2D tensor of floats with the shape 
      (m_steps, #gene).
    method(str): A string representing the integral approximation method. The 
      following methods are implemented:
      - riemann_trapezoidal(default)
      - riemann_left
      - riemann_midpoint
      - riemann_right 
  Returns:
    integrated_gradients(Tensor): A 1D tensor of floats with the shape
      (#gene).
  """
  if method == 'riemann_trapezoidal':  
    # grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    grads = (gradients[:-1] + gradients[1:]) / 2.0
  elif method == 'riemann_left':
    grads = gradients
  elif method == 'riemann_midpoint':
    grads = gradients
  elif method == 'riemann_right':    
    grads = gradients
  else:
    raise AssertionError("Provided Riemann approximation method is not valid.")

  # Average integration approximation.
#   integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  integrated_gradients = np.mean(grads, axis=0)

  return integrated_gradients

@tf.function
def integrated_gradients(model,
                         baseline, 
                         input,  
                         target_class_idx,
                         m_steps=50,
                         method='riemann_trapezoidal',
                         batch_size=32
                        ):
  """
  Args:
    model(keras.Model): A trained model to generate predictions and inspect.
    baseline(Tensor): A 1D tensor with the shape 
      (#gene) with the same shape as the input tensor.
    input(Tensor): A 1D tensor with the shape 
      (#gene).
    target_gene_idx(Tensor): An integer that corresponds to the target 
      gene index in the model's output tensor.
    m_steps(Tensor): A 0D tensor of an integer corresponding to the number of 
      linear interpolation steps for computing an approximate integral. Default 
      value is 50 steps.           
    method(str): A string representing the integral approximation method. The 
      following methods are implemented:
      - riemann_trapezoidal(default)
      - riemann_left
      - riemann_midpoint
      - riemann_right
    batch_size(Tensor): A 0D tensor of an integer corresponding to a batch
      size for alpha to scale computation and prevent OOM errors. Note: needs to
      be tf.int64 and shoud be < m_steps. Default value is 32.      
  Returns:
    integrated_gradients(Tensor): A 1D tensor of floats with the same 
      shape as the input tensor (#gene).
  """

  # 1. Generate alphas.
  alphas = generate_alphas(m_steps=m_steps,
                           method=method)

  # Initialize TensorArray outside loop to collect gradients. Note: this data structure
  # is similar to a Python list but more performant and supports backpropogation.
  # See https://www.tensorflow.org/api_docs/python/tf/TensorArray for additional details.
  gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

  # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
  # Note: this implementation opted for lightweight tf.range iteration with @tf.function.
  # Alternatively, you could also use tf.data, which adds performance overhead for the IG 
  # algorithm but provides more functionality for working with tensors and image data pipelines.
  for alpha in tf.range(0, len(alphas), batch_size):
    from_ = alpha
    to = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]

    # 2. Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = generate_path_inputs(baseline=baseline,
                                                         input=input,
                                                         alphas=alpha_batch)

    # 3. Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(model=model,
                                       path_inputs=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx)
    
    # Write batch indices and gradients to TensorArray. Note: writing batch indices with
    # scatter() allows for uneven batch sizes. Note: this operation is similar to a Python list extend().
    # See https://www.tensorflow.org/api_docs/python/tf/TensorArray#scatter for additional details.
    gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    
  
  # Stack path gradients together row-wise into single tensor.
  total_gradients = gradient_batches.stack()
    
  # 4. Integral approximation through averaging gradients.
  avg_gradients = integral_approximation(gradients=total_gradients,
                                         method=method)
    
  # 5. Scale integrated gradients with respect to input.
  integrated_gradients = (input - baseline) * avg_gradients

  return integrated_gradients


class Model(object):
    def __init__(self, data_dir, dataset_dir, output_dir, dims, learning_rate, batch_size, lambda_a, lambda_b, lambda_c, lambda_d, epochs=2000, seed=0, n_cores=-1, noise_sd=1.5, ig=False):
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

        # with tf.compat.v1.variable_scope("sc"):
        self.gene_b = tf.nn.relu(tf.compat.v1.get_variable(name="gb", shape=[self.dims[0]], dtype=tf.float32, initializer=glorot_uniform(seed=self.seed)))
        self.GRN = tf.compat.v1.get_variable(name='grn', shape=[self.dims[0], self.dims[0]], dtype=tf.float32, initializer=glorot_uniform(seed=self.seed))
        self.noise = GaussianNoise(self.noise_sd, name='noise')
        self.noised_x = self.noise(self.x, training=self.training_flag)
        if ig:
            self.noised_x = BatchNormalization()(self.noised_x, training=self.training_flag)
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
        if ig:
            self.target_gene_idx = tf.compat.v1.placeholder(tf.int32, shape=[], name="target_gene_idx")
            self.ig_post_zero_mask = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dims[0]], name="ig_post_zero_mask")
            self.gradient_batch = tf.gradients(self.imX[:, self.target_gene_idx] * (1 - self.ig_post_zero_mask[:, self.target_gene_idx]), self.x)
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

    def check_ig(self, adata, adata_unscaled, adata_cnt, gpu_option, target_genes=["Pecam1", "Ptprc"], m_steps=50, method='riemann_trapezoidal', ig_batch_size=32, post_zero_mask=None):

        if post_zero_mask is None:
            post_zero_mask = np.zeros(adata.shape, dtype=np.float32)
        full_X = adata.X.astype(np.float32)
        full_unscale_X = adata_unscaled.X.astype(np.float32)
        full_count_X = adata_cnt.X.astype(np.float32)
        full_nonzero_mask = get_mask(full_unscale_X)
        self.full_iteration_per_epoch = math.ceil(full_X.shape[0] / self.batch_size)
        full_len = len(full_X)

        cells_name = adata.obs['cell_name']
        genes_name = adata.var['gene_name']
        batch_size = self.batch_size

        # self.pretrain_epochs = 1

        if full_X.shape[0] < batch_size:
            batch_size = full_X.shape[0]

        # print("Mixed data has {} total clusters".format(n_clusters))

        # print("end the data proprocess")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option

        config_ = tf.compat.v1.ConfigProto()
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        # config_.gpu_options.per_process_gpu_memory_fraction = 0.45
        # config_.gpu_options.per_process_gpu_memory_fraction = 0.2
        config_.intra_op_parallelism_threads = 0
        config_.inter_op_parallelism_threads = 0
        # config_.log_device_placement = True
        self.sess = tf.compat.v1.Session(config=config_)
        init = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())

        self.sess.run(init)
        saver = Saver()
        saver.restore(self.sess, osp.join(self.data_dir, self.dataset_dir, self.output_dir, 'best_on_validation.{}'.format(self.seed)))
        self.set_training(False)
        target_gene_indice = genes_name.index[[genes_name.tolist().index(target_gene) for target_gene in target_genes]]
        assert len(target_gene_indice) == len(target_genes)
        target_gene_dict = dict(zip(target_genes, target_gene_indice))
        # print("target_gene_idx", target_gene_idx, type(target_gene_idx))
        # print(genes_name.head())
        alphas = generate_alphas(m_steps=m_steps, method=method)
        # print("alphas.shape", alphas.shape) # (m_steps+1,)
        # print("alphas", type(alphas), alphas)
        baseline = np.zeros(shape=(full_X.shape[1]))
        all_igs = []
        for ig_input_idx in trange(full_len):
            gradient_batches = defaultdict(list)
            integrated_gradients = dict()
            for alpha in range(0, alphas.shape[0], ig_batch_size):
                from_ = alpha
                to = np.minimum(from_ + ig_batch_size, alphas.shape[0])
                alpha_batch = alphas[from_:to]
                interpolated_path_X_batch = generate_path_inputs(baseline=baseline,
                                                                    input=full_X[ig_input_idx],
                                                                    alphas=alpha_batch)
                interpolated_path_count_X_batch = generate_path_inputs(baseline=baseline,
                                                                    input=full_count_X[ig_input_idx],
                                                                    alphas=alpha_batch)
                for target_gene, target_gene_idx in target_gene_dict.items():
                    gradient_batch = self.sess.run(self.gradient_batch,
                                feed_dict={
                                    self.training_flag: False,
                                    self.x: interpolated_path_X_batch,
                                    # self.unscale_x: full_unscale_X[ig_input_idx],
                                    self.x_count: interpolated_path_count_X_batch,
                                    self.non_zero_mask: np.broadcast_to(full_nonzero_mask[[ig_input_idx]], interpolated_path_X_batch.shape),
                                    self.target_gene_idx: target_gene_idx,
                                    self.ig_post_zero_mask: np.broadcast_to(post_zero_mask[[ig_input_idx]], interpolated_path_X_batch.shape),
                                })
                    # print("target: batch_ig_imX", batch_ig_imX.shape)
                    # print("sources: interpolated_path_X_batch", interpolated_path_X_batch.shape)
                    # print("gradient_batch:", len(gradient_batch), gradient_batch[0].shape)
                    # print("test_gradient_batch:", len(test_gradient_batch), test_gradient_batch[0].shape)
                    gradient_batches[target_gene].append(gradient_batch[0])
            for target_gene in target_gene_dict.keys():
                integrated_gradients[target_gene] = (full_X[ig_input_idx] - baseline) * integral_approximation(gradients=np.concatenate(gradient_batches[target_gene]), method=method)
            all_igs.append(integrated_gradients)
        target_gene_ig_dict ={}
        for target_gene in target_genes:
            target_gene_ig_dict[target_gene] = np.concatenate([[all_ig[target_gene]] for all_ig in all_igs])
        baseline_imX = self.sess.run(self.imX,
                    feed_dict={
                        self.training_flag: False,
                        self.x: np.zeros_like(full_X[[0]]),
                        # self.unscale_x: np.zeros_like(full_X[[0]]),
                        self.x_count: np.zeros_like(full_X[[0]]),
                        self.non_zero_mask: np.zeros_like(full_X[[0]]),
                        self.target_gene_idx: 0
                    })
        return target_gene_ig_dict, baseline_imX


    def train(self, adata, adata_unscaled, adata_cnt, post_zero_mask, valid_split, valid_dropout, rng, gpu_option, ig=False):

        X = adata.X[:valid_split].astype(np.float32)
        unscale_X = adata_unscaled[:valid_split].X.astype(np.float32)
        count_X = adata_cnt.X[:valid_split].astype(np.float32)
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
            # valid_observe_mask = get_rand_mask(valid_unscale_X, rng, valid_dropout)
            valid_observe_mask = get_rand_mask_efficient(valid_unscale_X, rng, valid_dropout, valid_nonzero_mask=valid_nonzero_mask)

            assert (valid_nonzero_mask.sum(1) == 0).sum() == 0
            assert (valid_observe_mask.sum(1) == 0).sum() == 0

            # valid_nonzero_mask = post_zero_mask[valid_split:]
            # valid_observe_mask = get_rand_mask_for_val(valid_unscale_X, rng, valid_dropout, valid_nonzero_mask)

            valid_bench_mask = valid_nonzero_mask - valid_observe_mask
            valid_bench_mask_sum = np.sum(valid_bench_mask)
            valid_input_X = np.multiply(valid_observe_mask, valid_X) + np.multiply(valid_bench_mask, np.min(valid_X, 0, keepdims=True))
            valid_input_unscale_X = np.multiply(valid_observe_mask, valid_unscale_X)
            valid_input_count_X = np.multiply(valid_observe_mask, valid_count_X)
            valid_len = len(valid_X)
            full_X = np.concatenate([X, valid_X], axis=0)
            full_unscale_X = np.concatenate([unscale_X, valid_unscale_X], axis=0)
            full_count_X = np.concatenate([count_X, valid_count_X], axis=0)
            full_nonzero_mask = np.concatenate([nonzero_mask, valid_nonzero_mask], axis=0)
            self.full_iteration_per_epoch = math.ceil(full_X.shape[0] / self.batch_size)
            full_len = len(full_X)
        else:
            full_X = X
            full_unscale_X = unscale_X
            full_count_X = count_X
            full_nonzero_mask = nonzero_mask
            self.full_iteration_per_epoch = math.ceil(full_X.shape[0] / self.batch_size)
            full_len = len(full_X)
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
        if valid_split != len(adata.X):
            self.val_iteration_per_epoch = math.ceil(len(valid_X) / batch_size)
        self.stored_train_ae_loss = []
        self.stored_normalized_grn_loss = []
        self.stored_mask_loss = []
        self.stored_imputation_mse = []
        self.stored_valid_mse = []
        # Stage: Imputation
        # index_for_sampling = range(X.shape[0])
        index_for_sampling = X.shape[0]
        min_valid_error = np.inf
        min_valid_error_epoch = 0
        early_stop_cnt = 0
        # print("begin model pretrain(Imputation)")
        # tic = time.perf_counter()
        for i in range(2):
            batch_indices = rng.choice(index_for_sampling, [self.pretrain_epochs, batch_size])
            for j in range(self.iteration_per_epoch):
                # if i == 1 and j == 1:
                #     print(self.sess.run(BytesLimit())/1024/1024/1024)
                #     print(self.sess.run(MaxBytesInUse())/1024/1024/1024)
                #     print(self.sess.run(BytesInUse())/1024/1024/1024)
                # batch_idx = rng.choice(index_for_sampling, batch_size)
                batch_idx = batch_indices[i]
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
            batch_indices = rng.choice(index_for_sampling, [self.pretrain_epochs, batch_size])
            for j in range(self.iteration_per_epoch):
                # if i == 5 and j == 1:
                #     print(i, j, self.sess.run(MaxBytesInUse())/1024/1024/1024)
                #     print(i, j, self.sess.run(BytesInUse())/1024/1024/1024)
                    # raise
                # batch_idx = rng.choice(index_for_sampling, batch_size)
                batch_idx = batch_indices[i]
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
                corresponding_valid_imX = []
                for vbi in range(self.val_iteration_per_epoch):
                    # print("vbi", vbi, vbi * self.batch_size, min((vbi + 1) * self.batch_size, valid_len))
                    val_batch_idx = np.arange(vbi * self.batch_size, min((vbi + 1) * self.batch_size, valid_len))
                    batch_corresponding_valid_imX = self.sess.run(
                        self.imX,
                        feed_dict={
                            self.training_flag: False,
                            self.x: valid_input_X[val_batch_idx],
                            self.unscale_x: valid_input_unscale_X[val_batch_idx],
                            self.x_count: valid_input_count_X[val_batch_idx],
                            self.non_zero_mask: valid_observe_mask[val_batch_idx],
                            })
                    corresponding_valid_imX.append(batch_corresponding_valid_imX)
                corresponding_valid_imX = np.concatenate(corresponding_valid_imX, axis=0)

                curr_valid_error = valid_mse(valid_count_X, valid_bench_mask, corresponding_valid_imX, valid_bench_mask_sum)
                self.stored_valid_mse.append(curr_valid_error)
                if min_valid_error > curr_valid_error:
                    min_valid_error = curr_valid_error
                    min_valid_error_epoch = i
                    if ig:
                        saver.save(self.sess, osp.join(self.data_dir, self.dataset_dir, self.output_dir, 'best_on_validation.{}'.format(self.seed)))

                    min_imX = []
                    min_select_m = []
                    min_h = []
                    for tbi in range(self.full_iteration_per_epoch):
                        batch_idx = np.arange(tbi * self.batch_size, min((tbi + 1) * self.batch_size, full_len))
                        batch_min_imX, batch_min_select_m, batch_min_h = self.sess.run([self.imX, self.select_m, self.h],
                                    feed_dict={
                                        self.training_flag: False,
                                        self.x: full_X[batch_idx],
                                        self.unscale_x: full_unscale_X[batch_idx],
                                        self.x_count: full_count_X[batch_idx],
                                        self.non_zero_mask: full_nonzero_mask[batch_idx]
                                    })
                        min_imX.append(batch_min_imX)
                        min_select_m.append(batch_min_select_m)
                        min_h.append(batch_min_h)
                    min_imX = np.concatenate(min_imX, axis=0)
                    min_select_m = np.concatenate(min_select_m, axis=0)
                    min_h = np.concatenate(min_h, axis=0)
                    min_GRN, min_gene_b, min_alp = self.sess.run([self.GRN, self.gene_b, self.alp],
                                feed_dict={
                                    self.training_flag: False,
                                    # self.x: X[[0]],
                                    # self.unscale_x: unscale_X[[0]],
                                    # self.x_count: count_X[[0]],
                                    # self.non_zero_mask: nonzero_mask[[0]]
                                })
                if min_valid_error < curr_valid_error:
                    early_stop_cnt += 1
                    if early_stop_cnt >= 1000:
                        # saver.save(self.sess, osp.join(self.data_dir, self.dataset_dir, self.output_dir, 'last.{}'.format(self.seed)))
                        break
                else:
                    early_stop_cnt = 0
            else:
                if i == self.pretrain_epochs - 1: # last epoch
                    min_imX = []
                    min_select_m = []
                    min_h = []
                    for tbi in range(self.full_iteration_per_epoch):
                        batch_idx = np.arange(tbi * self.batch_size, min((tbi + 1) * self.batch_size, full_len))
                        batch_min_imX, batch_min_select_m, batch_min_h = self.sess.run([self.imX, self.select_m, self.h],
                                    feed_dict={
                                        self.training_flag: False,
                                        self.x: full_X[batch_idx],
                                        self.unscale_x: full_unscale_X[batch_idx],
                                        self.x_count: full_count_X[batch_idx],
                                        self.non_zero_mask: full_nonzero_mask[batch_idx]
                                    })
                        min_imX.append(batch_min_imX)
                        min_select_m.append(batch_min_select_m)
                        min_h.append(batch_min_h)
                    min_imX = np.concatenate(min_imX, axis=0)
                    min_select_m = np.concatenate(min_select_m, axis=0)
                    min_h = np.concatenate(min_h, axis=0)
                    min_GRN, min_gene_b, min_alp = self.sess.run([self.GRN, self.gene_b, self.alp],
                                feed_dict={
                                    self.training_flag: False,
                                    # self.x: X[[0]],
                                    # self.unscale_x: unscale_X[[0]],
                                    # self.x_count: count_X[[0]],
                                    # self.non_zero_mask: nonzero_mask[[0]]
                                })
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
        self.stored_mask_loss = np.array(self.stored_mask_loss)
        self.stored_imputation_mse = np.array(self.stored_imputation_mse)
        if valid_split != len(adata.X):
            self.stored_valid_mse = np.array(self.stored_valid_mse)
            print(min_valid_error_epoch, min_valid_error)

        self.set_training(False)

        ### revised ###
        min_imX[min_imX < 0] = 0
        min_raw_imX = min_imX

        self.raw_imX_df = pd.DataFrame(np.array(min_raw_imX), index=cells_name, columns=genes_name)

        if valid_split == len(adata.X):
            valid_count_X = np.zeros((0, count_X.shape[1]))
            valid_nonzero_mask = np.zeros((0, nonzero_mask.shape[1]))
        min_recover_imX = np.multiply(np.concatenate([count_X, valid_count_X], axis=0), 
                np.concatenate([nonzero_mask, valid_nonzero_mask], axis=0)) + \
            np.multiply(min_imX, 
                1 - np.concatenate([nonzero_mask, valid_nonzero_mask], axis=0))

        self.recover_imX_df = pd.DataFrame(np.array(min_recover_imX), index=cells_name, columns=genes_name)

        min_imX = np.multiply(np.concatenate([count_X, valid_count_X], axis=0), 
                post_zero_mask) + \
            np.multiply(min_imX, 
                1 - post_zero_mask)
        
        self.h_df = pd.DataFrame(np.array(np.squeeze(min_h)), index=cells_name)
        self.imX_df = pd.DataFrame(np.array(min_imX), index=cells_name, columns=genes_name)
        ### revised ###

        # ### original ###
        # corresponding_imX[corresponding_imX < 0] = 0
        # corresponding_raw_imX = corresponding_imX

        # self.raw_imX_df = pd.DataFrame(np.array(corresponding_raw_imX), index=cells_name, columns=genes_name)

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

        # ### original ###

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


        # self.select_m_df = pd.DataFrame(np.array(corresponding_select_m), index=cells_name, columns=genes_name)
        # self.grn_df = pd.DataFrame(np.array(np.squeeze(corresponding_GRN)), index=genes_name, columns=genes_name)
        # self.gene_b_df = pd.DataFrame(np.array(corresponding_gene_b), index=genes_name).T
        # self.alp_df = pd.DataFrame(np.array(corresponding_alp), index=genes_name).T
        self.select_m_df = pd.DataFrame(np.array(min_select_m), index=cells_name, columns=genes_name)
        self.grn_df = pd.DataFrame(np.array(np.squeeze(min_GRN)), index=genes_name, columns=genes_name)
        self.gene_b_df = pd.DataFrame(np.array(min_gene_b), index=genes_name).T
        self.alp_df = pd.DataFrame(np.array(min_alp), index=genes_name).T




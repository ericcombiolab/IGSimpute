# IGSimpute

IGSimpute is an accurate and interpretable imputation method for recovering missing values in scRNA-seq data with an interpretable instance-wise gene selection layer.

## Requirements

Download with

```
$ git clone https://github.com/ker2xu/IGSimpute
```

You need to create an enviroment named `IGSimpute` using `conda` from `environment.yml`.

```
$ conda env create -f environment.yml
```

## Usage

You need to first activate the environment by:

`conda activate IGSimpute`

You can run the command below to perform imputation on the heart-and-aorta tissue in the Tabula Muris atlas:

`./run_IGSimpute.sh`

If you want to perform imputation on your own dataset, you need to modify parameters defined in `run_IGSimpute.sh`.

### Parameters
Name | Default value | Description 
------------ | ------------ | ------------
data_dir | - | Path to the directory that contains all datasets.
dataset_dir | 'tm_droplet_Heart_and_Aorta' | Directory name of the dataset to be imputed.
exp_file_name | 'X.csv' | Expression file name.
output_dir | 'imputation_output' | Output directory name.
hg | '0.1' | The percentage or the number of used highly variable genes.
epochs | 100 | The maximum allowed epochs.
split_pct | "0.8" | The percentage of cells used as training dataset, and the left will be used for validtaion.
target_format | "count" | The expected output format.
ggl_loss_weight | "1" | The weight of $L_{gg}$.
gsl_L1_weight | "0.01 | The weight of $L_{gs}$.
rec_loss_weight | "0.1 | The weight of $L_{rec}$.
batch_size | 256 | Minibatch size.
dim | 400 | Size of the innermost embedding.
encoder_dropout_rate | "0.2" | Dropout rate of the dropout layer in the encoder part.
gpu_node | 0 | The index of GPU to use.
low_expression_percentage | 0.80 | If the percentage of low expression neighbor entries exceeds low_expression_percentage, the gene expression target entry will be changed to zero.
low_expression_threshold | 0.20 | All zero entries, and non-zero entries with expression less than low_expression_threshold quantile will be taken as low expression entires in the KNN post-processing.
lr | "1e-4" | Learning rate.
seed | 0 | Seed number.
sub_sampling_num | "None" | Randomly select sub_sampling_num cells for training and validation.
valid_dropout | "0.2" | The percentage of non-zero entries to be used for validation.

### Expected input format
IGSimpute accepts expression profiles in `h5ad` or `csv` format. Each row should correspond to a cell and each column should correspond to a gene.

## Results
The output will be put inside `"data_dir/dataset_dir/imputation_output"` directory. `"IGSimpute.name.csv.gz"` is the imputed expression matrix without KNN post-processing and `"IGSimpute.KNN.name.csv"` is the imputed expression matrix wit KNN post-processing.



source ~/miniconda3/etc/profile.d/conda.sh
conda activate IGSimpute

data_dir="/mnt/f/OneDrive - Hong Kong Baptist University/year1_1/cgi_datasets"
data_dir="/mnt/e"
#data_dir="/path/to/data_dir"
dataset_dir='tm_droplet_Heart_and_Aorta'
exp_file_name="need_to_be_specified_when_using_customized_dataset"
output_dir="imputation_output"

hg="0.1"
epochs=500000
split_pct="0.8"
target_format="count"

ggl_loss_weight="1"
gsl_L1_weight="0.01"
rec_loss_weight="0.1"

batch_size=256
dim=400
encoder_dropout_rate="0.2"
gpu_node=0
low_expression_percentage=0.80
low_expression_threshold=0.20
lr="1e-4"
seed=0
sub_sampling_num="None"
valid_dropout="0.2"

# checklist
#exp_name
#output_dir
#early_stop_threshold
#what_to_recover att_network.py
#what_to_save run.py


if [ ! -d "$data_dir/$dataset_dir/$output_dir" ]
then
    mkdir -p "$data_dir/$dataset_dir/$output_dir"
fi

PYTHONHASHSEED=0 python run.py --generate-files --highly-variable-genes $hg --data-dir "$data_dir" --bench-dataset "$dataset_dir" --exp-file-name "$exp_file_name" --seed $seed --epochs 0

python generate_marker_zero_masks.py --highly-variable-genes $hg --data-dir "$data_dir" --bench-dataset "$dataset_dir"

exp_name="seed$seed"
PYTHONHASHSEED=$seed python run.py --highly-variable-genes $hg --dim $dim --data-dir "$data_dir" --bench-dataset "$dataset_dir" --exp-file-name "$exp_file_name" --output-dir "$output_dir" --output-file-prefix "our.$exp_name" --seed $seed --ggl-loss-weight $ggl_loss_weight --gsl-L1-weight $gsl_L1_weight --rec-loss-weight $rec_loss_weight --encoder-dropout-rate $encoder_dropout_rate --epochs $epochs --gpu_node $gpu_node --valid-dropout $valid_dropout --low-expression-percentage $low_expression_percentage --low-expression-threshold $low_expression_threshold --lr $lr --batch-size $batch_size --split-percentage $split_pct --sub-sampling-num $sub_sampling_num

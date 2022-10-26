

data_path="/data/data/molecule/molecular_property_prediction/"  # replace to your data path
# save_dir="/data/models/unimol/save/1026/"  # replace to your save path
tsb_dir="./tsb"
n_gpu=4
MASTER_PORT=10086
dict_name="dict.txt"
weight_path="/data/models/unimol/save/1026/checkpoint_best.pt"  # replace to your ckpt path
task_name="qm9dft"  # molecular property prediction task name 
task_num=3
loss_func="finetune_smooth_mae"
lr=1e-4
batch_size=1000
epoch=0
dropout=0
warmup=0.06
only_polar=0
conf_size=11
seed=0
results_path="./results"

if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ]; then
	metric="valid_agg_mae"
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES="0" python ./unimol/infer.py --user-dir ./unimol $data_path --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task mol_finetune --loss $loss_func --arch unimol_base  \
       --classification-head-name $task_name --num-classes $task_num \
       --task-name $task_name \
       --path $weight_path \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --only-polar $only_polar --dict-name $dict_name \
       --log-interval 50 --log-format simple \
       --reg

# --reg, for regression task
# --maximize-best-checkpoint-metric, for classification task

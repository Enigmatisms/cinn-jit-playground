n_batch=$1
n_height=$2
n_channel=$3
log_path=$4
cuda_device=$5

CUDA_VISIBLE_DEVICES=$cuda_device ../build/trial_batched $n_batch $n_height $n_channel >> $log_path
echo "[`date`] Finished ($n_batch, $n_height, $n_channel) on (device:$cuda_device)" >> $log_path

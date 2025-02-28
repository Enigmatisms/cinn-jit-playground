n_batch=$1
n_height=$2
n_channel=$3
log_path=$4
cuda_device=7

output_path="./prof-cmd"

if [ ! -d $output_path ]; then
        mkdir -p $output_path
fi

CUDA_VISIBLE_DEVICES=$cuda_device /usr/local/NVIDIA-Nsight-Compute/ncu \
        --launch-skip 11 \
        --section MemoryWorkloadAnalysis \
        --section Occupancy \
        --section SpeedOfLight \
        --section LaunchStats \
        bash ./run_batched.sh $n_batch $n_height $n_channel $log_path $cuda_device \
        &> "$output_path/prof-$n_batch-$n_height-$n_height-$n_channel.log"
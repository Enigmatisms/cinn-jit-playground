cuda_device=7

CUDA_VISIBLE_DEVICES=$cuda_device /usr/local/NVIDIA-Nsight-Compute/ncu --set full \
    --export "./isolated-test" \
    --force-overwrite \
    --call-stack \
    --import-source yes \
    --launch-skip 11 \
    ../build/trial_batched 
echo "Starting BLIP2 Multi-GPU Training..."
echo "Using GPUs: 0,1,2,3"
echo "Model: Salesforce/blip2-opt-2.7b"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=29500


torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29500 \
    safe_multi_gpu_training.py


echo "Training completed!"


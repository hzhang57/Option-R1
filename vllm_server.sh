CUDA_VISIBLE_DEVICES=5 \
        swift rollout \
        --model Qwen/Qwen3-VL-4B-Instruct \
        --vllm_data_parallel_size 1

#        --model Qwen/Qwen2.5-VL-3B-Instruct \

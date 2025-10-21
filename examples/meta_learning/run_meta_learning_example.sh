#!/bin/bash

# Vanilla meta-learning example
# This demonstrates the simplest possible setup for meta-learning training

set -ex

# # Clean up previous runs
# pkill -9 sglang || true
# sleep 3
# ray stop --force || true
# pkill -9 ray || true
# pkill -9 python || true
# sleep 3

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh"
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-0.5B.sh"
source /data/slime/.env


# add megatron to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/root/Megatron-LM

# Model configuration
CKPT_ARGS=(
   # --hf-checkpoint Qwen/Qwen3-4B
   --hf-checkpoint /data/Qwen2-0.5B-Instruct/
   --ref-load /data/Qwen2-0.5B-Instruct-mt/
   --student-checkpoint /data/Qwen2-0.5B-Instruct/
   # --load /root/Qwen3-4B_slime/
   # --save /root/Qwen3-4B_slime/
   # --save-interval 20
)

TRAIN_IPB=16
HELDOUT_IPB=4
SPI=2
N_PROBLEMS_PER_PROMPT=4
TOTAL_IPB=$(($TRAIN_IPB + $HELDOUT_IPB))
INNER_ITERS=2
GBS=$((TRAIN_IPB / N_PROBLEMS_PER_PROMPT * SPI))

# Meta-learning configuration (minimal setup with shared inference pool)
META_LEARNING_ARGS=(
   --use-meta-learning
   --num-inner-iterations $INNER_ITERS
   --n-samples-per-prompt $SPI
   --heldout-ipb $HELDOUT_IPB
   --n-problems-per-prompt $N_PROBLEMS_PER_PROMPT
   --openai-api-key ${OPENAI_API_KEY}

   --grader-concurrency 16

   --teacher-temperature 0.7
   --teacher-max-tokens 2048
   --teacher-concurrency 16

   --rubric-key rubric
   --additional-info-position suffix
)

# Training data configuration
ROLLOUT_ARGS=(
   # --prompt-data /data/slime/gsm8k/train.parquet
   # --label-key label
   --prompt-data /data/handshake_data.jsonl
   --input-key messages
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 100
   --rollout-batch-size $TOTAL_IPB
   --rollout-max-response-len 1024
   --rollout-temperature 0.8
   --global-batch-size $GBS
   --sglang-cuda-graph-bs 1 256
)

# Evaluation configuration
# EVAL_ARGS=(
#    --eval-interval 20
#    --eval-prompt-data test_set /path/to/your/test_data.parquet
#    --n-samples-per-eval-prompt 1
#    --eval-max-response-len 1024
# )

# Performance configuration (minimal)
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

# GRPO training configuration
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.0
   --eps-clip 0.2
)

# Optimizer configuration
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
)

# SGLang configuration
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-attention-backend flashinfer
)

LAYOUT_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   --rollout-num-gpus 8
   --rollout-num-gpus-per-engine 2
   --student-num-gpus 8
   --student-num-gpus-per-engine 2
   --colocate
)

# # Launch Ray cluster
# ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 /data/slime/train.py \
   --calculate-per-token-loss \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${META_LEARNING_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${LAYOUT_ARGS[@]}

#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2112}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/hy-tmp/distillm"}

#CKPT_NAME=${4-"gpt2_kd/t3/768"} #24.88

#CKPT_NAME=${4-"gpt2_kd/t3/576"} #

#CKPT_NAME=${4-"hr0.01_no_rdistill_0.1B_sft1.5B_off_epochs8_ratio0.7-1_tmp1-4/t3/1416"} #25.2
#CKPT_NAME=${4-"hr0.01_no_jsd_0.1B_sft1.5B_on_epochs8_ratio0.7-1_tmp1-4/t3/944"}
#CKPT_NAME=${4-"hr0.01_no_jsd_0.1B_sft1.5B_off_epochs8_ratio0.7-1_tmp1-2/t3/890"} #27.8 #25.99
#CKPT_NAME=${4-"hr0.01_no_jsd_0.1B_sft1.5B_off_epochs8_ratio0.7-1_tmp1-4/t3/890"} #26.2 #23.57
#CKPT_NAME=${4-"jsd_0.1B_sft1.5B_off_epochs20_tmp1/3560"} #26.6 #25.19
CKPT_NAME=${4-"gpt2-base"} #

CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
# data
DATA_NAMES="dolly"
DATA_DIR="${BASE_PATH}/data/dolly"
# hp
EVAL_BATCH_SIZE=16
# runtime
SAVE_PATH="${BASE_PATH}/processed_data/dolly/full/generate_data/"
TYPE="eval_main"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type gpt2"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --type ${TYPE}"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 0.01"


export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/tools/generate_data.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}

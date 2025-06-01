BASE_PATH=${1-"/home/pocl"}

export TF_CPP_MIN_LOG_LEVEL=3


PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path ${BASE_PATH}/checkpoints/gpt2-base \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type gpt2

bash ${BASE_PATH}/scripts/gpt2/tools/generate_data_dolly.sh

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/rank.py \
    --generate-data-dir ${BASE_PATH}/processed_data/dolly/full/generate_data/dolly-512/gpt2-base/10/answers.jsonl \
    --truth-data-dir ${BASE_PATH}/processed_data/dolly/full/gpt2/train.jsonl \
    --save-dir ${BASE_PATH}/processed_data/dolly/full/gpt2/\
    --model-name gpt2-base\
    --num-stages 4

python3 <<EOF
import os
import shutil
num_stages=4
for i in range(num_stages):
    output_dir = os.path.join('${BASE_PATH}/processed_data/dolly/full/gpt2', 'train'+str(i))
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy('${BASE_PATH}/processed_data/dolly/full/gpt2/valid_0.bin', output_dir)
    shutil.copy('${BASE_PATH}/processed_data/dolly/full/gpt2/valid_0.idx', output_dir)
    shutil.copy('${BASE_PATH}/processed_data/dolly/full/gpt2/valid.jsonl', output_dir)
EOF

for stage in {0..3}; do  
    PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly_stage.py \
        --data-dir ${BASE_PATH}/processed_data/dolly/full/gpt2/${stage}/ \
        --processed-data-dir ${BASE_PATH}/processed_data/dolly/full/gpt2/train${stage} \
        --model-path ${BASE_PATH}/checkpoints/gpt2-base \
        --data-process-workers 32 \
        --max-prompt-length 256 \
        --dev-num 1000 \
        --model-type gpt2
done 





	


    
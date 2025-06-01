#!/bin/bash

BASE_PATH=${1-"/home/pocl"}

bash ${BASE_PATH}/scripts/gpt2/pocl/train_0.1B_1.5B/t0.sh
bash ${BASE_PATH}/scripts/gpt2/pocl/train_0.1B_1.5B/t1.sh
bash ${BASE_PATH}/scripts/gpt2/pocl/train_0.1B_1.5B/t2.sh
bash ${BASE_PATH}/scripts/gpt2/pocl/train_0.1B_1.5B/t3.sh

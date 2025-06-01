#!/bin/bash

BASE_PATH=${1-"/home/pocl"}

bash ${BASE_PATH}/scripts/opt/l2m/train_0.3B_2.7B/t0.sh
bash ${BASE_PATH}/scripts/opt/l2m/train_0.3B_2.7B/t1.sh
bash ${BASE_PATH}/scripts/opt/l2m/train_0.3B_2.7B/t2.sh
bash ${BASE_PATH}/scripts/opt/l2m/train_0.3B_2.7B/t3.sh

base_path=${1-"/home/pocl"}
port=2040


for data in dolly self_inst vicuna sinst uinst   
do
    # Evaluate     
    for seed in 10 20 30 40 50
    do
        ckpt="opt_pocl"
        bash ${base_path}/scripts/opt/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 16
    done

    
done
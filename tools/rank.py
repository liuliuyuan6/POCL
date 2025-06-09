import json
from utils import read_json_file,RougeL
from arguments import get_args
import os


def main():
    args = get_args()

    hybrid_data=read_json_file(args.truth_data_dir)
    generate_data=read_json_file(args.generate_data_dir)

    for i in range(len(hybrid_data)): 
        hybrid_data[i][args.model_name]=generate_data[i]['LLM_output']
        hybrid_data[i]['Rouge_L']=RougeL(hybrid_data[i][args.model_name],hybrid_data[i]['output'])
        hybrid_data[i]['lm_loss']=generate_data[i]['lm_loss']

    sorted_hybrid_data = sorted(hybrid_data, key=lambda x: x["Rouge_L"], reverse=True)
    for i in range(len(sorted_hybrid_data)):
        sorted_hybrid_data[i]['RoughL_rank']=i+1

    sorted_hybrid_data = sorted(sorted_hybrid_data, key=lambda x: x["lm_loss"], reverse=False)
    for i in range(len(sorted_hybrid_data)):
        sorted_hybrid_data[i]['lmloss_rank']=i+1

    for i in range(len(sorted_hybrid_data)):
        sorted_hybrid_data[i]['hybrid_rank_score']=1/(int(sorted_hybrid_data[i]['RoughL_rank'])+60)+1/(int(sorted_hybrid_data[i]['lmloss_rank'])+60)

    sorted_hybrid_data = sorted(sorted_hybrid_data, key=lambda x: x["hybrid_rank_score"], reverse=True)
    for i in range(len(sorted_hybrid_data)):
        sorted_hybrid_data[i]['rank']=i+1

    with open(os.path.join(args.save_dir, "sorted_train.jsonl"), 'w') as f:
        for item in sorted_hybrid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")  

    num_stages=args.num_stages
    each_stage_num=int(len(sorted_hybrid_data)/num_stages)

    for stage in range(num_stages-1):
        stage_data=sorted_hybrid_data[:(stage+1)*each_stage_num]
        output_dir = os.path.join(args.save_dir, str(stage))
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"raw.jsonl"), "w") as f:
            for item in stage_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")  # 保持中文不乱码


    stage_data=sorted_hybrid_data[:]
    output_dir = os.path.join(args.save_dir, str(num_stages-1))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"raw.jsonl"), "w") as f:
        for item in stage_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")  # 保持中文不乱码  
    

    
if __name__ == "__main__":
    main()
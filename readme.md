## Being Strong Progressively! Enhancing Knowledge Distillation of Large Language Models through a Curriculum Learning Framework

We propose POCL, a plug-and80 play curriculum learning framework inspired by the "progressive overload" principle in strength training. The framework leverages the student model to assess and rank sample difficulty using reciprocal rank fusion, partitioning the dataset into easy-to-hard subsets. A training scheduler—referred to as Baby Step—iteratively expands the training set, starting from the simplest samples and gradually incorporating more difficult ones after fixed intervals or convergence, until the full dataset is utilized.

<img src="framwork.png" alt="workflow" style="width: 800px; height: 500px;">

POCL enhances student LLM performance across diverse generative tasks, such as instruction following and text summarization, under various white-box KD settings, while introducing minimal additional computational overhead .
## 1 Requirements

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install transformers==4.47.0
pip install vllm==0.5.0
pip install deepspeed==0.16.5
pip install nltk==3.9.1
pip install numerize==0.12
pip install rouge-score==0.1.2
pip install torchtyping==0.1.5
pip install rich==14.0.0
pip install accelerate==1.2.1
pip install datasets==3.2.0
pip install sentencepiece
pip install protobuf==4.23.4
pip install peft==0.14.0
```
or
```bash
bash install.sh
```

Please download the GPT2 and OPT pretrained model checkpoints and put them in the checkpoints/ folder before running the training or evaluation scripts. Additionally, you need to fine-tune the teacher model on the Dolly dataset before proceeding with the distillation process. 

## 2 Data Processing

The raw datasets for Dolly, Self-Inst, Vicuna, SInst, and UInst are stored in the data/ directory. Before running the training or evaluation scripts, please preprocess the datasets using the scripts:

```bash
bash scripts/gpt2/tools/process_data_dolly_pocl.sh 
bash scripts/opt/tools/process_data_dolly_pocl.sh
```
These scripts will generate processed data in the appropriate directories under processed_data/.
## 3 Training

We provide example commands for GPT-2 models. Similar scripts for model families can be found in `scripts/opt`. All our experiments are conducted on 4 \* A800-80G, which can be reduced for small models.


### 3.1 Baselines
The final checkpoints are selected by the Rouge-L scores.
#### Fine-tune the teacher models
```bash
bash scripts/gpt2/sft/sft_xlarge.sh /PATH_TO/pocl
```
#### SFT Baseline
```bash
bash scripts/gpt2/sft/sft_base.sh /PATH_TO/pocl
```

#### SeqKD Baseline
```bash
bash scripts/gpt2/seqkd/seqkd_base.sh /PATH_TO/pocl
```

#### GKD Baseline
```bash
bash scripts/gpt2/g/gkd_base.sh /PATH_TO/pocl
```

#### KD series Baselines
```bash
bash scripts/gpt2/kd/kd_base.sh --type kd
bash scripts/gpt2/kd/kd_base.sh --type rkl
bash scripts/gpt2/kd/kd_base.sh --type jsd
bash scripts/gpt2/kd/kd_base.sh --type tvd
bash scripts/gpt2/kd/kd_base.sh --type sfkl
bash scripts/gpt2/kd/kd_base.sh --type srkl
```


### 3.2 POCL

The final checkpoints are selected by the Rouge-L scores.
```bash
bash scripts/gpt2/pocl/train_0.1B_1.5B/train_0.1B_1.5B.sh --type tfkl
bash scripts/gpt2/pocl/train_0.1B_1.5B/train_0.1B_1.5B.sh --type trkl
bash scripts/gpt2/pocl/train_0.1B_1.5B/train_0.1B_1.5B.sh --type tjsd
bash scripts/gpt2/pocl/train_0.1B_1.5B/train_0.1B_1.5B.sh --type ttvdf
bash scripts/gpt2/pocl/train_0.1B_1.5B/train_0.1B_1.5B.sh --type tsfkl
bash scripts/gpt2/pocl/train_0.1B_1.5B/train_0.1B_1.5B.sh --type tsrkl
```


## 4 Run Evaluation
```bash
bash scripts/gpt2/eval/run_eval.sh /PATH_TO/pocl
```


## 5 Results

### 🧾 Experiment Results

The training and evaluation results for the **GPT2-Base** model using **FKL** as the loss function under the **POCL framework** are stored in:

- `results_gpt2_fkl_dolly_training/`  
- `results_gpt2_fkl_dolly_eval/`

The following table compares the performance of various distillation methods on different evaluation datasets, using GPT2-1.5B as the teacher model and GPT2-0.1B as the student model.


| Teacher       | Student        | Model                | DollyEval | SelfInst | VicunaEval | S-NI   | UnNI   | Avg.   |
|---------------|----------------|----------------------|-----------|----------|------------|--------|--------|--------|
| GPT2-1.5B     | GPT2-0.1B      | teacher             | 27.19     | 14.04    | 16.47      | 27.66  | 31.86  | 23.44  |
|               |                | SFT                 | 23.33     | 10.01    | 14.72      | 16.38  | 19.57  | 16.80  |
|               |                | SeqKD               | 23.70      | 11.23    | 14.31      | 16.48  | 19.81  | 17.11  |
|               |                | KLD                 | 23.49     | 10.33    | 14.96      | 19.70   | 22.01  | 18.10  |
|               |                | KLD+POCL             | 24.87     | 11.56    | 16.13      | 21.59  | 24.34  | 19.70  |
|               |                | RKLD                | 23.79     | 12.13    | 14.94      | 23.81  | 22.52  | 19.44  |
|               |                | RKLD+POCL            | 25.01     | 12.76    | 16.01      | 25.63  | 25.42  | 20.97  |
|               |                | JSD                 | 24.07     | 11.38    | 15.87      | 22.84  | 23.06  | 19.44  |
|               |                | JSD+POCL             | 25.97     | 11.74    | 16.77      | 26.61  | 25.21  | 21.26  |
|               |                | tv_distance         | 24.32     | 11.09    | 15.51      | 25.90   | 26.55  | 20.67  |
|               |                | tv_distance+POCL     | 25.35     | 13.19    | 16.17      | 28.98  | 30.09  | 22.76  |
|               |                | skewKLD             | 24.24     | 12.27    | 15.70       | 23.33  | 24.02  | 19.91  |
|               |                | skewKLD+POCL        | 25.87    | 13.08    | 16.45      | 28.35  | 28.79  | 22.51  |
|               |                | reverse_skewKLD     | 25.22     | 12.86    | 15.18      | 25.50   | 28.43  | 21.44  |
|               |                | reverse_skewKLD+POCL | 26.17     | 13.28    | 16.66      | 28.49  | 30.12  | 22.94  |
|               |                | GKD(on policy)      | 24.67     | 11.48    | 15.66      | 23.80   | 25.26  | 20.17  |
|               |                | GKD(on policy)+POCL  | 26.60      | 12.62    | 16.70       | 27.02  | 29.61  | 22.51  |

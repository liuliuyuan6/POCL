export NCCL_DEBUG=""

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
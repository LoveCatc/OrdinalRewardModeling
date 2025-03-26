# Reward Modeling with Ordinal Feedback: Wisdom of the Crowd

This repository contains the code for the paper "Reward Modeling with Ordinal Feedback: Wisdom of the Crowd".

[Arxiv: 2411.12843](https://arxiv.org/abs/2411.12843)

## Setup

The code is tested on Ubuntu 22.04 with Python 3.10 and cuda 12.1. We run the experiments on A6000 and A100-80G GPU servers. Please make sure you have installed cuda >= 11.6 and satisfy the minimal requirement for flash attention.
You can use either
```bash
conda env create -f py310_env.yaml
```
to directly create the conda environment or manually install the required packages by running
```bash
conda create -n rm_dev python=3.10.15
conda activate rm_dev    

pip3 install torch==2.1.2 torchvision torchaudio 
pip3 install numpy==1.26.4
pip3 install flash-attn==2.6.3
pip3 install accelerate==0.33.0 
pip3 install deepspeed==0.12.2
pip3 install transformers==4.43.4
pip3 install wandb peft click datasets sentencepiece bitsandbytes rewardbench loguru
pip3 install "fschat[model_worker,webui]"
pip3 install "huggingface_hub[cli]"
```

Then please login the wandb account by running `wandb login` and huggingface account by running `huggingface-cli login`.

## Usage

We use json configs to manage the experiment settings. You can find all the experiment configs in `paper_experiment_configs/`. To reproduce, first prepare the oracle-annotated dataset by running 
```bash
python prepare_oracle_data.py
```
It would download and annotate the dataset then save it in `statdata/prefer_skywork_Skywork/Skywork-Reward-Gemma-2-27B-v0.2`. You can also pull it from huggingface hub by running
```bash
python prepare_oracle_data.py --built_from hub
```

To run the experiments, use `torchrun` or `accelerate`:
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 run.py PATH/TO/CONFIG.JSON --seed 42   # for single gpu
accelerate launch run.py --mixed_precision=bf16 PATH/TO/CONFIG.JSON --seed 42  # for multi-gpu
```
We recommend running our experiments with the number of GPUs mentioned in config names to ensure the correct batch sizes. 

If you have any questions, feel free to open an issue or contact us via [email](mailto:py4contact@gmail.com).

## Acknowledgements
This codebase is built on top of [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/bradley-terry-rm). Special thanks to its creators for their valuable contributions and insights.

## Citation
If you find this code useful for your research, please consider citing:
```
@article{liu2024rewardordinal,
  title={Reward Modeling with Ordinal Feedback: Wisdom of the Crowd},
  author={Liu, Shang and Pan, Yu and Chen, Guanting and Li, Xiaocheng},
  journal={arXiv preprint arXiv:2411.12843},
  year={2024}
}
```
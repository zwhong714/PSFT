<div align="center">
<h1>Proximal Supervised Fine-Tuning
</h1>
Wenhong Zhu<sup>1,2</sup>, Ruobing Xie<sup>3</sup>, Rui Wang<sup>1,2</sup>, Xingwu Sun <sup>3,4</sup>, Di Wang<sup>3</sup>,  Pengfei Liu <sup>1,2</sup>

<sup>1</sup> SJTU,   <sup>2</sup>SII, <sup>3</sup>Tencent, <sup>4</sup>UM

[<a href="https://github.com/zwhong714/PSFT">Paper</a>] | [<a href="https://github.com/zwhong714/PSFT">Code</a>] | [<a href="https://huggingface.co/wh-zhu">Model</a>]
</div>

# 📖 Overview


## 1. No entropy collapse
<p align="center">
  <img src="./img/qwen_entropy.png" alt="图1" width="45%"/>
  <img src="./img/llama-entropy.png" alt="图2" width="45%"/>
</p>

## 2. Superious Performance

<p align="center">
  <img src="./img/qwen-acc.png" alt="图1" width="45%"/>
  <img src="./img/llama-acc.png" alt="图2" width="45%"/>
</p>


## 3. Generalization

<p align="center">
  <img src="./img/qwen_gpqa.png" alt="图1" width="45%"/>
  <img src="./img/llama-gpqa.png" alt="图2" width="45%"/>
</p>

## 4. A promising start point for RL 

<p align="center">
  <img src="./img/qwen_rl_acc.png" alt="图1" width="45%"/>
  <img src="./img/llama-rl-acc.png" alt="图2" width="45%"/>
</p>


**For a more detailed and comprehensive evaluation, please refer to our paper.**


# ⚒️ Installation

torch2.6.0+cu124+vllm0.8.5

```
git clone https://github.com/zwhong714/PSFT
cd PSFT

conda create -n psft python==3.10
conda activate psft
cd verl
pip install --no-deps -e .
```

# 🚀 Quick Start

## Prepare Data


## Training

## Evaluation

```
cd evaluation
serve run eval.llm:build_app model=aaa/bbb/ccc tensor-parallel-size=1

# open another terminal
python eval/eval.py --temperature 0.7 --top_p 0.95 --max_tokens 10240 --model ccc --test_file eval/data/aime-2024.parquet
```

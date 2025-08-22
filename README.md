<div align="center">
<h1>Proximal Supervised Fine-Tuning
</h1>
Wenhong Zhu<sup>1,2</sup>, Ruobing Xie<sup>3</sup>, Rui Wang<sup>1,2</sup>, Xingwu Sun <sup>3,4</sup>, Di Wang<sup>3</sup>,  Pengfei Liu <sup>1,2</sup>

<sup>1</sup> SJTU,   <sup>2</sup>SII, <sup>3</sup>Tencent, <sup>4</sup>UM

[<a href="https://github.com/zwhong714/PSFT">Paper</a>] | [<a href="https://github.com/zwhong714/PSFT">Code</a>] | [<a href="https://huggingface.co/wh-zhu">Model</a>]
</div>

# 📖 Overview
The left figure presents PSFT applied to Qwen2.5-7B-Instruct, whereas the right corresponds to Llama3.1-8B-Instruct.

## 1. No entropy collapse
<p align="center">
  <img src="./img/qwen_entropy.png" alt="Qwen2.5-7B-Instruct" width="45%"/>
  <img src="./img/llama-entropy.png" alt="LLama3.1-8B-Instruct" width="45%"/>
</p>

## 2. Superior Performance

<p align="center">
  <img src="./img/qwen-acc.png" alt="Qwen2.5-7B-Instruct" width="45%"/>
  <img src="./img/llama-acc.png" alt="LLama3.1-8B-Instruct" width="45%"/>
</p>


## 3. Generalization

<p align="center">
  <img src="./img/qwen_gpqa.png" alt="Qwen2.5-7B-Instruct" width="45%"/>
  <img src="./img/llama-gpqa.png" alt="LLama3.1-8B-Instruct" width="45%"/>
</p>

## 4. A promising start point for RL 

<p align="center">
  <img src="./img/qwen_rl_acc.png" alt="Qwen2.5-7B-Instruct" width="45%"/>
  <img src="./img/llama-rl-acc.png" alt="LLama3.1-8B-Instruct" width="45%"/>
</p>


**For a more detailed and comprehensive evaluation, please refer to our paper.**


# 🧸 A Toy Example

We apply both SFT and PSFT on **Qwen2.5-7B-Instruct** using the **LIMO** dataset.

<p align="center">
  <img src="./img/limo_entropy.png" alt="Qwen2.5-7B-Instruct" width="45%"/>
  <img src="./img/limo_acc.png" alt="LLama3.1-8B-Instruct" width="45%"/>
</p>

We select the best in-domain checkpoint (216 steps, 18 epochs) and evaluate it on multiple benchmarks. Evaluation is reported with **Avg\@32**, while generation is conducted at **32k** context length.

| Qwen2.5-7B-Instruct | AIME-24   | AIME-25   | AMC       |
| ------------------- | --------- | --------- | --------- |
| SFT                 | 14.69     | 15.42     | 56.64     |
| PSFT                | **15.94** | **18.13** | **58.05** |

| Qwen2.5-7B-Instruct | GPQA      | MMLU-Pro  |
| ------------------- | --------- | --------- |
| SFT                 | 35.80     | 46.50     |
| PSFT                | **38.89** | **51.28** |

**Conclusion.** PSFT consistently outperforms standard SFT across both in-domain and out-of-domain benchmarks. Importantly, it maintains stable entropy throughout training—whereas standard SFT rapidly collapses to near-zero entropy after ~150 steps—thereby preserving diversity in generation and providing a stronger foundation for subsequent RL-based optimization.



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

## Prepare Train Data

`python ./prepare_data.py`

You can modify this file to support your PSFT dataset, ensuring that the key r1 is retained in the parquet.


## Training

We provide the recipe within the verl framework.

## Evaluation

```
cd evaluation
serve run eval.llm:build_app model=aaa/bbb/ccc tensor-parallel-size=1

# open another terminal
python eval/eval.py --temperature 0.7 --top_p 0.95 --max_tokens 10240 --model ccc --test_file eval/data/aime-2024.parquet
```

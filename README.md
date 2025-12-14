# ConvertNet: Training-time Model Scaling for In-vehicle Continuous Learning

## Table of Contents

- [1 Introduction](#1-introduction)
- [2 Installation](#2-installation)
  - [2.1 Requirements](#21-requirements)
  - [2.2 Preparing Environment](#22-preparing-environment)
- [3 Running Example](#3-running-example)
  - [3.1 Settings](#31-settings)
  - [3.2 Pre-training](#32-pre-training)
  - [3.3 Retraining](#33-retraining)
- [4 Full Evaluation Results](#4-full-evaluation-results)
  - [4.1 Settings](#41-settings)
  - [4.2 Results](#42-results)

## 1 Introduction

Vehicles are transforming from transportation means to mobile intelligent terminals, powered by artificial intelligence (AI) technologies such as large language (LLM) and vision large models (VLM). To maintain high learning accuracy in all situations, in-vehicle AI models encounter the acute need to continuously learn and dynamically retrain AI models on the fly. The volatile resource demands of such stochastic retraining jobs and inference jobs of higher priority however can not be accommodated simultaneously by existing training systems, which rely on pre-generated compressed models of fixed architecture. In this paper, we propose ConvertNet, a novel continuous learning system that effectively scales up/down a compressed model through training-time neuron memorizer (TNM) - a max heap of neurons in tree structure. Based on TNM, ConvertNet estimates the model's resource-accuracy trade-off per neuron, thus efficiently utilizing the limited available resources to scale up and (re)train the model to improve its accuracy. Evaluated on six representative in-vehicle scenarios, comparative experiments against eleven state-of-the-art techniques show that ConvertNet achieves as much as 22.33\% improvement in learning accuracy, and reduces energy consumption by 3.65x.

## 2 Installation

### 2.1 Requirements

- Linux
- Python 3.8+
- PyTorch 1.9.0+
- CUDA 10.2+

### 2.2 Preparing Environment

First, create a conda virtual environment and activate it:

```bash
conda create -n convertnet python=3.8 -y
conda activate convertnet
```

Second, install torch and torchvision according to the [official instructions](https://pytorch.org/get-started/locally/):

![image](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ec360791671f4a4ab322eb4e71cc9e62~tplv-k3u1fbpfcp-zoom-1.image)

Finally, install the required packages:

```bash
pip install -r requirements.txt
```

## 3. Running Example 

### 3.1 Settings

**Model**. We use a vision transformer (ViT-B/16) model for image classification from HuggingFace.

**Dataset**. We use the dataset [SYNSIGNS](https://www.bing.com/ck/a?!&&p=dff6ea3b21ba57437144c9de249f78f6291cf6b155e637f0125ca3d5e97f2a74JmltdHM9MTc2NTY3MDQwMA&ptn=3&ver=2&hsh=4&fclid=2adb0aaa-c490-6599-2b13-1f9ec5d364fb&psq=SYNSIGNS+dataset&u=a1aHR0cHM6Ly96ZW5vZG8ub3JnL3JlY29yZHMvNjU5ODIyMi9maWxlcy9TeW50aGV0aWNfVHJhZmZpY19TaWduc19EYXRhc2V0X2Zvcl9UcmFmZmljX1NpZ25fRGV0ZWN0aW9uX2FuZF9SZWNvZ25pdGlvbl9Jbl9EaXN0cmlidXRlZF9TbWFydF9TeXN0ZW1zLnBkZg) as the distribution in pre-training, and the dataset [GTSRB](https://benchmark.ini.rub.de/?section=gtsrb) as the distribution in retraining.

### 3.2 Pre-training

Run the following command sequentially to pre-train the model, generate the samples for accuracy predictor construction, and pre-train the accuracy predictor:

```bash
python exp/offline_preparing/vit_b_16/img_cls/lora_fine_tune.py
python exp/offline_preparing/vit_b_16/img_cls/gen_knowledge_base.py
python exp/offline_preparing/vit_b_16/img_cls/gen_neuron_index.py
python exp/offline_preparing/vit_b_16/img_cls/gen_scaling_law_data_points.py
python analysis/scaling_law_trial/only_consider_pruned_block_index2.py
```

### 3.3 Retraining

Run the following command sequentially to retrain the model:

```bash
python exp/online_retraining/vit_b_16/img_cls/vit.py "[]"
python exp/online_retraining/vit_b_16/img_cls/vit.py "[(10, ('s', 1)), (20, ('s', 2))]"
```

The first argument of `vit.py` specifies the training-time scaling options. For example, `"[(10, ('s', 1)), (20, ('s', 2))]"` means scaling up the model from 1/8x to 2/8x at iteration 10, and scaling up the model from 2/8x to 3/8x at iteration 20.

After running the above commands, you can run the following command to visualize the retraining accuracy with and without training-time scaling:

```bash
python exp/online_retraining/vit_b_16/img_cls/draw.py
```

And you will see the following figure. The results show that training-time scaling can improve the retraining accuracy.

![](https://github.com/LINC-BIT/ConvertNet/blob/main/exp/online_retraining/vit_b_16/img_cls/accuracy_comparison.png)

## 4. Full Evaluation Results

### 4.1 Settings

**Testbeds**. We choose four mainstream in-vehicle GPU devices~\cite{agx-orin} with different architectures: (1) NVIDIA Xavier NX with 16GB memory and 384-core Volta GPU; (2) NVIDIA AGX Xavier with 32GB memory and 512-core Volta GPU; (3) NVIDIA AGX Orin with 32GB memory and 1792-core Ampere GPU; and (4) NVIDIA AGX Orin with 64GB memory and 2048-core Ampere GPU. They are equipped with Ubuntu 18.04.5, CUDA 10.2, and Python 3.8.5.

**Baselines**. We implement and compare ConvertNet with two types of eleven retraining techniques (static compression and adaptive retraining) on each workload.

**Workloads**. We choose six AI applications which target the most important continuous learning tasks on vehicles, and construct six distinct workloads based on six Hugging Face models. The details are listed in the table below.

![](https://github.com/LINC-BIT/ConvertNet/blob/main/full_eval_settings.png)

### 4.2 Results

![](https://github.com/LINC-BIT/ConvertNet/blob/main/full_eval_res.png)

Overall, ConvertNet improves the retraining accuracy by 19.37% over static compression techniques, 25.23% over adaptive retraining techniques, and 22.33% over all baselines.

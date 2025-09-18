# DKL

The official PyTorch implementation of the paper "Dual Knowledge Distillation Framework with Class-Adaptive Temperature and TopK Feature Perturbation for Few-Shot Prompt Learning".

# Overview
This repo contains the PyTorch implementation of DKL, described in the paper "Dual Knowledge Distillation Framework with Class-Adaptive Temperature and TopK Feature Perturbation for Few-Shot Prompt Learning". Due to the paper has not been accepted yet, we currently only release the test code and our trained models. The complete source code will be made available after the paper is accepted.


# Installation
This code is built upon [KgCoOp](https://github.com/htyao89/KgCoOp) . For environment setup and dataset preparation, please refer to the corresponding sections in their README.


# models of DKL
base2new: link: https://pan.baidu.com/s/1qZmaHLpQBEvd6Hy7LPVkYg?pwd=i7pt download code: i7pt

few_shot8: link: https://pan.baidu.com/s/1kCETdLS0lwuflbtiwZQsCg?pwd=px5n download code: px5n

few_shot16: link: https://pan.baidu.com/s/1PMrBGQ8-vVOYMKm9mKDY_A?pwd=1wm4 download code: 1wm4

After downloading and uncompressing the model, modify the model path in the corresponding sh file to your own directory.

# Test
for base: sh eval_base.sh $dataset $seed $gpu $model_epoch

for new: sh eval_new.sh $dataset $seed $gpu $model_epoch

for few-shot: sh eval_all.sh $dataset $seed $gpu $shot_num $model_epoch

# Acknowledgements
Our code is based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [KgCoOp](https://github.com/htyao89/KgCoOp) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.

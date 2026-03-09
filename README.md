# Overview
This repository contains the official PyTorch implementation of **DKL** (Dual Knowledge Distillation Framework with Class-Adaptive Temperature and TopK Feature Perturbation) for few-shot prompt learning in vision-language models.


If you find this code useful for your research, please consider citing our paper (see [Citation](#citation) section).


# Installation
This code is built upon [KgCoOp](https://github.com/htyao89/KgCoOp) . For environment setup and dataset preparation, please refer to the corresponding sections in their README.


# models of DKL
**base2new**: link: https://pan.baidu.com/s/1qZmaHLpQBEvd6Hy7LPVkYg?pwd=i7pt download code: i7pt

**few_shot4**: link: https://pan.baidu.com/s/1a6UGa6zTpEouNkI-inmRcQ?pwd=bpmi download code: bpmi

**few_shot8**: link: https://pan.baidu.com/s/1kCETdLS0lwuflbtiwZQsCg?pwd=px5n download code: px5n

**few_shot16**: link: https://pan.baidu.com/s/1PMrBGQ8-vVOYMKm9mKDY_A?pwd=1wm4 download code: 1wm4

Each compressed file in the links above contains a total of 33 models trained with 3 random seeds across all 11 datasets. After downloading and extracting these models, simply modify the model paths in the test script file to the actual paths.

# Train
**base2new**: sh main.sh $dataset $seed $gpu $shot_num

**few-shot**: sh main_all.sh $dataset $seed $gpu $shot_num

**Note**: Only the ImageNet dataset uses 50 epochs, while all others use 200.

# Test
**base2new_base**: sh eval_base.sh $dataset $seed $gpu $epoch

**base2new_new**: sh eval_new.sh $dataset $seed $gpu $epoch

**few-shot**: sh eval_all.sh $dataset $seed $gpu $shot_num $epoch



# Acknowledgements
Our code is based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [KgCoOp](https://github.com/htyao89/KgCoOp) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.

# Citation

If you use DKL in your research or find this repository helpful, please cite our TCSVT paper:

```bibtex
@article{chen2026dual,
  author={Chen, Wenjie and Li, Weisheng and Shu, Yucheng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Dual Knowledge Distillation Framework with Class-Adaptive Temperature and TopK Feature Perturbation for Few-Shot Prompt Learning}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2026.3662460}
}

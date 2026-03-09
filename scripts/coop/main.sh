#!/bin/bash

cd ../..

# custom config
DATA=/home/cwj/dataset
TRAINER=CoOp

DATASET=$1
#CFG=vit_b16_ep50_ctxv1  # config file
CFG=vit_b16_ep200_ctxv1  # config file
SEED=$2
GPU=$3

CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=$4
CSC=False  # class-specific context (False or True)



DIR=output_coop/base2new/train_base/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
  echo "The results exist at ${DIR}. Deleting..."
  rm -rf "$DIR"

  python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --gpu ${GPU} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --gpu ${GPU} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi

#!/bin/bash

cd ../..

# custom config
DATA=/home/cwj/dataset
TRAINER=CoOp
SHOTS=16
NCTX=4
#NCTX=16
CSC=False
CTP=end

DATASET=$1
SEED=$2
GPU=$3
EPOCH=$4
CFG=vit_b16_ep200_ctxv1
#CFG=rn50

COMMON_DIR=${DATASET}/seed${SEED}/
#MODEL_DIR=output_coop/base2new/train_base/${DATASET}/seed${SEED}/
MODEL_DIR=/home/cwj/coop/CoOp-main/testModel_inPaper/base2new/cwjv2/${DATASET}/seed${SEED}/
DIR=output_coop/base2new/test_new/${DATASET}/seed${SEED}


if [ -d "$DIR" ]; then
    echo "The results exist at ${DIR}. Deleting..."
    rm -rf "$DIR"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${EPOCH} \
    --eval-only \
    --gpu ${GPU} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.SUBSAMPLE_CLASSES new
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${EPOCH} \
    --eval-only \
    --gpu ${GPU} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.SUBSAMPLE_CLASSES new
fi
